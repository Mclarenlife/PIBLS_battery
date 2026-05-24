"""State-Augmented Gradient-Descent Broad Learning System for battery cycles.

This module implements a BLS variant for the BattNN single-cycle voltage
prediction task. The mapping and enhancement layers are fixed random feature
nodes. Only the linear output weights are optimized, using Adam instead of
the usual least-squares or pseudo-inverse solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch


DEFAULT_RC_DECAYS = (0.99, 0.98, 0.95, 0.90, 0.80, 0.65, 0.50)


def _as_1d_float(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def activation(values: np.ndarray, name: str) -> np.ndarray:
    """Apply one supported BLS activation function."""

    key = name.lower()
    if key == "sigmoid":
        clipped = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    if key == "tanh":
        return np.tanh(values)
    if key == "sin":
        return np.sin(values)
    if key == "softplus":
        return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)
    if key == "relu":
        return np.maximum(values, 0.0)
    if key == "linear":
        return values
    raise ValueError(
        f"Unsupported activation {name!r}. "
        "Choose from sigmoid, tanh, sin, softplus, relu, linear."
    )


def regression_metrics(true: Sequence[float], pred: Sequence[float]) -> dict[str, float]:
    """Return the same metrics used by the BattNN experiment code."""

    y_true = _as_1d_float(true, "true")
    y_pred = _as_1d_float(pred, "pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Metric shapes differ: {y_true.shape} != {y_pred.shape}.")
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-12))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }


@dataclass
class SAGDBLS:
    """State-augmented BLS trained by gradient descent on output weights."""

    n_map: int = 100
    n_enhance: int = 100
    map_activation: str = "sigmoid"
    enhance_activation: str = "sigmoid"
    seed: int = 1
    learning_rate: float = 0.01
    epochs: int = 20_000
    l2: float = 1e-3
    map_scale: float = 1.0
    enhance_scale: float = 1.0
    backend: str = "numpy"
    device: str = "auto"
    torch_dtype: str = "float32"
    smooth_l1_delta: float = 1.0
    rc_decays: tuple[float, ...] = field(default_factory=lambda: DEFAULT_RC_DECAYS)
    time_scale: float = 60.0
    charge_scale: float = 360.0
    current_scale: float = 8.0
    diff_scale: float = 6.0
    verbose: bool = False

    beta_: np.ndarray | None = field(init=False, default=None)
    train_length_: int | None = field(init=False, default=None)
    input_mean_: np.ndarray | None = field(init=False, default=None)
    input_std_: np.ndarray | None = field(init=False, default=None)
    design_mean_: np.ndarray | None = field(init=False, default=None)
    design_std_: np.ndarray | None = field(init=False, default=None)
    target_mean_: float | None = field(init=False, default=None)
    target_std_: float | None = field(init=False, default=None)
    context_dim_: int = field(init=False, default=0)
    W_map_: np.ndarray | None = field(init=False, default=None)
    b_map_: np.ndarray | None = field(init=False, default=None)
    W_enhance_: np.ndarray | None = field(init=False, default=None)
    b_enhance_: np.ndarray | None = field(init=False, default=None)
    training_summary_: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.n_map = int(self.n_map)
        self.n_enhance = int(self.n_enhance)
        self.epochs = int(self.epochs)
        if self.n_map <= 0 or self.n_enhance <= 0:
            raise ValueError("n_map and n_enhance must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative.")
        if self.map_scale <= 0 or self.enhance_scale <= 0:
            raise ValueError("map_scale and enhance_scale must be positive.")
        if self.backend not in {"numpy", "torch"}:
            raise ValueError("backend must be 'numpy' or 'torch'.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        if self.torch_dtype not in {"float32", "float64"}:
            raise ValueError("torch_dtype must be 'float32' or 'float64'.")
        if self.smooth_l1_delta <= 0:
            raise ValueError("smooth_l1_delta must be positive.")

    @staticmethod
    def state_features(
        current_sequence: Sequence[float] | np.ndarray,
        rc_decays: Sequence[float] = DEFAULT_RC_DECAYS,
        current_scale: float = 8.0,
        time_scale: float = 60.0,
        charge_scale: float = 360.0,
        diff_scale: float = 6.0,
    ) -> np.ndarray:
        """Build state coordinates from current only.

        The voltage sequence is intentionally not an input. This makes test-time
        prediction depend only on the current load profile.
        """

        current = _as_1d_float(current_sequence, "current_sequence")
        t = np.arange(current.size, dtype=np.float64)
        cumulative_current = np.cumsum(current)
        current_delta = np.concatenate([[0.0], np.diff(current)])

        columns = [
            current / current_scale,
            t / time_scale,
            cumulative_current / charge_scale,
            current_delta / diff_scale,
        ]
        for decay in rc_decays:
            filtered = np.empty_like(current)
            value = 0.0
            for idx, u in enumerate(current):
                value = float(decay) * value + (1.0 - float(decay)) * float(u)
                filtered[idx] = value
            columns.append(filtered / current_scale)
        return np.column_stack(columns)

    def fit(
        self,
        currents: Sequence[Sequence[float] | np.ndarray],
        voltages: Sequence[Sequence[float] | np.ndarray],
        train_length: int = 60,
        contexts: Sequence[Sequence[float] | np.ndarray | float | int] | None = None,
    ) -> "SAGDBLS":
        """Fit output weights using gradient descent on fixed BLS features."""

        if len(currents) != len(voltages):
            raise ValueError("currents and voltages must contain the same number of sequences.")
        if contexts is not None and len(contexts) != len(currents):
            raise ValueError("contexts must be None or contain one context per sequence.")
        train_length = int(train_length)
        if train_length <= 0:
            raise ValueError("train_length must be positive.")

        feature_blocks: list[np.ndarray] = []
        target_blocks: list[np.ndarray] = []
        time_indices: list[np.ndarray] = []
        if contexts is None:
            context_values = [None] * len(currents)
        else:
            context_values = list(contexts)

        self.context_dim_ = 0
        for current_seq, voltage_seq, context in zip(currents, voltages, context_values):
            current = _as_1d_float(current_seq, "current")
            voltage = _as_1d_float(voltage_seq, "voltage")
            usable = min(current.size, voltage.size)
            if usable < train_length:
                continue
            state = self.state_features(
                current[:train_length],
                rc_decays=self.rc_decays,
                current_scale=self.current_scale,
                time_scale=self.time_scale,
                charge_scale=self.charge_scale,
                diff_scale=self.diff_scale,
            )
            feature_blocks.append(self._append_context(state, context))
            target_blocks.append(voltage[:train_length])
            time_indices.append(np.arange(train_length, dtype=np.int64))

        if not feature_blocks:
            raise ValueError("No training sequences are at least train_length long.")

        self.train_length_ = train_length
        x_train = np.vstack(feature_blocks)
        y_train = np.concatenate(target_blocks)
        step_indices = np.concatenate(time_indices)

        self.input_mean_ = x_train.mean(axis=0)
        self.input_std_ = x_train.std(axis=0) + 1e-8
        x_scaled = (x_train - self.input_mean_) / self.input_std_

        self._initialize_random_layers(x_scaled.shape[1])
        design = self._raw_design_matrix(x_scaled)
        self.design_mean_ = design[:, 1:].mean(axis=0)
        self.design_std_ = design[:, 1:].std(axis=0) + 1e-8
        h_train = self._scale_design_matrix(design)

        self.target_mean_ = float(y_train.mean())
        self.target_std_ = float(y_train.std() + 1e-8)
        y_scaled = (y_train - self.target_mean_) / self.target_std_

        weights = np.linspace(train_length / 5.0, 1.0, train_length)[step_indices]
        weights = weights / weights.mean()

        self.beta_ = self._adam_output_fit(h_train, y_scaled, weights)
        self.training_summary_ = {
            "initial_loss": self._objective(h_train, y_scaled, weights, np.zeros(h_train.shape[1])),
            "final_loss": self._objective(h_train, y_scaled, weights, self.beta_),
            "train_sequences": float(len(feature_blocks)),
            "train_points": float(h_train.shape[0]),
        }
        return self

    def predict(
        self,
        current_sequence: Sequence[float] | np.ndarray,
        context: Sequence[float] | np.ndarray | float | int | None = None,
    ) -> np.ndarray:
        """Predict the full voltage curve for a current sequence."""

        self._require_fitted()
        state = self.state_features(
            current_sequence,
            rc_decays=self.rc_decays,
            current_scale=self.current_scale,
            time_scale=self.time_scale,
            charge_scale=self.charge_scale,
            diff_scale=self.diff_scale,
        )
        x = self._append_context(state, context, fitted=True)
        assert self.input_mean_ is not None and self.input_std_ is not None
        x_scaled = (x - self.input_mean_) / self.input_std_
        design = self._scale_design_matrix(self._raw_design_matrix(x_scaled))
        assert self.beta_ is not None and self.target_mean_ is not None and self.target_std_ is not None
        return (design @ self.beta_) * self.target_std_ + self.target_mean_

    def evaluate(
        self,
        test_iter: Iterable[tuple],
    ) -> dict[str, float]:
        """Evaluate on an iterator of current and voltage sequences."""

        rows = []
        for item in test_iter:
            if len(item) == 2:
                current, voltage = item
                context = None
            elif len(item) == 3:
                current, voltage, context = item
            else:
                raise ValueError("Each test item must be (current, voltage) or (current, voltage, context).")
            pred = self.predict(current, context=context)
            true_voltage = _as_1d_float(voltage, "voltage")
            if pred.size != true_voltage.size:
                raise ValueError(
                    f"Prediction length {pred.size} differs from voltage length {true_voltage.size}."
                )
            rows.append(regression_metrics(true_voltage, pred))
        if not rows:
            raise ValueError("test_iter produced no sequences.")
        keys = ("MAE", "MAPE", "MSE", "RMSE")
        return {key: float(np.mean([row[key] for row in rows])) for key in keys} | {
            "n_sequences": float(len(rows))
        }

    def _initialize_random_layers(self, n_features: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.W_map_ = rng.normal(0.0, self.map_scale / np.sqrt(n_features), size=(n_features, self.n_map))
        self.b_map_ = rng.uniform(-1.0, 1.0, size=self.n_map)
        self.W_enhance_ = rng.normal(
            0.0,
            self.enhance_scale / np.sqrt(self.n_map),
            size=(self.n_map, self.n_enhance),
        )
        self.b_enhance_ = rng.uniform(-1.0, 1.0, size=self.n_enhance)

    def _raw_design_matrix(self, x_scaled: np.ndarray) -> np.ndarray:
        assert self.W_map_ is not None and self.b_map_ is not None
        assert self.W_enhance_ is not None and self.b_enhance_ is not None
        h_map = activation(x_scaled @ self.W_map_ + self.b_map_, self.map_activation)
        h_enhance = activation(h_map @ self.W_enhance_ + self.b_enhance_, self.enhance_activation)
        return np.column_stack([np.ones(x_scaled.shape[0]), x_scaled, h_map, h_enhance])

    def _scale_design_matrix(self, design: np.ndarray) -> np.ndarray:
        assert self.design_mean_ is not None and self.design_std_ is not None
        return np.column_stack([np.ones(design.shape[0]), (design[:, 1:] - self.design_mean_) / self.design_std_])

    def _adam_output_fit(self, h_train: np.ndarray, y_scaled: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if self.backend == "torch":
            return self._adam_output_fit_torch(h_train, y_scaled, weights)

        beta = np.zeros(h_train.shape[1], dtype=np.float64)
        first_moment = np.zeros_like(beta)
        second_moment = np.zeros_like(beta)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        n = float(y_scaled.size)

        for epoch in range(1, self.epochs + 1):
            pred = h_train @ beta
            residual = pred - y_scaled
            abs_residual = np.abs(residual)
            loss_grad = np.where(
                abs_residual < self.smooth_l1_delta,
                residual / self.smooth_l1_delta,
                np.sign(residual),
            )
            grad = h_train.T @ (weights * loss_grad) / n
            grad += self.l2 * beta

            first_moment = beta1 * first_moment + (1.0 - beta1) * grad
            second_moment = beta2 * second_moment + (1.0 - beta2) * (grad * grad)
            m_hat = first_moment / (1.0 - beta1**epoch)
            v_hat = second_moment / (1.0 - beta2**epoch)
            beta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            if self.verbose and (epoch == 1 or epoch % 1000 == 0 or epoch == self.epochs):
                loss = self._objective(h_train, y_scaled, weights, beta)
                print(f"epoch={epoch:6d} loss={loss:.8f}")
        return beta

    def _adam_output_fit_torch(self, h_train: np.ndarray, y_scaled: np.ndarray, weights: np.ndarray) -> np.ndarray:
        device = self._torch_device()
        dtype = torch.float32 if self.torch_dtype == "float32" else torch.float64
        h = torch.as_tensor(h_train, device=device, dtype=dtype)
        y = torch.as_tensor(y_scaled, device=device, dtype=dtype)
        w = torch.as_tensor(weights, device=device, dtype=dtype)
        beta = torch.zeros(h.shape[1], device=device, dtype=dtype)
        first_moment = torch.zeros_like(beta)
        second_moment = torch.zeros_like(beta)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        n = float(y_scaled.size)

        for epoch in range(1, self.epochs + 1):
            pred = h @ beta
            residual = pred - y
            abs_residual = torch.abs(residual)
            loss_grad = torch.where(
                abs_residual < self.smooth_l1_delta,
                residual / self.smooth_l1_delta,
                torch.sign(residual),
            )
            grad = h.T @ (w * loss_grad) / n
            grad = grad + self.l2 * beta

            first_moment = beta1 * first_moment + (1.0 - beta1) * grad
            second_moment = beta2 * second_moment + (1.0 - beta2) * (grad * grad)
            m_hat = first_moment / (1.0 - beta1**epoch)
            v_hat = second_moment / (1.0 - beta2**epoch)
            beta = beta - self.learning_rate * m_hat / (torch.sqrt(v_hat) + eps)

            if self.verbose and (epoch == 1 or epoch % 1000 == 0 or epoch == self.epochs):
                loss = self._objective(h_train, y_scaled, weights, beta.detach().cpu().numpy())
                print(f"epoch={epoch:6d} loss={loss:.8f}")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return beta.detach().cpu().numpy().astype(np.float64)

    def _torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this Python environment has CPU-only torch.")
        return torch.device(self.device)

    def _objective(
        self,
        h_train: np.ndarray,
        y_scaled: np.ndarray,
        weights: np.ndarray,
        beta: np.ndarray,
    ) -> float:
        residual = h_train @ beta - y_scaled
        abs_residual = np.abs(residual)
        smooth_l1 = np.where(
            abs_residual < self.smooth_l1_delta,
            0.5 * residual**2 / self.smooth_l1_delta,
            abs_residual - 0.5 * self.smooth_l1_delta,
        )
        return float(np.mean(weights * smooth_l1) + 0.5 * self.l2 * np.dot(beta, beta))

    def _require_fitted(self) -> None:
        if self.beta_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")

    def _append_context(
        self,
        state: np.ndarray,
        context: Sequence[float] | np.ndarray | float | int | None,
        fitted: bool = False,
    ) -> np.ndarray:
        if context is None:
            if fitted and self.context_dim_:
                raise ValueError("This model was fitted with context; predict/evaluate needs context.")
            return state

        context_array = np.asarray(context, dtype=np.float64).reshape(-1)
        if context_array.size == 0:
            raise ValueError("context must not be empty.")
        if not np.all(np.isfinite(context_array)):
            raise ValueError("context contains non-finite values.")

        if self.context_dim_ == 0 and not fitted:
            self.context_dim_ = int(context_array.size)
        elif context_array.size != self.context_dim_:
            raise ValueError(
                f"Context dimension {context_array.size} does not match fitted dimension {self.context_dim_}."
            )
        return np.column_stack([state, np.tile(context_array, (state.shape[0], 1))])
