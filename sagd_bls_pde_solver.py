"""General SAGD-BLS solver prototype for PDE residual training.

This file starts the method line that is separate from the battery adapter.
It contains no battery-specific current, cycle, or RC-state features. The
unknown field is represented by fixed broad random features and a single
trainable linear output layer.

Current demo:
    1D Poisson equation
        -u''(x) = pi^2 sin(pi x), x in [0, 1]
        u(0) = u(1) = 0
    exact solution:
        u(x) = sin(pi x)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


RESULTS_CSV = Path("results") / "sagd_bls_pde_poisson_demo.csv"


def _as_column(values: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def activation_with_derivatives(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return activation, first derivative, and second derivative."""

    key = name.lower()
    if key == "tanh":
        y = np.tanh(values)
        dy = 1.0 - y * y
        d2y = -2.0 * y * dy
        return y, dy, d2y
    if key == "sigmoid":
        clipped = np.clip(values, -60.0, 60.0)
        y = 1.0 / (1.0 + np.exp(-clipped))
        dy = y * (1.0 - y)
        d2y = dy * (1.0 - 2.0 * y)
        return y, dy, d2y
    if key == "sin":
        return np.sin(values), np.cos(values), -np.sin(values)
    if key == "softplus":
        clipped = np.clip(values, -60.0, 60.0)
        sigmoid = 1.0 / (1.0 + np.exp(-clipped))
        y = np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)
        return y, sigmoid, sigmoid * (1.0 - sigmoid)
    if key == "linear":
        return values, np.ones_like(values), np.zeros_like(values)
    raise ValueError("Supported activations for PDE derivatives: tanh, sigmoid, sin, softplus, linear.")


def regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(true, dtype=np.float64).reshape(-1) - np.asarray(pred, dtype=np.float64).reshape(-1)
    mse = float(np.mean(err**2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(mse)),
        "MAXAE": float(np.max(np.abs(err))),
    }


@dataclass
class SAGDBLSPDESolver1D:
    """Fixed-feature BLS PDE solver that trains only output weights beta."""

    n_map: int = 120
    n_enhance: int = 120
    map_activation: str = "tanh"
    enhance_activation: str = "tanh"
    seed: int = 1
    learning_rate: float = 3e-4
    epochs: int = 80_000
    l2: float = 1e-6
    residual_weight: float = 1.0
    boundary_weight: float = 100.0
    verbose: bool = False

    beta_: np.ndarray | None = field(init=False, default=None)
    input_mean_: np.ndarray | None = field(init=False, default=None)
    input_std_: np.ndarray | None = field(init=False, default=None)
    design_mean_: np.ndarray | None = field(init=False, default=None)
    design_std_: np.ndarray | None = field(init=False, default=None)
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

    def fit_poisson(
        self,
        interior_x: np.ndarray,
        source_fn: Callable[[np.ndarray], np.ndarray],
        boundary_x: np.ndarray,
        boundary_u: np.ndarray,
    ) -> "SAGDBLSPDESolver1D":
        """Fit beta for a 1D Poisson problem: -u_xx = source_fn(x)."""

        x_int = _as_column(interior_x, "interior_x")
        x_bnd = _as_column(boundary_x, "boundary_x")
        u_bnd = np.asarray(boundary_u, dtype=np.float64).reshape(-1)
        if x_bnd.shape[0] != u_bnd.size:
            raise ValueError("boundary_x and boundary_u must have the same length.")

        self._fit_feature_scalers(np.vstack([x_int, x_bnd]))
        phi_int, _, phi_xx_int = self.design_and_derivatives(x_int)
        phi_bnd, _, _ = self.design_and_derivatives(x_bnd)
        source = np.asarray(source_fn(x_int.reshape(-1)), dtype=np.float64).reshape(-1)
        if source.size != x_int.shape[0]:
            raise ValueError("source_fn must return one value per interior point.")

        self.beta_ = self._adam_poisson_fit(phi_int, phi_xx_int, source, phi_bnd, u_bnd)
        self.training_summary_ = self.poisson_objective(phi_int, phi_xx_int, source, phi_bnd, u_bnd, self.beta_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._require_fitted()
        phi, _, _ = self.design_and_derivatives(_as_column(x, "x"))
        assert self.beta_ is not None
        return phi @ self.beta_

    def residual(self, x: np.ndarray, source_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        self._require_fitted()
        x_col = _as_column(x, "x")
        _, _, phi_xx = self.design_and_derivatives(x_col)
        assert self.beta_ is not None
        return -(phi_xx @ self.beta_) - np.asarray(source_fn(x_col.reshape(-1)), dtype=np.float64).reshape(-1)

    def evaluate_poisson(
        self,
        exact_fn: Callable[[np.ndarray], np.ndarray],
        source_fn: Callable[[np.ndarray], np.ndarray],
        eval_x: np.ndarray,
        boundary_x: np.ndarray,
        boundary_u: np.ndarray,
    ) -> dict[str, float]:
        x_eval = _as_column(eval_x, "eval_x").reshape(-1)
        pred = self.predict(x_eval)
        exact = np.asarray(exact_fn(x_eval), dtype=np.float64).reshape(-1)
        metrics = regression_metrics(exact, pred)
        residual = self.residual(x_eval, source_fn)
        b_pred = self.predict(boundary_x)
        b_true = np.asarray(boundary_u, dtype=np.float64).reshape(-1)
        metrics.update(
            {
                "residual_RMSE": float(np.sqrt(np.mean(residual**2))),
                "boundary_MAXAE": float(np.max(np.abs(b_pred - b_true))),
                "n_eval": float(x_eval.size),
            }
        )
        return metrics

    def _fit_feature_scalers(self, x: np.ndarray) -> None:
        self.input_mean_ = x.mean(axis=0)
        self.input_std_ = x.std(axis=0) + 1e-8
        rng = np.random.default_rng(self.seed)
        n_features = x.shape[1]
        self.W_map_ = rng.normal(0.0, 1.0 / np.sqrt(n_features), size=(n_features, self.n_map))
        self.b_map_ = rng.uniform(-1.0, 1.0, size=self.n_map)
        self.W_enhance_ = rng.normal(0.0, 1.0 / np.sqrt(self.n_map), size=(self.n_map, self.n_enhance))
        self.b_enhance_ = rng.uniform(-1.0, 1.0, size=self.n_enhance)

        raw_design, _, _ = self.raw_design_and_derivatives(x)
        self.design_mean_ = raw_design[:, 1:].mean(axis=0)
        self.design_std_ = raw_design[:, 1:].std(axis=0) + 1e-8

    def design_and_derivatives(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_design, raw_dx, raw_dxx = self.raw_design_and_derivatives(x)
        assert self.design_mean_ is not None and self.design_std_ is not None
        design = np.column_stack([np.ones(x.shape[0]), (raw_design[:, 1:] - self.design_mean_) / self.design_std_])
        design_x = np.column_stack([np.zeros(x.shape[0]), raw_dx[:, 1:] / self.design_std_])
        design_xx = np.column_stack([np.zeros(x.shape[0]), raw_dxx[:, 1:] / self.design_std_])
        return design, design_x, design_xx

    def raw_design_and_derivatives(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.input_mean_ is not None and self.input_std_ is not None
        assert self.W_map_ is not None and self.b_map_ is not None
        assert self.W_enhance_ is not None and self.b_enhance_ is not None

        x_scaled = (x - self.input_mean_) / self.input_std_
        x_dx = np.zeros_like(x_scaled)
        x_dx[:, 0] = 1.0 / self.input_std_[0]
        x_dxx = np.zeros_like(x_scaled)

        a_map = x_scaled @ self.W_map_ + self.b_map_
        map_value, map_prime, map_second = activation_with_derivatives(a_map, self.map_activation)
        a_map_dx = x_dx @ self.W_map_
        a_map_dxx = x_dxx @ self.W_map_
        map_dx = map_prime * a_map_dx
        map_dxx = map_second * (a_map_dx * a_map_dx) + map_prime * a_map_dxx

        a_enhance = map_value @ self.W_enhance_ + self.b_enhance_
        enhance_value, enhance_prime, enhance_second = activation_with_derivatives(
            a_enhance,
            self.enhance_activation,
        )
        a_enhance_dx = map_dx @ self.W_enhance_
        a_enhance_dxx = map_dxx @ self.W_enhance_
        enhance_dx = enhance_prime * a_enhance_dx
        enhance_dxx = enhance_second * (a_enhance_dx * a_enhance_dx) + enhance_prime * a_enhance_dxx

        raw_design = np.column_stack([np.ones(x.shape[0]), x_scaled, map_value, enhance_value])
        raw_dx = np.column_stack([np.zeros(x.shape[0]), x_dx, map_dx, enhance_dx])
        raw_dxx = np.column_stack([np.zeros(x.shape[0]), x_dxx, map_dxx, enhance_dxx])
        return raw_design, raw_dx, raw_dxx

    def _adam_poisson_fit(
        self,
        phi_int: np.ndarray,
        phi_xx_int: np.ndarray,
        source: np.ndarray,
        phi_bnd: np.ndarray,
        boundary_u: np.ndarray,
    ) -> np.ndarray:
        beta = np.zeros(phi_int.shape[1], dtype=np.float64)
        first_moment = np.zeros_like(beta)
        second_moment = np.zeros_like(beta)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        residual_operator = -phi_xx_int
        n_int = float(phi_int.shape[0])
        n_bnd = float(phi_bnd.shape[0])

        for epoch in range(1, self.epochs + 1):
            residual = residual_operator @ beta - source
            boundary_residual = phi_bnd @ beta - boundary_u
            grad = self.residual_weight * (2.0 / n_int) * (residual_operator.T @ residual)
            grad += self.boundary_weight * (2.0 / n_bnd) * (phi_bnd.T @ boundary_residual)
            grad += self.l2 * beta

            first_moment = beta1 * first_moment + (1.0 - beta1) * grad
            second_moment = beta2 * second_moment + (1.0 - beta2) * (grad * grad)
            m_hat = first_moment / (1.0 - beta1**epoch)
            v_hat = second_moment / (1.0 - beta2**epoch)
            beta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            if self.verbose and (epoch == 1 or epoch % 1000 == 0 or epoch == self.epochs):
                summary = self.poisson_objective(phi_int, phi_xx_int, source, phi_bnd, boundary_u, beta)
                print(
                    f"epoch={epoch:6d} loss={summary['loss']:.8e} "
                    f"residual={summary['residual_loss']:.8e} boundary={summary['boundary_loss']:.8e}"
                )
        return beta

    def poisson_objective(
        self,
        phi_int: np.ndarray,
        phi_xx_int: np.ndarray,
        source: np.ndarray,
        phi_bnd: np.ndarray,
        boundary_u: np.ndarray,
        beta: np.ndarray,
    ) -> dict[str, float]:
        residual = -(phi_xx_int @ beta) - source
        boundary_residual = phi_bnd @ beta - boundary_u
        residual_loss = float(np.mean(residual**2))
        boundary_loss = float(np.mean(boundary_residual**2))
        regularization = float(0.5 * self.l2 * np.dot(beta, beta))
        loss = self.residual_weight * residual_loss + self.boundary_weight * boundary_loss + regularization
        return {
            "loss": float(loss),
            "residual_loss": residual_loss,
            "boundary_loss": boundary_loss,
            "regularization": regularization,
        }

    def _require_fitted(self) -> None:
        if self.beta_ is None:
            raise ValueError("Model is not fitted. Call fit_poisson() first.")


def poisson_source(x: np.ndarray) -> np.ndarray:
    return (np.pi**2) * np.sin(np.pi * x)


def poisson_exact(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAGD-BLS PDE solver research script.")
    parser.add_argument("--demo", choices=["poisson"], default="poisson")
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-map", type=int, default=120)
    parser.add_argument("--n-enhance", type=int, default=120)
    parser.add_argument("--activation", nargs=2, default=["tanh", "tanh"], metavar=("MAP", "ENHANCE"))
    parser.add_argument("--epochs", type=int, default=80_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--boundary-weight", type=float, default=100.0)
    parser.add_argument("--n-interior", type=int, default=128)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_poisson_demo(args: argparse.Namespace) -> dict[str, object]:
    interior_x = np.linspace(0.0, 1.0, args.n_interior + 2, dtype=np.float64)[1:-1]
    boundary_x = np.array([0.0, 1.0], dtype=np.float64)
    boundary_u = np.array([0.0, 0.0], dtype=np.float64)
    eval_x = np.linspace(0.0, 1.0, args.n_eval, dtype=np.float64)

    model = SAGDBLSPDESolver1D(
        n_map=args.n_map,
        n_enhance=args.n_enhance,
        map_activation=args.activation[0],
        enhance_activation=args.activation[1],
        seed=args.seed,
        learning_rate=args.lr,
        epochs=args.epochs,
        l2=args.l2,
        boundary_weight=args.boundary_weight,
        verbose=args.verbose,
    )
    model.fit_poisson(interior_x, poisson_source, boundary_x, boundary_u)
    metrics = model.evaluate_poisson(poisson_exact, poisson_source, eval_x, boundary_x, boundary_u)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": "SAGD-BLS-PDE",
        "demo": "1D Poisson",
        "equation": "-u_xx=pi^2 sin(pi x), u(0)=u(1)=0",
        "seed": args.seed,
        "n_map": args.n_map,
        "n_enhance": args.n_enhance,
        "map_activation": args.activation[0],
        "enhance_activation": args.activation[1],
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "l2": args.l2,
        "boundary_weight": args.boundary_weight,
        "n_interior": args.n_interior,
        "n_eval": args.n_eval,
        **model.training_summary_,
        **metrics,
    }


def save_row(row: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = sorted(row.keys())
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.demo != "poisson":
        raise ValueError(f"Unsupported demo: {args.demo}")
    row = run_poisson_demo(args)
    save_row(row, args.results_csv)
    print(
        "SAGD-BLS PDE demo complete: "
        f"MAE={row['MAE']:.6e}, RMSE={row['RMSE']:.6e}, MAXAE={row['MAXAE']:.6e}, "
        f"residual_RMSE={row['residual_RMSE']:.6e}, boundary_MAXAE={row['boundary_MAXAE']:.6e}"
    )
    print(f"saved results: {args.results_csv}")


if __name__ == "__main__":
    main()
