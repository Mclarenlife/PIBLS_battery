"""Standard unforced Burgers experiment with a finite-difference reference.

Problem:
    u_t + u u_x - nu u_xx = 0, x in [-1, 1], t in [0, 1]
    u(x, 0) = -sin(pi x)
    u(-1, t) = u(1, t) = 0

This is the standard PINN-style Burgers benchmark. Unlike the forced Burgers
sanity check, no source term is manufactured from the exact solution. A
finite-difference method-of-lines reference is generated with scipy.solve_ivp.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from run_burgers_experiment import BroadFeature2D, PINN2D
from sagd_bls_pde_solver import regression_metrics


RESULTS_CSV = Path("results") / "standard_burgers_single_seed_results.csv"
SUMMARY_CSV = Path("results") / "standard_burgers_single_seed_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard unforced Burgers baseline comparison.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--nu", type=float, default=0.01 / np.pi)
    parser.add_argument("--n-map", type=int, default=160)
    parser.add_argument("--n-enhance", type=int, default=160)
    parser.add_argument("--activation", nargs=2, default=["tanh", "tanh"], metavar=("MAP", "ENHANCE"))
    parser.add_argument("--n-collocation", type=int, default=1200)
    parser.add_argument("--n-initial", type=int, default=160)
    parser.add_argument("--n-boundary", type=int, default=160)
    parser.add_argument("--n-ref-x", type=int, default=257)
    parser.add_argument("--n-ref-t", type=int, default=101)
    parser.add_argument("--reference-method", type=str, default="BDF")
    parser.add_argument("--sagd-epochs", type=int, default=40_000)
    parser.add_argument("--sagd-lr", type=float, default=5e-4)
    parser.add_argument("--sagd-l2", type=float, default=1e-6)
    parser.add_argument("--hard-sagd-epochs", type=int, default=15_000)
    parser.add_argument("--hard-sagd-lr", type=float, default=1.5e-3)
    parser.add_argument("--hard-trial", choices=["decay", "stationary"], default="stationary")
    parser.add_argument("--ic-weight", type=float, default=100.0)
    parser.add_argument("--bc-weight", type=float, default=100.0)
    parser.add_argument("--pibls-ridge", type=float, default=1e-10)
    parser.add_argument("--pinn-epochs", type=int, default=5_000)
    parser.add_argument("--pinn-lr", type=float, default=1e-3)
    parser.add_argument("--pinn-hidden", type=int, default=64)
    parser.add_argument("--pinn-depth", type=int, default=4)
    parser.add_argument("--pinn-l2", type=float, default=0.0)
    parser.add_argument("--skip-soft-sagd", action="store_true")
    parser.add_argument("--skip-hard-sagd", action="store_true")
    parser.add_argument("--skip-pibls", action="store_true")
    parser.add_argument("--skip-pinn", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args()


def initial_condition(x: np.ndarray) -> np.ndarray:
    return -np.sin(np.pi * x)


def solve_reference(args: argparse.Namespace) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, args.n_ref_x, dtype=np.float64)
    t_eval = np.linspace(0.0, 1.0, args.n_ref_t, dtype=np.float64)
    dx = float(x[1] - x[0])
    y0_full = initial_condition(x)
    y0_full[0] = 0.0
    y0_full[-1] = 0.0
    y0 = y0_full[1:-1]

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        u = np.zeros(args.n_ref_x, dtype=np.float64)
        u[1:-1] = y
        flux = 0.5 * u * u
        wave_speed = np.maximum(np.abs(u[:-1]), np.abs(u[1:]))
        interface_flux = 0.5 * (flux[:-1] + flux[1:]) - 0.5 * wave_speed * (u[1:] - u[:-1])
        convection = -(interface_flux[1:] - interface_flux[:-1]) / dx
        u_xx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
        return convection + args.nu * u_xx

    solution = solve_ivp(
        rhs,
        (0.0, 1.0),
        y0,
        method=args.reference_method,
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(f"Reference solve failed: {solution.message}")

    u = np.zeros((args.n_ref_t, args.n_ref_x), dtype=np.float64)
    u[:, 1:-1] = solution.y.T
    x_mesh, t_mesh = np.meshgrid(x, t_eval, indexing="xy")
    return {
        "x_ref": x,
        "t_ref": t_eval,
        "u_ref": u,
        "x_eval": x_mesh.reshape(-1),
        "t_eval": t_mesh.reshape(-1),
        "u_eval": u.reshape(-1),
        "reference_steps": np.array([solution.nfev], dtype=np.float64),
    }


def make_points(args: argparse.Namespace, seed: int, reference: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_col = rng.uniform(-1.0, 1.0, size=args.n_collocation)
    t_col = rng.uniform(0.0, 1.0, size=args.n_collocation)

    x_ic = np.linspace(-1.0, 1.0, args.n_initial, dtype=np.float64)
    t_ic = np.zeros_like(x_ic)
    u_ic = initial_condition(x_ic)

    t_side = np.linspace(0.0, 1.0, args.n_boundary, dtype=np.float64)
    x_bc = np.concatenate([np.full_like(t_side, -1.0), np.full_like(t_side, 1.0)])
    t_bc = np.concatenate([t_side, t_side])
    u_bc = np.zeros_like(x_bc)

    return {
        "x_col": x_col,
        "t_col": t_col,
        "x_ic": x_ic,
        "t_ic": t_ic,
        "u_ic": u_ic,
        "x_bc": x_bc,
        "t_bc": t_bc,
        "u_bc": u_bc,
        **reference,
    }


@dataclass
class SAGDBLSStandardBurgers:
    features: BroadFeature2D
    nu: float
    learning_rate: float
    epochs: int
    l2: float
    ic_weight: float
    bc_weight: float

    beta_: np.ndarray | None = field(init=False, default=None)
    training_summary_: dict[str, float] = field(init=False, default_factory=dict)

    def fit(self, points: dict[str, np.ndarray]) -> "SAGDBLSStandardBurgers":
        self.features.fit(
            np.concatenate([points["x_col"], points["x_ic"], points["x_bc"]]),
            np.concatenate([points["t_col"], points["t_ic"], points["t_bc"]]),
        )
        phi, phi_x, phi_t, phi_xx = self.features.design_and_derivatives(points["x_col"], points["t_col"])
        phi_ic, _, _, _ = self.features.design_and_derivatives(points["x_ic"], points["t_ic"])
        phi_bc, _, _, _ = self.features.design_and_derivatives(points["x_bc"], points["t_bc"])
        self.beta_ = self._adam_fit(phi, phi_x, phi_t, phi_xx, phi_ic, points["u_ic"], phi_bc, points["u_bc"])
        self.training_summary_ = self.objective(phi, phi_x, phi_t, phi_xx, phi_ic, points["u_ic"], phi_bc, points["u_bc"], self.beta_)
        return self

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            raise ValueError("Model is not fitted.")
        phi, _, _, _ = self.features.design_and_derivatives(x, t)
        return phi @ self.beta_

    def residual(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            raise ValueError("Model is not fitted.")
        phi, phi_x, phi_t, phi_xx = self.features.design_and_derivatives(x, t)
        u = phi @ self.beta_
        return (phi_t @ self.beta_) + u * (phi_x @ self.beta_) - self.nu * (phi_xx @ self.beta_)

    def _adam_fit(
        self,
        phi: np.ndarray,
        phi_x: np.ndarray,
        phi_t: np.ndarray,
        phi_xx: np.ndarray,
        phi_ic: np.ndarray,
        u_ic: np.ndarray,
        phi_bc: np.ndarray,
        u_bc: np.ndarray,
    ) -> np.ndarray:
        beta = np.zeros(phi.shape[1], dtype=np.float64)
        m = np.zeros_like(beta)
        v = np.zeros_like(beta)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        n_col = float(phi.shape[0])
        n_ic = float(phi_ic.shape[0])
        n_bc = float(phi_bc.shape[0])
        for epoch in range(1, self.epochs + 1):
            u = phi @ beta
            residual = (phi_t @ beta) + u * (phi_x @ beta) - self.nu * (phi_xx @ beta)
            jacobian = phi_t + (phi_x @ beta)[:, None] * phi + u[:, None] * phi_x - self.nu * phi_xx
            grad = (2.0 / n_col) * (jacobian.T @ residual)
            ic_res = phi_ic @ beta - u_ic
            bc_res = phi_bc @ beta - u_bc
            grad += self.ic_weight * (2.0 / n_ic) * (phi_ic.T @ ic_res)
            grad += self.bc_weight * (2.0 / n_bc) * (phi_bc.T @ bc_res)
            grad += self.l2 * beta

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            beta -= self.learning_rate * (m / (1.0 - beta1**epoch)) / (np.sqrt(v / (1.0 - beta2**epoch)) + eps)
        return beta

    def objective(
        self,
        phi: np.ndarray,
        phi_x: np.ndarray,
        phi_t: np.ndarray,
        phi_xx: np.ndarray,
        phi_ic: np.ndarray,
        u_ic: np.ndarray,
        phi_bc: np.ndarray,
        u_bc: np.ndarray,
        beta: np.ndarray,
    ) -> dict[str, float]:
        u = phi @ beta
        residual = (phi_t @ beta) + u * (phi_x @ beta) - self.nu * (phi_xx @ beta)
        ic_res = phi_ic @ beta - u_ic
        bc_res = phi_bc @ beta - u_bc
        residual_loss = float(np.mean(residual**2))
        ic_loss = float(np.mean(ic_res**2))
        bc_loss = float(np.mean(bc_res**2))
        regularization = float(0.5 * self.l2 * np.dot(beta, beta))
        return {
            "loss": residual_loss + self.ic_weight * ic_loss + self.bc_weight * bc_loss + regularization,
            "residual_loss": residual_loss,
            "ic_loss": ic_loss,
            "bc_loss": bc_loss,
            "regularization": regularization,
        }


def trial_components(x: np.ndarray, t: np.ndarray, hard_trial: str) -> dict[str, np.ndarray]:
    """Analytic IC/BC envelope for standard Burgers."""

    u0 = initial_condition(x)
    u0_x = -np.pi * np.cos(np.pi * x)
    u0_xx = (np.pi**2) * np.sin(np.pi * x)
    if hard_trial == "decay":
        base = (1.0 - t) * u0
        base_x = (1.0 - t) * u0_x
        base_t = -u0
        base_xx = (1.0 - t) * u0_xx
    elif hard_trial == "stationary":
        base = u0
        base_x = u0_x
        base_t = np.zeros_like(u0)
        base_xx = u0_xx
    else:
        raise ValueError("hard_trial must be 'decay' or 'stationary'.")
    return {
        "base": base,
        "base_x": base_x,
        "base_t": base_t,
        "base_xx": base_xx,
        "envelope": t * (1.0 - x * x),
        "envelope_x": -2.0 * t * x,
        "envelope_t": 1.0 - x * x,
        "envelope_xx": -2.0 * t,
    }


@dataclass
class SAGDBLSHardICBCStandardBurgers:
    features: BroadFeature2D
    nu: float
    learning_rate: float
    epochs: int
    l2: float
    hard_trial: str

    beta_: np.ndarray | None = field(init=False, default=None)
    training_summary_: dict[str, float] = field(init=False, default_factory=dict)

    def fit(self, points: dict[str, np.ndarray]) -> "SAGDBLSHardICBCStandardBurgers":
        self.features.fit(
            np.concatenate([points["x_col"], points["x_ic"], points["x_bc"]]),
            np.concatenate([points["t_col"], points["t_ic"], points["t_bc"]]),
        )
        phi, phi_x, phi_t, phi_xx = self.features.design_and_derivatives(points["x_col"], points["t_col"])
        self.beta_ = self._adam_fit(points["x_col"], points["t_col"], phi, phi_x, phi_t, phi_xx)
        self.training_summary_ = self.objective(
            points["x_col"],
            points["t_col"],
            phi,
            phi_x,
            phi_t,
            phi_xx,
            points,
            self.beta_,
        )
        return self

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            raise ValueError("Model is not fitted.")
        phi, _, _, _ = self.features.design_and_derivatives(x, t)
        return self._u_and_derivatives(x, t, phi, None, None, None, self.beta_)["u"]

    def residual(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            raise ValueError("Model is not fitted.")
        phi, phi_x, phi_t, phi_xx = self.features.design_and_derivatives(x, t)
        values = self._u_and_derivatives(x, t, phi, phi_x, phi_t, phi_xx, self.beta_)
        return values["u_t"] + values["u"] * values["u_x"] - self.nu * values["u_xx"]

    def _adam_fit(
        self,
        x: np.ndarray,
        t: np.ndarray,
        phi: np.ndarray,
        phi_x: np.ndarray,
        phi_t: np.ndarray,
        phi_xx: np.ndarray,
    ) -> np.ndarray:
        beta = np.zeros(phi.shape[1], dtype=np.float64)
        m = np.zeros_like(beta)
        v = np.zeros_like(beta)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        n_col = float(phi.shape[0])
        for epoch in range(1, self.epochs + 1):
            values = self._u_and_derivatives(x, t, phi, phi_x, phi_t, phi_xx, beta)
            residual = values["u_t"] + values["u"] * values["u_x"] - self.nu * values["u_xx"]
            jacobian = (
                values["du_t"]
                + values["u_x"][:, None] * values["du"]
                + values["u"][:, None] * values["du_x"]
                - self.nu * values["du_xx"]
            )
            grad = (2.0 / n_col) * (jacobian.T @ residual)
            grad += self.l2 * beta

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            beta -= self.learning_rate * (m / (1.0 - beta1**epoch)) / (np.sqrt(v / (1.0 - beta2**epoch)) + eps)
        return beta

    def objective(
        self,
        x: np.ndarray,
        t: np.ndarray,
        phi: np.ndarray,
        phi_x: np.ndarray,
        phi_t: np.ndarray,
        phi_xx: np.ndarray,
        points: dict[str, np.ndarray],
        beta: np.ndarray,
    ) -> dict[str, float]:
        values = self._u_and_derivatives(x, t, phi, phi_x, phi_t, phi_xx, beta)
        residual = values["u_t"] + values["u"] * values["u_x"] - self.nu * values["u_xx"]
        phi_ic, _, _, _ = self.features.design_and_derivatives(points["x_ic"], points["t_ic"])
        phi_bc, _, _, _ = self.features.design_and_derivatives(points["x_bc"], points["t_bc"])
        ic_pred = self._u_and_derivatives(points["x_ic"], points["t_ic"], phi_ic, None, None, None, beta)["u"]
        bc_pred = self._u_and_derivatives(points["x_bc"], points["t_bc"], phi_bc, None, None, None, beta)["u"]
        ic_res = ic_pred - points["u_ic"]
        bc_res = bc_pred - points["u_bc"]
        residual_loss = float(np.mean(residual**2))
        ic_loss = float(np.mean(ic_res**2))
        bc_loss = float(np.mean(bc_res**2))
        regularization = float(0.5 * self.l2 * np.dot(beta, beta))
        return {
            "loss": residual_loss + regularization,
            "residual_loss": residual_loss,
            "ic_loss": ic_loss,
            "bc_loss": bc_loss,
            "regularization": regularization,
        }

    def _u_and_derivatives(
        self,
        x: np.ndarray,
        t: np.ndarray,
        phi: np.ndarray,
        phi_x: np.ndarray | None,
        phi_t: np.ndarray | None,
        phi_xx: np.ndarray | None,
        beta: np.ndarray,
    ) -> dict[str, np.ndarray]:
        terms = trial_components(x, t, self.hard_trial)
        v = phi @ beta
        u = terms["base"] + terms["envelope"] * v
        values: dict[str, np.ndarray] = {"u": u}
        if phi_x is None or phi_t is None or phi_xx is None:
            return values

        v_x = phi_x @ beta
        v_t = phi_t @ beta
        v_xx = phi_xx @ beta
        u_x = terms["base_x"] + terms["envelope_x"] * v + terms["envelope"] * v_x
        u_t = terms["base_t"] + terms["envelope_t"] * v + terms["envelope"] * v_t
        u_xx = (
            terms["base_xx"]
            + terms["envelope_xx"] * v
            + 2.0 * terms["envelope_x"] * v_x
            + terms["envelope"] * v_xx
        )
        values.update(
            {
                "u_x": u_x,
                "u_t": u_t,
                "u_xx": u_xx,
                "du": terms["envelope"][:, None] * phi,
                "du_x": terms["envelope_x"][:, None] * phi + terms["envelope"][:, None] * phi_x,
                "du_t": terms["envelope_t"][:, None] * phi + terms["envelope"][:, None] * phi_t,
                "du_xx": (
                    terms["envelope_xx"][:, None] * phi
                    + 2.0 * terms["envelope_x"][:, None] * phi_x
                    + terms["envelope"][:, None] * phi_xx
                ),
            }
        )
        return values


def evaluate_model(
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    residual_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    points: dict[str, np.ndarray],
) -> dict[str, float]:
    pred = predict_fn(points["x_eval"], points["t_eval"])
    metrics = regression_metrics(points["u_eval"], pred)
    residual = residual_fn(points["x_eval"], points["t_eval"])
    ic_pred = predict_fn(points["x_ic"], points["t_ic"])
    bc_pred = predict_fn(points["x_bc"], points["t_bc"])
    metrics.update(
        {
            "residual_RMSE": float(np.sqrt(np.mean(residual**2))),
            "ic_MAXAE": float(np.max(np.abs(ic_pred - points["u_ic"]))),
            "bc_MAXAE": float(np.max(np.abs(bc_pred - points["u_bc"]))),
            "n_eval": float(points["x_eval"].size),
        }
    )
    return metrics


def run_sagd(args: argparse.Namespace, seed: int, points: dict[str, np.ndarray]) -> dict[str, object]:
    start = time.perf_counter()
    feature = BroadFeature2D(args.n_map, args.n_enhance, args.activation[0], args.activation[1], seed)
    model = SAGDBLSStandardBurgers(feature, args.nu, args.sagd_lr, args.sagd_epochs, args.sagd_l2, args.ic_weight, args.bc_weight)
    model.fit(points)
    metrics = evaluate_model(model.predict, model.residual, points)
    return make_row(args, "SAGD-BLS-standard-Burgers", seed, time.perf_counter() - start, feature.width, model.training_summary_, metrics)


def run_sagd_hard_icbc(args: argparse.Namespace, seed: int, points: dict[str, np.ndarray]) -> dict[str, object]:
    start = time.perf_counter()
    feature = BroadFeature2D(args.n_map, args.n_enhance, args.activation[0], args.activation[1], seed)
    model = SAGDBLSHardICBCStandardBurgers(
        feature,
        args.nu,
        args.hard_sagd_lr,
        args.hard_sagd_epochs,
        args.sagd_l2,
        args.hard_trial,
    )
    model.fit(points)
    metrics = evaluate_model(model.predict, model.residual, points)
    method = "SAGD-BLS-hard-ICBC" if args.hard_trial == "decay" else "SAGD-BLS-hard-ICBC-stationary"
    return make_row(args, method, seed, time.perf_counter() - start, feature.width, model.training_summary_, metrics)


def run_pibls_linearized(args: argparse.Namespace, seed: int, points: dict[str, np.ndarray]) -> dict[str, object]:
    start = time.perf_counter()
    feature = BroadFeature2D(args.n_map, args.n_enhance, args.activation[0], args.activation[1], seed)
    feature.fit(
        np.concatenate([points["x_col"], points["x_ic"], points["x_bc"]]),
        np.concatenate([points["t_col"], points["t_ic"], points["t_bc"]]),
    )
    _, _, phi_t, phi_xx = feature.design_and_derivatives(points["x_col"], points["t_col"])
    phi_ic, _, _, _ = feature.design_and_derivatives(points["x_ic"], points["t_ic"])
    phi_bc, _, _, _ = feature.design_and_derivatives(points["x_bc"], points["t_bc"])
    a = np.vstack(
        [
            phi_t - args.nu * phi_xx,
            np.sqrt(args.ic_weight) * phi_ic,
            np.sqrt(args.bc_weight) * phi_bc,
        ]
    )
    b = np.concatenate(
        [
            np.zeros(points["x_col"].size, dtype=np.float64),
            np.sqrt(args.ic_weight) * points["u_ic"],
            np.sqrt(args.bc_weight) * points["u_bc"],
        ]
    )
    beta = np.linalg.solve(a.T @ a + args.pibls_ridge * np.eye(a.shape[1]), a.T @ b)

    def predict(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        phi_eval, _, _, _ = feature.design_and_derivatives(x, t)
        return phi_eval @ beta

    def residual(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        phi_eval, phi_x, phi_t_eval, phi_xx_eval = feature.design_and_derivatives(x, t)
        u = phi_eval @ beta
        return (phi_t_eval @ beta) + u * (phi_x @ beta) - args.nu * (phi_xx_eval @ beta)

    metrics = evaluate_model(predict, residual, points)
    summary = {"loss": "", "residual_loss": "", "ic_loss": "", "bc_loss": "", "regularization": ""}
    return make_row(args, "PIBLS-linearized-pinv", seed, time.perf_counter() - start, feature.width, summary, metrics)


class PINNWrapper(nn.Module):
    def __init__(self, hidden: int, depth: int) -> None:
        super().__init__()
        self.net = PINN2D(hidden, depth)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def pinn_residual(model: nn.Module, x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    z = torch.cat([x, t], dim=1).detach().clone().requires_grad_(True)
    u = model(z)
    grad = torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True)[0]
    u_x = grad[:, 0:1]
    u_t = grad[:, 1:2]
    grad_x = torch.autograd.grad(u_x, z, torch.ones_like(u_x), create_graph=True)[0]
    u_xx = grad_x[:, 0:1]
    return u_t + u * u_x - nu * u_xx


def run_pinn(args: argparse.Namespace, seed: int, points: dict[str, np.ndarray]) -> dict[str, object]:
    start = time.perf_counter()
    torch.manual_seed(seed)
    dtype = torch.float64
    model = PINNWrapper(args.pinn_hidden, args.pinn_depth).to(dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pinn_lr, weight_decay=args.pinn_l2)
    x_col = torch.tensor(points["x_col"].reshape(-1, 1), dtype=dtype)
    t_col = torch.tensor(points["t_col"].reshape(-1, 1), dtype=dtype)
    z_ic = torch.tensor(np.column_stack([points["x_ic"], points["t_ic"]]), dtype=dtype)
    u_ic = torch.tensor(points["u_ic"].reshape(-1, 1), dtype=dtype)
    z_bc = torch.tensor(np.column_stack([points["x_bc"], points["t_bc"]]), dtype=dtype)
    u_bc = torch.tensor(points["u_bc"].reshape(-1, 1), dtype=dtype)

    initial_loss = None
    final_loss = None
    for epoch in range(1, args.pinn_epochs + 1):
        optimizer.zero_grad()
        residual = pinn_residual(model, x_col, t_col, args.nu)
        ic_res = model(z_ic) - u_ic
        bc_res = model(z_bc) - u_bc
        loss = torch.mean(residual**2) + args.ic_weight * torch.mean(ic_res**2) + args.bc_weight * torch.mean(bc_res**2)
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        if epoch == args.pinn_epochs:
            final_loss = float(loss.detach().cpu())

    def predict(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.tensor(np.column_stack([x, t]), dtype=dtype)
            return model(z).detach().cpu().numpy().reshape(-1)

    def residual_fn(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x.reshape(-1, 1), dtype=dtype)
        t_tensor = torch.tensor(t.reshape(-1, 1), dtype=dtype)
        return pinn_residual(model, x_tensor, t_tensor, args.nu).detach().cpu().numpy().reshape(-1)

    metrics = evaluate_model(predict, residual_fn, points)
    parameter_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    summary = {
        "loss": final_loss,
        "residual_loss": "",
        "ic_loss": "",
        "bc_loss": "",
        "regularization": "",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
    }
    return make_row(args, "PINN-standard-Burgers", seed, time.perf_counter() - start, parameter_count, summary, metrics)


def make_row(
    args: argparse.Namespace,
    method: str,
    seed: int,
    runtime: float,
    trainable_parameters: int,
    training_summary: dict[str, object],
    metrics: dict[str, float],
) -> dict[str, object]:
    is_bls = "BLS" in method or "PIBLS" in method
    is_sagd = method == "SAGD-BLS-standard-Burgers"
    is_hard_sagd = method.startswith("SAGD-BLS-hard-ICBC")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "problem": "Standard unforced Burgers",
        "method": method,
        "seed": seed,
        "nu": args.nu,
        "n_map": args.n_map if is_bls else "",
        "n_enhance": args.n_enhance if is_bls else "",
        "map_activation": args.activation[0] if is_bls else "tanh",
        "enhance_activation": args.activation[1] if is_bls else "",
        "hard_trial": args.hard_trial if is_hard_sagd else "",
        "n_collocation": args.n_collocation,
        "n_initial": args.n_initial,
        "n_boundary": args.n_boundary,
        "n_eval": metrics["n_eval"],
        "reference_grid": f"{args.n_ref_x}x{args.n_ref_t}",
        "epochs": args.sagd_epochs if is_sagd else (args.hard_sagd_epochs if is_hard_sagd else (args.pinn_epochs if method == "PINN-standard-Burgers" else 0)),
        "learning_rate": args.sagd_lr if is_sagd else (args.hard_sagd_lr if is_hard_sagd else (args.pinn_lr if method == "PINN-standard-Burgers" else "")),
        "l2": args.sagd_l2 if is_sagd or is_hard_sagd else (args.pinn_l2 if method == "PINN-standard-Burgers" else args.pibls_ridge),
        "ic_weight": args.ic_weight,
        "bc_weight": args.bc_weight,
        "trainable_parameters": trainable_parameters,
        "runtime_sec": runtime,
    }
    row.update(training_summary)
    row.update(metrics)
    return row


def save_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        out: dict[str, object] = {"problem": "Standard unforced Burgers", "method": method, "n": len(subset)}
        for metric in ("MAE", "RMSE", "MAXAE", "residual_RMSE", "ic_MAXAE", "bc_MAXAE", "runtime_sec"):
            values = np.array([float(row[metric]) for row in subset], dtype=np.float64)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std())
        out["trainable_parameters"] = subset[0]["trainable_parameters"]
        summary_rows.append(out)
    return summary_rows


def print_summary(summary_rows: list[dict[str, object]]) -> None:
    print("Standard unforced Burgers summary:")
    for row in summary_rows:
        print(
            f"{row['method']}: "
            f"MAE={row['MAE_mean']:.6e}, RMSE={row['RMSE_mean']:.6e}, "
            f"residual_RMSE={row['residual_RMSE_mean']:.6e}, runtime={row['runtime_sec_mean']:.3f}s"
        )


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    reference = solve_reference(args)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        points = make_points(args, seed, reference)
        if not args.skip_soft_sagd:
            sagd = run_sagd(args, seed, points)
            rows.append(sagd)
            print(f"SAGD-BLS-standard-Burgers seed={seed}: MAE={sagd['MAE']:.6e}, RMSE={sagd['RMSE']:.6e}")

        if not args.skip_hard_sagd:
            sagd_hard = run_sagd_hard_icbc(args, seed, points)
            rows.append(sagd_hard)
            print(f"{sagd_hard['method']} seed={seed}: MAE={sagd_hard['MAE']:.6e}, RMSE={sagd_hard['RMSE']:.6e}")

        if not args.skip_pibls:
            pibls = run_pibls_linearized(args, seed, points)
            rows.append(pibls)
            print(f"PIBLS-linearized-pinv seed={seed}: MAE={pibls['MAE']:.6e}, RMSE={pibls['RMSE']:.6e}")

        if not args.skip_pinn:
            pinn = run_pinn(args, seed, points)
            rows.append(pinn)
            print(f"PINN-standard-Burgers seed={seed}: MAE={pinn['MAE']:.6e}, RMSE={pinn['RMSE']:.6e}")

    summary_rows = summarize(rows)
    save_rows(rows, args.results_csv)
    save_rows(summary_rows, args.summary_csv)
    print_summary(summary_rows)
    print(f"saved results: {args.results_csv}")
    print(f"saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
