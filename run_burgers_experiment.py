"""Forced Burgers experiment for the general SAGD-BLS PDE solver line.

Problem:
    u_t + u u_x - nu u_xx = f(x, t), x in [-1, 1], t in [0, 1]

Manufactured exact solution:
    u(x, t) = -sin(pi x) exp(-t)

This gives zero Dirichlet boundaries at x=-1 and x=1, and a known initial
condition at t=0. The source f is chosen from the exact solution.

Baselines:
    SAGD-BLS-Burgers:
        Fixed broad features; Adam trains only output beta on the nonlinear
        Burgers residual plus IC/BC losses.
    PIBLS-linearized-pinv:
        A PIBLS-style pseudo-inverse baseline on the linearized residual
        u_t - nu u_xx = f. This is intentionally labeled linearized because
        the original pinv-style PIBLS system is directly linear only for linear
        PDEs; Burgers' u u_x term makes the residual nonlinear in beta.
    PINN-Burgers:
        Tanh MLP trained with PyTorch autograd on the full nonlinear residual.
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

from sagd_bls_pde_solver import activation_with_derivatives, regression_metrics


RESULTS_CSV = Path("results") / "burgers_baseline_results.csv"
SUMMARY_CSV = Path("results") / "burgers_baseline_summary.csv"


def exact_solution(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -np.sin(np.pi * x) * np.exp(-t)


def burgers_source(x: np.ndarray, t: np.ndarray, nu: float) -> np.ndarray:
    sin = np.sin(np.pi * x)
    cos = np.cos(np.pi * x)
    exp_t = np.exp(-t)
    u_t = sin * exp_t
    u_ux = np.pi * sin * cos * np.exp(-2.0 * t)
    u_xx = (np.pi**2) * sin * exp_t
    return u_t + u_ux - nu * u_xx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forced Burgers baseline comparison.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--nu", type=float, default=0.01 / np.pi)
    parser.add_argument("--n-map", type=int, default=160)
    parser.add_argument("--n-enhance", type=int, default=160)
    parser.add_argument("--activation", nargs=2, default=["tanh", "tanh"], metavar=("MAP", "ENHANCE"))
    parser.add_argument("--n-collocation", type=int, default=1200)
    parser.add_argument("--n-initial", type=int, default=160)
    parser.add_argument("--n-boundary", type=int, default=160)
    parser.add_argument("--n-eval-x", type=int, default=201)
    parser.add_argument("--n-eval-t", type=int, default=101)
    parser.add_argument("--sagd-epochs", type=int, default=40_000)
    parser.add_argument("--sagd-lr", type=float, default=5e-4)
    parser.add_argument("--sagd-l2", type=float, default=1e-6)
    parser.add_argument("--ic-weight", type=float, default=100.0)
    parser.add_argument("--bc-weight", type=float, default=100.0)
    parser.add_argument("--pibls-ridge", type=float, default=1e-10)
    parser.add_argument("--pinn-epochs", type=int, default=5_000)
    parser.add_argument("--pinn-lr", type=float, default=1e-3)
    parser.add_argument("--pinn-hidden", type=int, default=64)
    parser.add_argument("--pinn-depth", type=int, default=4)
    parser.add_argument("--pinn-l2", type=float, default=0.0)
    parser.add_argument("--skip-pinn", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args()


def make_points(args: argparse.Namespace, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_col = rng.uniform(-1.0, 1.0, size=args.n_collocation)
    t_col = rng.uniform(0.0, 1.0, size=args.n_collocation)

    x_ic = np.linspace(-1.0, 1.0, args.n_initial)
    t_ic = np.zeros_like(x_ic)
    u_ic = exact_solution(x_ic, t_ic)

    t_side = np.linspace(0.0, 1.0, args.n_boundary)
    x_bc = np.concatenate([np.full_like(t_side, -1.0), np.full_like(t_side, 1.0)])
    t_bc = np.concatenate([t_side, t_side])
    u_bc = exact_solution(x_bc, t_bc)

    x_eval = np.linspace(-1.0, 1.0, args.n_eval_x)
    t_eval = np.linspace(0.0, 1.0, args.n_eval_t)
    x_mesh, t_mesh = np.meshgrid(x_eval, t_eval, indexing="ij")

    return {
        "x_col": x_col,
        "t_col": t_col,
        "x_ic": x_ic,
        "t_ic": t_ic,
        "u_ic": u_ic,
        "x_bc": x_bc,
        "t_bc": t_bc,
        "u_bc": u_bc,
        "x_eval": x_mesh.reshape(-1),
        "t_eval": t_mesh.reshape(-1),
    }


@dataclass
class BroadFeature2D:
    n_map: int
    n_enhance: int
    map_activation: str
    enhance_activation: str
    seed: int
    map_scale: float = 1.0
    enhance_scale: float = 1.0

    input_mean_: np.ndarray | None = field(init=False, default=None)
    input_std_: np.ndarray | None = field(init=False, default=None)
    design_mean_: np.ndarray | None = field(init=False, default=None)
    design_std_: np.ndarray | None = field(init=False, default=None)
    W_map_: np.ndarray | None = field(init=False, default=None)
    b_map_: np.ndarray | None = field(init=False, default=None)
    W_enhance_: np.ndarray | None = field(init=False, default=None)
    b_enhance_: np.ndarray | None = field(init=False, default=None)

    def fit(self, x: np.ndarray, t: np.ndarray) -> "BroadFeature2D":
        z = np.column_stack([x, t])
        self.input_mean_ = z.mean(axis=0)
        self.input_std_ = z.std(axis=0) + 1e-8
        rng = np.random.default_rng(self.seed)
        self.W_map_ = rng.normal(0.0, self.map_scale / np.sqrt(2), size=(2, self.n_map))
        self.b_map_ = rng.uniform(-1.0, 1.0, size=self.n_map)
        self.W_enhance_ = rng.normal(
            0.0,
            self.enhance_scale / np.sqrt(self.n_map),
            size=(self.n_map, self.n_enhance),
        )
        self.b_enhance_ = rng.uniform(-1.0, 1.0, size=self.n_enhance)
        raw, _, _, _ = self.raw_design_and_derivatives(x, t)
        self.design_mean_ = raw[:, 1:].mean(axis=0)
        self.design_std_ = raw[:, 1:].std(axis=0) + 1e-8
        return self

    @property
    def width(self) -> int:
        return 3 + self.n_map + self.n_enhance

    def design_and_derivatives(
        self,
        x: np.ndarray,
        t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw, raw_x, raw_t, raw_xx = self.raw_design_and_derivatives(x, t)
        assert self.design_mean_ is not None and self.design_std_ is not None
        phi = np.column_stack([np.ones(raw.shape[0]), (raw[:, 1:] - self.design_mean_) / self.design_std_])
        phi_x = np.column_stack([np.zeros(raw.shape[0]), raw_x[:, 1:] / self.design_std_])
        phi_t = np.column_stack([np.zeros(raw.shape[0]), raw_t[:, 1:] / self.design_std_])
        phi_xx = np.column_stack([np.zeros(raw.shape[0]), raw_xx[:, 1:] / self.design_std_])
        return phi, phi_x, phi_t, phi_xx

    def raw_design_and_derivatives(
        self,
        x: np.ndarray,
        t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.input_mean_ is not None and self.input_std_ is not None
        assert self.W_map_ is not None and self.b_map_ is not None
        assert self.W_enhance_ is not None and self.b_enhance_ is not None
        z = np.column_stack([x, t])
        z_scaled = (z - self.input_mean_) / self.input_std_

        z_x = np.zeros_like(z_scaled)
        z_t = np.zeros_like(z_scaled)
        z_x[:, 0] = 1.0 / self.input_std_[0]
        z_t[:, 1] = 1.0 / self.input_std_[1]

        a_map = z_scaled @ self.W_map_ + self.b_map_
        h_map, h_map_prime, h_map_second = activation_with_derivatives(a_map, self.map_activation)
        a_map_x = z_x @ self.W_map_
        a_map_t = z_t @ self.W_map_
        h_map_x = h_map_prime * a_map_x
        h_map_t = h_map_prime * a_map_t
        h_map_xx = h_map_second * (a_map_x * a_map_x)

        a_enhance = h_map @ self.W_enhance_ + self.b_enhance_
        h_enhance, h_enhance_prime, h_enhance_second = activation_with_derivatives(
            a_enhance,
            self.enhance_activation,
        )
        a_enhance_x = h_map_x @ self.W_enhance_
        a_enhance_t = h_map_t @ self.W_enhance_
        a_enhance_xx = h_map_xx @ self.W_enhance_
        h_enhance_x = h_enhance_prime * a_enhance_x
        h_enhance_t = h_enhance_prime * a_enhance_t
        h_enhance_xx = h_enhance_second * (a_enhance_x * a_enhance_x) + h_enhance_prime * a_enhance_xx

        raw = np.column_stack([np.ones_like(x), z_scaled, h_map, h_enhance])
        raw_x = np.column_stack([np.zeros_like(x), z_x, h_map_x, h_enhance_x])
        raw_t = np.column_stack([np.zeros_like(x), z_t, h_map_t, h_enhance_t])
        raw_xx = np.column_stack([np.zeros_like(x), np.zeros_like(z_scaled), h_map_xx, h_enhance_xx])
        return raw, raw_x, raw_t, raw_xx


@dataclass
class SAGDBLSBurgers:
    features: BroadFeature2D
    nu: float
    learning_rate: float
    epochs: int
    l2: float
    ic_weight: float
    bc_weight: float

    beta_: np.ndarray | None = field(init=False, default=None)
    training_summary_: dict[str, float] = field(init=False, default_factory=dict)

    def fit(self, points: dict[str, np.ndarray]) -> "SAGDBLSBurgers":
        self.features.fit(
            np.concatenate([points["x_col"], points["x_ic"], points["x_bc"]]),
            np.concatenate([points["t_col"], points["t_ic"], points["t_bc"]]),
        )
        phi, phi_x, phi_t, phi_xx = self.features.design_and_derivatives(points["x_col"], points["t_col"])
        phi_ic, _, _, _ = self.features.design_and_derivatives(points["x_ic"], points["t_ic"])
        phi_bc, _, _, _ = self.features.design_and_derivatives(points["x_bc"], points["t_bc"])
        source = burgers_source(points["x_col"], points["t_col"], self.nu)
        self.beta_ = self._adam_fit(phi, phi_x, phi_t, phi_xx, source, phi_ic, points["u_ic"], phi_bc, points["u_bc"])
        self.training_summary_ = self.objective(phi, phi_x, phi_t, phi_xx, source, phi_ic, points["u_ic"], phi_bc, points["u_bc"], self.beta_)
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
        u_x = phi_x @ self.beta_
        u_t = phi_t @ self.beta_
        u_xx = phi_xx @ self.beta_
        return u_t + u * u_x - self.nu * u_xx - burgers_source(x, t, self.nu)

    def _adam_fit(
        self,
        phi: np.ndarray,
        phi_x: np.ndarray,
        phi_t: np.ndarray,
        phi_xx: np.ndarray,
        source: np.ndarray,
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
            u_x = phi_x @ beta
            u_t = phi_t @ beta
            u_xx = phi_xx @ beta
            residual = u_t + u * u_x - self.nu * u_xx - source
            jacobian = phi_t + u_x[:, None] * phi + u[:, None] * phi_x - self.nu * phi_xx
            grad = (2.0 / n_col) * (jacobian.T @ residual)

            ic_res = phi_ic @ beta - u_ic
            bc_res = phi_bc @ beta - u_bc
            grad += self.ic_weight * (2.0 / n_ic) * (phi_ic.T @ ic_res)
            grad += self.bc_weight * (2.0 / n_bc) * (phi_bc.T @ bc_res)
            grad += self.l2 * beta

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            m_hat = m / (1.0 - beta1**epoch)
            v_hat = v / (1.0 - beta2**epoch)
            beta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        return beta

    def objective(
        self,
        phi: np.ndarray,
        phi_x: np.ndarray,
        phi_t: np.ndarray,
        phi_xx: np.ndarray,
        source: np.ndarray,
        phi_ic: np.ndarray,
        u_ic: np.ndarray,
        phi_bc: np.ndarray,
        u_bc: np.ndarray,
        beta: np.ndarray,
    ) -> dict[str, float]:
        u = phi @ beta
        residual = (phi_t @ beta) + u * (phi_x @ beta) - self.nu * (phi_xx @ beta) - source
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


def evaluate_model(
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    residual_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    points: dict[str, np.ndarray],
) -> dict[str, float]:
    pred = predict_fn(points["x_eval"], points["t_eval"])
    true = exact_solution(points["x_eval"], points["t_eval"])
    metrics = regression_metrics(true, pred)
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
    model = SAGDBLSBurgers(feature, args.nu, args.sagd_lr, args.sagd_epochs, args.sagd_l2, args.ic_weight, args.bc_weight)
    model.fit(points)
    metrics = evaluate_model(model.predict, model.residual, points)
    return make_row(args, "SAGD-BLS-Burgers", seed, time.perf_counter() - start, feature.width, model.training_summary_, metrics)


def run_pibls_linearized(args: argparse.Namespace, seed: int, points: dict[str, np.ndarray]) -> dict[str, object]:
    start = time.perf_counter()
    feature = BroadFeature2D(args.n_map, args.n_enhance, args.activation[0], args.activation[1], seed)
    feature.fit(
        np.concatenate([points["x_col"], points["x_ic"], points["x_bc"]]),
        np.concatenate([points["t_col"], points["t_ic"], points["t_bc"]]),
    )
    phi, _, phi_t, phi_xx = feature.design_and_derivatives(points["x_col"], points["t_col"])
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
            burgers_source(points["x_col"], points["t_col"], args.nu),
            np.sqrt(args.ic_weight) * points["u_ic"],
            np.sqrt(args.bc_weight) * points["u_bc"],
        ]
    )
    if args.pibls_ridge > 0:
        beta = np.linalg.solve(a.T @ a + args.pibls_ridge * np.eye(a.shape[1]), a.T @ b)
    else:
        beta = np.linalg.pinv(a) @ b

    def predict(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        phi_eval, _, _, _ = feature.design_and_derivatives(x, t)
        return phi_eval @ beta

    def residual(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        phi_eval, phi_x_eval, phi_t_eval, phi_xx_eval = feature.design_and_derivatives(x, t)
        u = phi_eval @ beta
        return (phi_t_eval @ beta) + u * (phi_x_eval @ beta) - args.nu * (phi_xx_eval @ beta) - burgers_source(x, t, args.nu)

    metrics = evaluate_model(predict, residual, points)
    summary = {"loss": "", "residual_loss": "", "ic_loss": "", "bc_loss": "", "regularization": ""}
    return make_row(args, "PIBLS-linearized-pinv", seed, time.perf_counter() - start, feature.width, summary, metrics)


class PINN2D(nn.Module):
    def __init__(self, hidden: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

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
    x_req = z[:, 0:1]
    t_req = z[:, 1:2]
    source = torch_source(x_req, t_req, nu)
    return u_t + u * u_x - nu * u_xx - source


def torch_exact(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return -torch.sin(torch.pi * x) * torch.exp(-t)


def torch_source(x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    sin = torch.sin(torch.pi * x)
    cos = torch.cos(torch.pi * x)
    exp_t = torch.exp(-t)
    return sin * exp_t + torch.pi * sin * cos * torch.exp(-2.0 * t) - nu * (torch.pi**2) * sin * exp_t


def run_pinn(args: argparse.Namespace, seed: int, points: dict[str, np.ndarray]) -> dict[str, object]:
    start = time.perf_counter()
    torch.manual_seed(seed)
    dtype = torch.float64
    model = PINN2D(args.pinn_hidden, args.pinn_depth).to(dtype)
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
    return make_row(args, "PINN-Burgers", seed, time.perf_counter() - start, parameter_count, summary, metrics)


def make_row(
    args: argparse.Namespace,
    method: str,
    seed: int,
    runtime: float,
    trainable_parameters: int,
    training_summary: dict[str, object],
    metrics: dict[str, float],
) -> dict[str, object]:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "problem": "Forced Burgers",
        "method": method,
        "seed": seed,
        "nu": args.nu,
        "n_map": args.n_map if "BLS" in method or "PIBLS" in method else "",
        "n_enhance": args.n_enhance if "BLS" in method or "PIBLS" in method else "",
        "map_activation": args.activation[0] if "BLS" in method or "PIBLS" in method else "tanh",
        "enhance_activation": args.activation[1] if "BLS" in method or "PIBLS" in method else "",
        "n_collocation": args.n_collocation,
        "n_initial": args.n_initial,
        "n_boundary": args.n_boundary,
        "n_eval": float(args.n_eval_x * args.n_eval_t),
        "epochs": args.sagd_epochs if method == "SAGD-BLS-Burgers" else (args.pinn_epochs if method == "PINN-Burgers" else 0),
        "learning_rate": args.sagd_lr if method == "SAGD-BLS-Burgers" else (args.pinn_lr if method == "PINN-Burgers" else ""),
        "l2": args.sagd_l2 if method == "SAGD-BLS-Burgers" else (args.pinn_l2 if method == "PINN-Burgers" else args.pibls_ridge),
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
        out: dict[str, object] = {"problem": "Forced Burgers", "method": method, "n": len(subset)}
        for metric in ("MAE", "RMSE", "MAXAE", "residual_RMSE", "ic_MAXAE", "bc_MAXAE", "runtime_sec"):
            values = np.array([float(row[metric]) for row in subset], dtype=np.float64)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std())
        out["trainable_parameters"] = subset[0]["trainable_parameters"]
        summary_rows.append(out)
    return summary_rows


def print_summary(summary_rows: list[dict[str, object]]) -> None:
    print("Forced Burgers summary:")
    for row in summary_rows:
        print(
            f"{row['method']}: "
            f"MAE={row['MAE_mean']:.6e} +/- {row['MAE_std']:.2e}, "
            f"RMSE={row['RMSE_mean']:.6e} +/- {row['RMSE_std']:.2e}, "
            f"residual_RMSE={row['residual_RMSE_mean']:.6e} +/- {row['residual_RMSE_std']:.2e}, "
            f"runtime={row['runtime_sec_mean']:.3f}s"
        )


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        points = make_points(args, seed)
        sagd = run_sagd(args, seed, points)
        rows.append(sagd)
        print(f"SAGD-BLS-Burgers seed={seed}: MAE={sagd['MAE']:.6e}, RMSE={sagd['RMSE']:.6e}")

        pibls = run_pibls_linearized(args, seed, points)
        rows.append(pibls)
        print(f"PIBLS-linearized-pinv seed={seed}: MAE={pibls['MAE']:.6e}, RMSE={pibls['RMSE']:.6e}")

        if not args.skip_pinn:
            pinn = run_pinn(args, seed, points)
            rows.append(pinn)
            print(f"PINN-Burgers seed={seed}: MAE={pinn['MAE']:.6e}, RMSE={pinn['RMSE']:.6e}")

    summary_rows = summarize(rows)
    save_rows(rows, args.results_csv)
    save_rows(summary_rows, args.summary_csv)
    print_summary(summary_rows)
    print(f"saved results: {args.results_csv}")
    print(f"saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
