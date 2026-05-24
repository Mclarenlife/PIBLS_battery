"""Compare SAGD-BLS-PDE against PIBLS-style and PINN baselines.

Baseline problem:
    1D Poisson
        -u''(x) = pi^2 sin(pi x), x in [0, 1]
        u(0) = u(1) = 0
        exact u(x) = sin(pi x)

Methods:
    SAGD-BLS-PDE:
        Fixed broad random features; Adam trains only the linear output beta.
    PIBLS-pinv:
        Same broad feature family and physics-informed residual system, but beta
        is solved by pseudo-inverse. This is the fair 1D analogue of PIBLS.py.
    PINN:
        Tanh MLP trained by Adam on residual and boundary losses.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sagd_bls_pde_solver import (
    SAGDBLSPDESolver1D,
    poisson_exact,
    poisson_source,
    regression_metrics,
)


RESULTS_CSV = Path("results") / "pde_poisson_baseline_results.csv"
SUMMARY_CSV = Path("results") / "pde_poisson_baseline_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poisson baseline comparison for SAGD-BLS-PDE.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--n-map", type=int, default=120)
    parser.add_argument("--n-enhance", type=int, default=120)
    parser.add_argument("--activation", nargs=2, default=["tanh", "tanh"], metavar=("MAP", "ENHANCE"))
    parser.add_argument("--n-interior", type=int, default=128)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--boundary-weight", type=float, default=100.0)
    parser.add_argument("--sagd-epochs", type=int, default=20_000)
    parser.add_argument("--sagd-lr", type=float, default=0.01)
    parser.add_argument("--sagd-l2", type=float, default=1e-6)
    parser.add_argument("--pibls-ridge", type=float, default=0.0)
    parser.add_argument("--pinn-epochs", type=int, default=8_000)
    parser.add_argument("--pinn-lr", type=float, default=1e-3)
    parser.add_argument("--pinn-hidden", type=int, default=64)
    parser.add_argument("--pinn-depth", type=int, default=3)
    parser.add_argument("--pinn-l2", type=float, default=0.0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--skip-pinn", action="store_true")
    return parser.parse_args()


def interior_points(n_interior: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_interior + 2, dtype=np.float64)[1:-1]


def boundary_points() -> tuple[np.ndarray, np.ndarray]:
    return np.array([0.0, 1.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)


def evaluation_points(n_eval: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_eval, dtype=np.float64)


def run_sagd(args: argparse.Namespace, seed: int) -> dict[str, object]:
    start = time.perf_counter()
    x_int = interior_points(args.n_interior)
    x_bnd, u_bnd = boundary_points()
    x_eval = evaluation_points(args.n_eval)

    model = SAGDBLSPDESolver1D(
        n_map=args.n_map,
        n_enhance=args.n_enhance,
        map_activation=args.activation[0],
        enhance_activation=args.activation[1],
        seed=seed,
        learning_rate=args.sagd_lr,
        epochs=args.sagd_epochs,
        l2=args.sagd_l2,
        boundary_weight=args.boundary_weight,
    )
    model.fit_poisson(x_int, poisson_source, x_bnd, u_bnd)
    metrics = model.evaluate_poisson(poisson_exact, poisson_source, x_eval, x_bnd, u_bnd)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "problem": "1D Poisson",
        "method": "SAGD-BLS-PDE",
        "seed": seed,
        "n_map": args.n_map,
        "n_enhance": args.n_enhance,
        "map_activation": args.activation[0],
        "enhance_activation": args.activation[1],
        "n_interior": args.n_interior,
        "n_eval": args.n_eval,
        "boundary_weight": args.boundary_weight,
        "epochs": args.sagd_epochs,
        "learning_rate": args.sagd_lr,
        "l2": args.sagd_l2,
        "trainable_parameters": 2 + args.n_map + args.n_enhance,  # design width: bias+x+map+enhance
        "runtime_sec": time.perf_counter() - start,
        **model.training_summary_,
        **metrics,
    }


@dataclass
class PIBLSPinv1D:
    """PIBLS-style pseudo-inverse baseline for the same 1D Poisson task."""

    n_map: int
    n_enhance: int
    map_activation: str
    enhance_activation: str
    seed: int
    boundary_weight: float
    ridge: float = 0.0

    core_: SAGDBLSPDESolver1D | None = None

    def fit(self, x_int: np.ndarray, x_bnd: np.ndarray, u_bnd: np.ndarray) -> "PIBLSPinv1D":
        self.core_ = SAGDBLSPDESolver1D(
            n_map=self.n_map,
            n_enhance=self.n_enhance,
            map_activation=self.map_activation,
            enhance_activation=self.enhance_activation,
            seed=self.seed,
            epochs=1,
            l2=self.ridge,
            boundary_weight=self.boundary_weight,
        )
        x_all = np.concatenate([x_int.reshape(-1), x_bnd.reshape(-1)]).reshape(-1, 1)
        self.core_._fit_feature_scalers(x_all)
        phi_int, _, phi_xx = self.core_.design_and_derivatives(x_int.reshape(-1, 1))
        phi_bnd, _, _ = self.core_.design_and_derivatives(x_bnd.reshape(-1, 1))
        residual_operator = -phi_xx
        source = poisson_source(x_int)
        scale_b = np.sqrt(self.boundary_weight)
        a_matrix = np.vstack([residual_operator, scale_b * phi_bnd])
        b_vector = np.concatenate([source, scale_b * u_bnd])
        if self.ridge > 0:
            lhs = a_matrix.T @ a_matrix + self.ridge * np.eye(a_matrix.shape[1])
            beta = np.linalg.solve(lhs, a_matrix.T @ b_vector)
        else:
            beta = np.linalg.pinv(a_matrix) @ b_vector
        self.core_.beta_ = beta
        self.core_.training_summary_ = self.core_.poisson_objective(phi_int, phi_xx, source, phi_bnd, u_bnd, beta)
        return self

    def evaluate(self, x_eval: np.ndarray, x_bnd: np.ndarray, u_bnd: np.ndarray) -> dict[str, float]:
        if self.core_ is None:
            raise ValueError("Call fit() before evaluate().")
        return self.core_.evaluate_poisson(poisson_exact, poisson_source, x_eval, x_bnd, u_bnd)


def run_pibls(args: argparse.Namespace, seed: int) -> dict[str, object]:
    start = time.perf_counter()
    x_int = interior_points(args.n_interior)
    x_bnd, u_bnd = boundary_points()
    x_eval = evaluation_points(args.n_eval)
    model = PIBLSPinv1D(
        n_map=args.n_map,
        n_enhance=args.n_enhance,
        map_activation=args.activation[0],
        enhance_activation=args.activation[1],
        seed=seed,
        boundary_weight=args.boundary_weight,
        ridge=args.pibls_ridge,
    ).fit(x_int, x_bnd, u_bnd)
    metrics = model.evaluate(x_eval, x_bnd, u_bnd)
    assert model.core_ is not None
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "problem": "1D Poisson",
        "method": "PIBLS-pinv",
        "seed": seed,
        "n_map": args.n_map,
        "n_enhance": args.n_enhance,
        "map_activation": args.activation[0],
        "enhance_activation": args.activation[1],
        "n_interior": args.n_interior,
        "n_eval": args.n_eval,
        "boundary_weight": args.boundary_weight,
        "epochs": 0,
        "learning_rate": "",
        "l2": args.pibls_ridge,
        "trainable_parameters": 2 + args.n_map + args.n_enhance,
        "runtime_sec": time.perf_counter() - start,
        **model.core_.training_summary_,
        **metrics,
    }


class PINN1D(nn.Module):
    def __init__(self, hidden: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        layers: list[nn.Module] = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pinn_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x_req = x.detach().clone().requires_grad_(True)
    u = model(x_req)
    du = torch.autograd.grad(u, x_req, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, x_req, torch.ones_like(du), create_graph=True)[0]
    source = (torch.pi**2) * torch.sin(torch.pi * x_req)
    return -d2u - source


def run_pinn(args: argparse.Namespace, seed: int) -> dict[str, object]:
    start = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_int_np = interior_points(args.n_interior)
    x_bnd_np, u_bnd_np = boundary_points()
    x_eval_np = evaluation_points(args.n_eval)

    dtype = torch.float64
    x_int = torch.tensor(x_int_np.reshape(-1, 1), dtype=dtype)
    x_bnd = torch.tensor(x_bnd_np.reshape(-1, 1), dtype=dtype)
    u_bnd = torch.tensor(u_bnd_np.reshape(-1, 1), dtype=dtype)

    model = PINN1D(hidden=args.pinn_hidden, depth=args.pinn_depth).to(dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pinn_lr, weight_decay=args.pinn_l2)

    initial_loss = None
    final_loss = None
    for epoch in range(1, args.pinn_epochs + 1):
        optimizer.zero_grad()
        residual = pinn_residual(model, x_int)
        boundary_residual = model(x_bnd) - u_bnd
        residual_loss = torch.mean(residual**2)
        boundary_loss = torch.mean(boundary_residual**2)
        loss = residual_loss + args.boundary_weight * boundary_loss
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        if epoch == args.pinn_epochs:
            final_loss = float(loss.detach().cpu())

    with torch.no_grad():
        x_eval = torch.tensor(x_eval_np.reshape(-1, 1), dtype=dtype)
        pred = model(x_eval).detach().cpu().numpy().reshape(-1)
    exact = poisson_exact(x_eval_np)
    metrics = regression_metrics(exact, pred)
    residual_eval = pinn_residual(model, torch.tensor(x_eval_np.reshape(-1, 1), dtype=dtype)).detach().cpu().numpy()
    with torch.no_grad():
        boundary_pred = model(x_bnd).detach().cpu().numpy().reshape(-1)
    parameter_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    metrics.update(
        {
            "residual_RMSE": float(np.sqrt(np.mean(residual_eval.reshape(-1) ** 2))),
            "boundary_MAXAE": float(np.max(np.abs(boundary_pred - u_bnd_np))),
            "n_eval": float(args.n_eval),
        }
    )
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "problem": "1D Poisson",
        "method": "PINN",
        "seed": seed,
        "n_map": "",
        "n_enhance": "",
        "map_activation": "tanh",
        "enhance_activation": "",
        "n_interior": args.n_interior,
        "n_eval": args.n_eval,
        "boundary_weight": args.boundary_weight,
        "epochs": args.pinn_epochs,
        "learning_rate": args.pinn_lr,
        "l2": args.pinn_l2,
        "trainable_parameters": parameter_count,
        "runtime_sec": time.perf_counter() - start,
        "loss": final_loss,
        "residual_loss": "",
        "boundary_loss": "",
        "regularization": "",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        **metrics,
    }


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
        summary: dict[str, object] = {"problem": "1D Poisson", "method": method, "n": len(subset)}
        for metric in ("MAE", "RMSE", "MAXAE", "residual_RMSE", "boundary_MAXAE", "runtime_sec"):
            values = np.array([float(row[metric]) for row in subset], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
        summary["trainable_parameters"] = subset[0]["trainable_parameters"]
        summary_rows.append(summary)
    return summary_rows


def print_summary(summary_rows: list[dict[str, object]]) -> None:
    print("Poisson baseline summary:")
    for row in summary_rows:
        print(
            f"{row['method']}: "
            f"MAE={row['MAE_mean']:.6e} +/- {row['MAE_std']:.2e}, "
            f"RMSE={row['RMSE_mean']:.6e} +/- {row['RMSE_std']:.2e}, "
            f"residual_RMSE={row['residual_RMSE_mean']:.6e} +/- {row['residual_RMSE_std']:.2e}, "
            f"boundary_MAXAE={row['boundary_MAXAE_mean']:.6e} +/- {row['boundary_MAXAE_std']:.2e}, "
            f"runtime={row['runtime_sec_mean']:.3f}s"
        )


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        sagd_row = run_sagd(args, seed)
        rows.append(sagd_row)
        print(f"SAGD-BLS-PDE seed={seed}: MAE={sagd_row['MAE']:.6e}, RMSE={sagd_row['RMSE']:.6e}")

        pibls_row = run_pibls(args, seed)
        rows.append(pibls_row)
        print(f"PIBLS-pinv seed={seed}: MAE={pibls_row['MAE']:.6e}, RMSE={pibls_row['RMSE']:.6e}")

        if not args.skip_pinn:
            pinn_row = run_pinn(args, seed)
            rows.append(pinn_row)
            print(f"PINN seed={seed}: MAE={pinn_row['MAE']:.6e}, RMSE={pinn_row['RMSE']:.6e}")

    summary_rows = summarize(rows)
    save_rows(rows, args.results_csv)
    save_rows(summary_rows, args.summary_csv)
    print_summary(summary_rows)
    print(f"saved results: {args.results_csv}")
    print(f"saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
