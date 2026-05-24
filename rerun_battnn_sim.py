"""Reproduce the BattNN SimData baseline in the current environment.

The original BattNN scripts assume the SciencePlots matplotlib style is
registered before modules call ``plt.style.use('science')``. This wrapper does
that registration, keeps plotting headless, and records five independent
BattNN runs without modifying the upstream BattNN files.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import scienceplots  # noqa: F401  Registers the "science" and "ieee" styles.
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
BATTNN_DIR = ROOT / "BattNN"
RESULTS_CSV = ROOT / "results" / "battnn_sim_reproduced.csv"


@dataclass
class BattNNSimArgs:
    V0: float = 4.183
    x0: tuple[float, float, float] = (7856.3254, 0.0, 0.0)
    dt: float = 1.0
    VEOD: float = 3.0
    Rp: float = 10000.0
    Rs: float = 0.1
    Csp: float = 10.0
    Cs: float = 400.0
    batch_size: int = 30
    seq_len: int = 60
    device: str = "cpu"
    epoch: int = 1000
    lr: float = 2e-2
    weight_decay: float = 5e-4
    model_name: str = "BattNN"
    save_model: None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun BattNN on SimData.")
    parser.add_argument("--experiments", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--verbose-train", action="store_true")
    return parser.parse_args()


def import_battnn_modules():
    os.chdir(BATTNN_DIR)
    sys.path.insert(0, str(BATTNN_DIR))
    from BattNN import BattNN
    from dataloader.load_sim_data import load_train_data, yield_test_data
    from functions import eval_metrix

    return BattNN, load_train_data, yield_test_data, eval_metrix


def evaluate(model, yield_test_data, eval_metrix) -> dict[str, float]:
    rows = []
    torch.nn.Module.train(model, False)
    with torch.no_grad():
        for current, voltage in yield_test_data():
            c_tensor = torch.from_numpy(current.astype(np.float32))
            pred, _ = model.predict(c_tensor.view(1, -1))
            pred = pred.detach().cpu().numpy()[0]
            rows.append(eval_metrix(voltage, pred))
    error = np.mean(np.array(rows, dtype=np.float64), axis=0)
    return {
        "MAE": float(error[0]),
        "MAPE": float(error[1]),
        "MSE": float(error[2]),
        "RMSE": float(error[3]),
        "n_sequences": float(len(rows)),
    }


def run() -> list[dict[str, object]]:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    BattNN, load_train_data, yield_test_data, eval_metrix = import_battnn_modules()
    rows: list[dict[str, object]] = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    # Recreate the original loader behavior: seed once, then let shuffling
    # advance across experiments.
    random.seed(args.seed)
    np.random.seed(args.seed)

    for experiment in range(1, args.experiments + 1):
        torch_seed = args.seed + experiment
        torch.manual_seed(torch_seed)

        config = BattNNSimArgs(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            epoch=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        train_x, train_y = load_train_data(n=config.batch_size, length=config.seq_len)
        model = BattNN(config)
        c_tensor = torch.from_numpy(train_x.astype(np.float32))
        v_tensor = torch.from_numpy(train_y.astype(np.float32))
        model.get_data(x=c_tensor, label=v_tensor)
        if args.verbose_train:
            model.train()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                model.train()
        if model.best_state is not None:
            model.load_state_dict(model.best_state["net"])
        metrics = evaluate(model, yield_test_data, eval_metrix)

        row: dict[str, object] = {
            "timestamp": timestamp,
            "method": "BattNN",
            "dataset": "SimData",
            "experiment": experiment,
            "seed": args.seed,
            "torch_seed": torch_seed,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "torch_version": torch.__version__,
        }
        row.update(metrics)
        rows.append(row)
        print(
            f"BattNN experiment={experiment}: "
            f"MAE={metrics['MAE']:.6f}, MAPE={metrics['MAPE']:.6f}, RMSE={metrics['RMSE']:.6f}"
        )

    save_rows(rows, args.results_csv)
    summarize(rows, args.results_csv)
    return rows


def save_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], path: Path) -> None:
    metrics = np.array([[float(row[name]) for name in ("MAE", "MAPE", "MSE", "RMSE")] for row in rows])
    mean = metrics.mean(axis=0)
    std = metrics.std(axis=0)
    print(
        "BattNN reproduced summary: "
        f"MAE={mean[0]:.6f} +/- {std[0]:.6f}, "
        f"MAPE={mean[1]:.6f} +/- {std[1]:.6f}, "
        f"MSE={mean[2]:.6f} +/- {std[2]:.6f}, "
        f"RMSE={mean[3]:.6f} +/- {std[3]:.6f}"
    )
    print(f"saved results: {path}")


if __name__ == "__main__":
    run()
