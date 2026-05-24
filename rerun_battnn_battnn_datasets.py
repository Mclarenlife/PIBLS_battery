"""Rerun BattNN on its processed LabData and NASAData benchmarks.

This is a headless wrapper around the upstream BattNN code. It registers the
SciencePlots style before importing BattNN modules, avoids overwriting the
upstream result pickle files, and saves metrics to a CSV for comparison with
SAGD-BLS.
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
import scienceplots  # noqa: F401  Registers matplotlib "science" style.
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
BATTNN_DIR = ROOT / "BattNN"
RESULTS_CSV = ROOT / "results" / "battnn_lab_nasa_reproduced.csv"


@dataclass
class BattNNArgs:
    V0: float
    x0: tuple[float, float, float]
    dt: float
    VEOD: float
    Rp: float
    Rs: float
    Csp: float
    Cs: float
    batch_size: int
    seq_len: int
    device: str
    epoch: int
    lr: float
    weight_decay: float
    model_name: str = "BattNN"
    save_model: None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun BattNN on LabData and NASAData.")
    parser.add_argument("--datasets", nargs="+", default=["Lab", "NASA"], choices=["Lab", "NASA"])
    parser.add_argument("--experiments", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=2000)
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
    from dataloader.load_NASA_data import NASAdata
    from functions import eval_metrix

    return BattNN, NASAdata, eval_metrix


def lab_config(args: argparse.Namespace) -> BattNNArgs:
    return BattNNArgs(
        V0=4.2,
        x0=(8000.0, 0.0, 0.0),
        dt=1.0,
        VEOD=3.2,
        Rp=6000.0,
        Rs=1.0,
        Csp=40.0,
        Cs=800.0,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device="cpu",
        epoch=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def nasa_config(args: argparse.Namespace) -> BattNNArgs:
    return BattNNArgs(
        V0=4.2,
        x0=(8000.0, 0.0, 0.0),
        dt=1.0,
        VEOD=3.2,
        Rp=1000.0,
        Rs=0.5,
        Csp=15.0,
        Cs=500.0,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device="cpu",
        epoch=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def evaluate(model, data_iter, eval_metrix) -> dict[str, float]:
    rows = []
    torch.nn.Module.train(model, False)
    with torch.no_grad():
        for current, voltage in data_iter():
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


def train_and_eval(BattNN, eval_metrix, config: BattNNArgs, train_x, train_y, data_iter, verbose: bool) -> dict[str, float]:
    model = BattNN(config)
    model.get_data(
        x=torch.from_numpy(train_x.astype(np.float32)),
        label=torch.from_numpy(train_y.astype(np.float32)),
    )
    if verbose:
        model.train()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            model.train()
    if model.best_state is not None:
        model.load_state_dict(model.best_state["net"])
    return evaluate(model, data_iter, eval_metrix)


def available_lab_ids() -> list[int]:
    lab_dir = BATTNN_DIR / "data" / "LabData"
    ids = set()
    for name in os.listdir(lab_dir):
        if name.endswith(".npy") and name.startswith("B"):
            prefix = name.split("-", 1)[0]
            try:
                ids.add(int(prefix[1:]))
            except ValueError:
                pass
    return sorted(ids)


def load_lab_split(test_id: int, n: int, length: int):
    lab_dir = BATTNN_DIR / "data" / "LabData"
    files = [name for name in os.listdir(lab_dir) if name.endswith(".npy")]
    test_battery = f"B{test_id}"
    train_files = [name for name in files if test_battery not in name]
    test_files = [name for name in files if test_battery in name]
    random.shuffle(train_files)
    train_files = train_files[:n]

    train_x, train_y = [], []
    for file_name in train_files:
        sample = np.load(lab_dir / file_name)
        train_x.append(sample[0, :length])
        train_y.append(sample[1, :length])

    test_rows = []
    for file_name in test_files:
        sample = np.load(lab_dir / file_name)
        test_rows.append((sample[0, :], sample[1, :]))

    return np.asarray(train_x, dtype=np.float32), np.asarray(train_y, dtype=np.float32), test_rows


def run() -> list[dict[str, object]]:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    BattNN, NASAdata, eval_metrix = import_battnn_modules()
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows: list[dict[str, object]] = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    if "Lab" in args.datasets:
        for test_id in available_lab_ids():
            for experiment in range(1, args.experiments + 1):
                torch_seed = args.seed + 1000 + test_id * 10 + experiment
                torch.manual_seed(torch_seed)
                train_x, train_y, test_rows = load_lab_split(test_id, args.batch_size, args.seq_len)
                metrics = train_and_eval(
                    BattNN,
                    eval_metrix,
                    lab_config(args),
                    train_x,
                    train_y,
                    lambda rows=test_rows: iter(rows),
                    args.verbose_train,
                )
                row = make_row(args, timestamp, "Lab", str(test_id), experiment, torch_seed, metrics)
                rows.append(row)
                print_metric(row)

    if "NASA" in args.datasets:
        for battery in ["Dis_RW3.mat", "Dis_RW4.mat", "Dis_RW5.mat", "Dis_RW6.mat"]:
            for experiment in range(1, args.experiments + 1):
                torch_seed = args.seed + 2000 + experiment
                torch.manual_seed(torch_seed)
                data = NASAdata(path=f"data/NASA11/{battery}", n=args.batch_size, length=args.seq_len)
                train_x, train_y = data.load_train_data()
                metrics = train_and_eval(
                    BattNN,
                    eval_metrix,
                    nasa_config(args),
                    train_x,
                    train_y,
                    data.yield_test_data,
                    args.verbose_train,
                )
                row = make_row(args, timestamp, "NASA", battery, experiment, torch_seed, metrics)
                rows.append(row)
                print_metric(row)

    save_rows(rows, args.results_csv)
    summarize(rows, args.results_csv)
    return rows


def make_row(
    args: argparse.Namespace,
    timestamp: str,
    dataset: str,
    split_id: str,
    experiment: int,
    torch_seed: int,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "method": "BattNN",
        "dataset": dataset,
        "split_id": split_id,
        "experiment": experiment,
        "seed": args.seed,
        "torch_seed": torch_seed,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "torch_version": torch.__version__,
        **metrics,
    }


def print_metric(row: dict[str, object]) -> None:
    print(
        f"{row['dataset']} split={row['split_id']} experiment={row['experiment']}: "
        f"MAE={float(row['MAE']):.6f}, "
        f"MAPE={float(row['MAPE']):.6f}, "
        f"RMSE={float(row['RMSE']):.6f}"
    )


def save_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], path: Path) -> None:
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        metrics = np.array([[float(row[name]) for name in ("MAE", "MAPE", "MSE", "RMSE")] for row in subset])
        mean = metrics.mean(axis=0)
        std = metrics.std(axis=0)
        print(
            f"{dataset} BattNN reproduced: "
            f"MAE={mean[0]:.6f} +/- {std[0]:.6f}, "
            f"MAPE={mean[1]:.6f} +/- {std[1]:.6f}, "
            f"MSE={mean[2]:.6f} +/- {std[2]:.6f}, "
            f"RMSE={mean[3]:.6f} +/- {std[3]:.6f}"
        )
    print(f"saved results: {path}")


if __name__ == "__main__":
    run()
