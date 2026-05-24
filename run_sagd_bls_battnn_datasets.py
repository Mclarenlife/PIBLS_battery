"""Run SAGD-BLS on BattNN's processed LabData and NASAData benchmarks."""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from sagd_bls_battery import SAGDBLS


LAB_PATH = Path(r"BattNN\data\LabData")
NASA_PATH = Path(r"BattNN\data\NASA11")
LAB_BASELINE = Path(r"BattNN\results\test record\LABdata results.txt")
NASA_BASELINE = Path(r"BattNN\results\test record\NASAdata results.txt")
RESULTS_CSV = Path(r"results\sagd_bls_battnn_datasets.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAGD-BLS on BattNN Lab/NASA processed datasets.")
    parser.add_argument("--datasets", nargs="+", default=["Lab", "NASA"], choices=["Lab", "NASA"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--train-length", type=int, default=60)
    parser.add_argument("--activation", nargs=2, default=["tanh", "sigmoid"], metavar=("MAP", "ENHANCE"))
    parser.add_argument("--n-map", type=int, default=100)
    parser.add_argument("--n-enhance", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    return parser.parse_args()


def lab_splits(path: Path, n_train: int, length: int) -> Iterable[tuple[str, int, list[np.ndarray], list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]]:
    files = [name for name in os.listdir(path) if name.endswith(".npy")]
    for test_id in range(1, 9):
        test_battery = f"B{test_id}"
        train_files = [name for name in files if test_battery not in name]
        test_files = [name for name in files if test_battery in name]
        if not test_files:
            continue
        random.shuffle(train_files)
        train_files = train_files[:n_train]
        train_x, train_y = [], []
        for file_name in train_files:
            sample = np.load(path / file_name)
            train_x.append(sample[0, :length].astype(np.float64))
            train_y.append(sample[1, :length].astype(np.float64))
        test_rows = []
        for file_name in test_files:
            sample = np.load(path / file_name)
            test_rows.append((sample[0, :].astype(np.float64), sample[1, :].astype(np.float64)))
        yield "Lab", test_id, train_x, train_y, test_rows


def nasa_splits(path: Path, n_train: int, length: int) -> Iterable[tuple[str, str, list[np.ndarray], list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]]:
    for battery in ["Dis_RW3.mat", "Dis_RW4.mat", "Dis_RW5.mat", "Dis_RW6.mat"]:
        data = loadmat(path / battery)["re_discharge_data"]
        random_index = np.random.permutation(data.shape[1])
        train_index = random_index[:n_train]
        test_index = random_index[n_train:]
        train_x, train_y = [], []
        for idx in train_index:
            current = data[0, idx][0][0, :length]
            voltage = data[0, idx][1][0, :length]
            if current is None:
                continue
            train_x.append(np.asarray(current, dtype=np.float64))
            train_y.append(np.asarray(voltage, dtype=np.float64))
        test_rows = []
        for idx in test_index:
            current = data[0, idx][0][0, :]
            voltage = data[0, idx][1][0, :]
            if current is not None:
                test_rows.append((np.asarray(current, dtype=np.float64), np.asarray(voltage, dtype=np.float64)))
        yield "NASA", battery, train_x, train_y, test_rows


def read_baselines() -> dict[tuple[str, str], dict[str, float]]:
    baselines: dict[tuple[str, str], dict[str, float]] = {}
    baselines.update(read_baseline_file(LAB_BASELINE, dataset="Lab", id_pattern=r"battery:(\d+)"))
    baselines.update(read_baseline_file(NASA_BASELINE, dataset="NASA", id_pattern=r"battery:([^\.]+\.mat)"))
    return baselines


def read_baseline_file(path: Path, dataset: str, id_pattern: str) -> dict[tuple[str, str], dict[str, float]]:
    if not path.exists():
        return {}
    metric_pattern = re.compile(r"(MAE|MAPE|MSE|RMSE)=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")
    id_re = re.compile(id_pattern)
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "model:BattNN" not in line or "MAE=" not in line:
            continue
        match = id_re.search(line)
        if not match:
            continue
        split_id = match.group(1)
        found = {name: float(value) for name, value in metric_pattern.findall(line)}
        if {"MAE", "MAPE", "MSE", "RMSE"}.issubset(found):
            grouped.setdefault((dataset, split_id), []).append(found)
    return {
        key: {
            f"baseline_{metric}_mean": float(np.mean([row[metric] for row in rows]))
            for metric in ("MAE", "MAPE", "MSE", "RMSE")
        }
        | {
            f"baseline_{metric}_std": float(np.std([row[metric] for row in rows]))
            for metric in ("MAE", "MAPE", "MSE", "RMSE")
        }
        | {"baseline_runs": float(len(rows))}
        for key, rows in grouped.items()
    }


def run_split(
    args: argparse.Namespace,
    dataset: str,
    split_id: str | int,
    train_x: list[np.ndarray],
    train_y: list[np.ndarray],
    test_rows: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> dict[str, object]:
    model = SAGDBLS(
        n_map=args.n_map,
        n_enhance=args.n_enhance,
        map_activation=args.activation[0],
        enhance_activation=args.activation[1],
        seed=seed,
        learning_rate=args.lr,
        epochs=args.epochs,
        l2=args.l2,
        smooth_l1_delta=args.delta,
    )
    model.fit(train_x, train_y, train_length=args.train_length)
    metrics = model.evaluate(test_rows)
    return {
        "method": "SAGD-BLS",
        "dataset": dataset,
        "split_id": str(split_id),
        "seed": seed,
        "map_activation": args.activation[0],
        "enhance_activation": args.activation[1],
        "n_map": args.n_map,
        "n_enhance": args.n_enhance,
        "n_train": len(train_x),
        "train_length": args.train_length,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "l2": args.l2,
        "smooth_l1_delta": args.delta,
        "initial_loss": model.training_summary_["initial_loss"],
        "final_loss": model.training_summary_["final_loss"],
        **metrics,
    }


def save_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> None:
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        arr = np.array([[float(row[m]) for m in ("MAE", "MAPE", "RMSE")] for row in subset])
        print(
            f"{dataset} SAGD-BLS: "
            f"MAE={arr[:, 0].mean():.6f} +/- {arr[:, 0].std():.6f}, "
            f"MAPE={arr[:, 1].mean():.6f} +/- {arr[:, 1].std():.6f}, "
            f"RMSE={arr[:, 2].mean():.6f} +/- {arr[:, 2].std():.6f}"
        )
        baseline_rows = [row for row in subset if "baseline_MAE_mean" in row]
        if baseline_rows:
            baseline_mae = np.mean([float(row["baseline_MAE_mean"]) for row in baseline_rows])
            print(f"{dataset} BattNN recorded baseline MAE={baseline_mae:.6f}")


def main() -> None:
    args = parse_args()
    random.seed(2022)
    np.random.seed(2022)
    baselines = read_baselines()
    rows: list[dict[str, object]] = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    split_generators = []
    if "Lab" in args.datasets:
        split_generators.append(lab_splits(LAB_PATH, args.n_train, args.train_length))
    if "NASA" in args.datasets:
        split_generators.append(nasa_splits(NASA_PATH, args.n_train, args.train_length))

    for split_generator in split_generators:
        for dataset, split_id, train_x, train_y, test_rows in split_generator:
            for seed in args.seeds:
                row = run_split(args, dataset, split_id, train_x, train_y, test_rows, seed)
                row["timestamp"] = timestamp
                baseline = baselines.get((dataset, str(split_id)), {})
                row.update(baseline)
                rows.append(row)
                print(
                    f"{dataset} split={split_id} seed={seed}: "
                    f"MAE={row['MAE']:.6f}, MAPE={row['MAPE']:.6f}, RMSE={row['RMSE']:.6f}"
                )

    save_rows(rows, args.results_csv)
    summarize(rows)
    print(f"saved results: {args.results_csv}")


if __name__ == "__main__":
    main()
