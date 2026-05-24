"""Run SAGD-BLS on the BattNN SimData benchmark."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from sagd_bls_battery import SAGDBLS


SIM_TRAIN_FOLDER = Path(r"BattNN\data\SimData\current=[2, 8] len=[60, 200] train")
SIM_TEST_FOLDER = Path(r"BattNN\data\SimData\current=[2, 8] len=[60, 200] test")
SIM_BASELINE_FILE = Path(r"BattNN\results\batch size and seq len\Simdata_BattNN_batch_size.txt")
DEFAULT_RESULTS_CSV = Path(r"results\sagd_bls_sim_results.csv")
ABLATION_ACTIVATIONS = [
    ("sigmoid", "sigmoid"),
    ("tanh", "sigmoid"),
    ("sin", "sigmoid"),
    ("softplus", "sigmoid"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAGD-BLS for BattNN SimData.")
    parser.add_argument("--train-folder", type=Path, default=SIM_TRAIN_FOLDER)
    parser.add_argument("--test-folder", type=Path, default=SIM_TEST_FOLDER)
    parser.add_argument("--baseline-file", type=Path, default=SIM_BASELINE_FILE)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--train-length", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--activation", nargs=2, default=["sigmoid", "sigmoid"], metavar=("MAP", "ENHANCE"))
    parser.add_argument("--n-map", type=int, default=100)
    parser.add_argument("--n-enhance", type=int, default=100)
    parser.add_argument("--map-scale", type=float, default=1.0)
    parser.add_argument("--enhance-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--skip-ablation", action="store_true", help="Only run the requested activation pair.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip quick implementation checks.")
    parser.add_argument(
        "--rerun-battnn",
        action="store_true",
        help="Try to rerun BattNN with the local Python environment before SAGD-BLS.",
    )
    return parser.parse_args()


def load_train_data(folder: Path, n: int, length: int) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    files = os.listdir(folder)
    random.seed(2022)
    random.shuffle(files)
    currents: list[np.ndarray] = []
    voltages: list[np.ndarray] = []
    names: list[str] = []
    for file_name in files:
        sample = np.load(folder / file_name)
        if sample.shape[1] < length:
            continue
        currents.append(sample[0, :length].astype(np.float64))
        voltages.append(sample[1, :length].astype(np.float64))
        names.append(file_name)
        if len(currents) >= n:
            break
    if len(currents) < n:
        raise RuntimeError(f"Only found {len(currents)} training samples with length >= {length}.")
    return currents, voltages, names


def iter_test_data(folder: Path) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    files = os.listdir(folder)
    random.seed(2022)
    random.shuffle(files)
    for file_name in files:
        sample = np.load(folder / file_name)
        yield sample[0, :].astype(np.float64), sample[1, :].astype(np.float64)


def read_battnn_baseline(path: Path, batch_size: int = 30, seq_len: int = 60) -> dict[str, float]:
    if not path.exists():
        return {}
    metric_pattern = re.compile(r"(MAE|MAPE|MSE|RMSE)=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "model:BattNN" not in line or "MAE=" not in line:
            continue
        if f"batch size:{batch_size}." not in line or f"seq len:{seq_len}" not in line:
            continue
        found = {name: float(value) for name, value in metric_pattern.findall(line)}
        if {"MAE", "MAPE", "MSE", "RMSE"}.issubset(found):
            rows.append(found)
    if not rows:
        return {}
    return {
        f"baseline_{metric}_mean": float(np.mean([row[metric] for row in rows]))
        for metric in ("MAE", "MAPE", "MSE", "RMSE")
    } | {
        f"baseline_{metric}_std": float(np.std([row[metric] for row in rows]))
        for metric in ("MAE", "MAPE", "MSE", "RMSE")
    } | {"baseline_runs": float(len(rows))}


def run_checks(train_x: list[np.ndarray], train_y: list[np.ndarray], test_folder: Path) -> None:
    features = SAGDBLS.state_features(train_x[0])
    expected_columns = 4 + 7
    if features.shape != (train_x[0].size, expected_columns):
        raise AssertionError(f"Unexpected state feature shape: {features.shape}.")

    check_model = SAGDBLS(n_map=8, n_enhance=8, seed=999, epochs=200, l2=1e-3)
    check_model.fit(train_x[:3], train_y[:3], train_length=min(20, train_x[0].size))
    summary = check_model.training_summary_
    if summary["final_loss"] >= summary["initial_loss"]:
        raise AssertionError(f"Adam did not reduce loss: {summary}.")

    current, voltage = next(iter_test_data(test_folder))
    pred = check_model.predict(current)
    if pred.shape != voltage.shape:
        raise AssertionError(f"Variable-length prediction shape mismatch: {pred.shape} != {voltage.shape}.")


def activation_plan(primary: tuple[str, str], skip_ablation: bool) -> list[tuple[str, str]]:
    if skip_ablation:
        return [primary]
    configs = [primary]
    for item in ABLATION_ACTIVATIONS:
        if item not in configs:
            configs.append(item)
    return configs


def run_sagd_bls(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, float]]:
    train_x, train_y, train_files = load_train_data(args.train_folder, args.n_train, args.train_length)
    if not args.skip_checks:
        run_checks(train_x, train_y, args.test_folder)

    baseline = read_battnn_baseline(args.baseline_file, batch_size=args.n_train, seq_len=args.train_length)
    configs = activation_plan((args.activation[0], args.activation[1]), args.skip_ablation)
    rows: list[dict[str, object]] = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    for map_act, enhance_act in configs:
        for seed in args.seeds:
            start = time.perf_counter()
            model = SAGDBLS(
                n_map=args.n_map,
                n_enhance=args.n_enhance,
                map_activation=map_act,
                enhance_activation=enhance_act,
                seed=seed,
                learning_rate=args.lr,
                epochs=args.epochs,
                l2=args.l2,
                map_scale=args.map_scale,
                enhance_scale=args.enhance_scale,
                backend=args.backend,
                device=args.device,
                torch_dtype=args.torch_dtype,
                smooth_l1_delta=args.delta,
            )
            model.fit(train_x, train_y, train_length=args.train_length)
            metrics = model.evaluate(iter_test_data(args.test_folder))
            runtime_sec = time.perf_counter() - start
            row: dict[str, object] = {
                "timestamp": timestamp,
                "method": "SAGD-BLS" if args.backend == "numpy" else "SAGD-BLS-torch",
                "dataset": "SimData",
                "map_activation": map_act,
                "enhance_activation": enhance_act,
                "seed": seed,
                "n_map": args.n_map,
                "n_enhance": args.n_enhance,
                "map_scale": args.map_scale,
                "enhance_scale": args.enhance_scale,
                "n_train": args.n_train,
                "train_length": args.train_length,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "l2": args.l2,
                "backend": args.backend,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "smooth_l1_delta": args.delta,
                "train_files": "|".join(train_files),
                "initial_loss": model.training_summary_["initial_loss"],
                "final_loss": model.training_summary_["final_loss"],
                "runtime_sec": runtime_sec,
            }
            row.update(metrics)
            row.update(baseline)
            rows.append(row)
            print(
                f"{map_act}/{enhance_act} seed={seed}: "
                f"MAE={metrics['MAE']:.6f}, MAPE={metrics['MAPE']:.6f}, RMSE={metrics['RMSE']:.6f}, "
                f"time={runtime_sec:.2f}s"
            )
        summarize_config(rows, map_act, enhance_act, baseline)
    return rows, baseline


def summarize_config(
    rows: list[dict[str, object]],
    map_act: str,
    enhance_act: str,
    baseline: dict[str, float],
) -> None:
    subset = [row for row in rows if row["map_activation"] == map_act and row["enhance_activation"] == enhance_act]
    mae = np.array([float(row["MAE"]) for row in subset])
    mape = np.array([float(row["MAPE"]) for row in subset])
    rmse = np.array([float(row["RMSE"]) for row in subset])
    runtime = np.array([float(row.get("runtime_sec", 0.0)) for row in subset])
    message = (
        f"summary {map_act}/{enhance_act}: "
        f"MAE={mae.mean():.6f} +/- {mae.std():.6f}, "
        f"MAPE={mape.mean():.6f} +/- {mape.std():.6f}, "
        f"RMSE={rmse.mean():.6f} +/- {rmse.std():.6f}, "
        f"time={runtime.mean():.2f}s"
    )
    if baseline:
        message += f", BattNN MAE baseline={baseline['baseline_MAE_mean']:.6f}"
    print(message)


def save_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rerun_battnn() -> None:
    missing = [
        package
        for package in ("torch", "sklearn", "matplotlib")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise RuntimeError(
            "Cannot rerun BattNN because this Python environment is missing: "
            + ", ".join(missing)
        )
    subprocess.run([sys.executable, "main_SimData.py"], cwd="BattNN", check=True)


def main() -> None:
    args = parse_args()
    if args.rerun_battnn:
        rerun_battnn()

    rows, baseline = run_sagd_bls(args)
    save_rows(rows, args.results_csv)

    primary_map, primary_enhance = args.activation
    primary_rows = [
        row
        for row in rows
        if row["map_activation"] == primary_map and row["enhance_activation"] == primary_enhance
    ]
    primary_mae = float(np.mean([float(row["MAE"]) for row in primary_rows]))
    print(f"saved results: {args.results_csv}")
    if baseline:
        battnn_mae = baseline["baseline_MAE_mean"]
        verdict = "PASS" if primary_mae < battnn_mae else "FAIL"
        print(f"{verdict}: primary SAGD-BLS MAE={primary_mae:.6f}, BattNN MAE={battnn_mae:.6f}")


if __name__ == "__main__":
    main()
