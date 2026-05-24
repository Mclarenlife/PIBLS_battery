"""Audit SimData comparison details between BattNN and SAGD-BLS.

This script mirrors the upstream BattNN loader's repeated sampling behavior:
seed once, then call ``random.shuffle`` once per experiment on a fresh file
list. It trains SAGD-BLS on those five train splits so the comparison is less
dependent on the fixed split used by the primary SAGD-BLS runner.
"""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from run_sagd_bls_sim import SIM_TEST_FOLDER, SIM_TRAIN_FOLDER, iter_test_data
from sagd_bls_battery import SAGDBLS


RESULTS_CSV = Path("results") / "sagd_bls_sim_battnn_sampling_audit.csv"


def battnn_style_train_splits(folder: Path, experiments: int, n: int, length: int) -> list[list[str]]:
    random.seed(2022)
    splits: list[list[str]] = []
    for _ in range(experiments):
        files = os.listdir(folder)
        random.shuffle(files)
        chosen: list[str] = []
        for file_name in files:
            sample = np.load(folder / file_name)
            if sample.shape[1] < length:
                continue
            chosen.append(file_name)
            if len(chosen) >= n:
                break
        if len(chosen) < n:
            raise RuntimeError(f"Only found {len(chosen)} train samples with length >= {length}.")
        splits.append(chosen)
    return splits


def load_named_split(folder: Path, files: list[str], length: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    currents: list[np.ndarray] = []
    voltages: list[np.ndarray] = []
    for file_name in files:
        sample = np.load(folder / file_name)
        currents.append(sample[0, :length].astype(np.float64))
        voltages.append(sample[1, :length].astype(np.float64))
    return currents, voltages


def main() -> None:
    splits = battnn_style_train_splits(SIM_TRAIN_FOLDER, experiments=5, n=30, length=60)
    rows: list[dict[str, object]] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for experiment, train_files in enumerate(splits, start=1):
        train_x, train_y = load_named_split(SIM_TRAIN_FOLDER, train_files, length=60)
        model = SAGDBLS(
            n_map=100,
            n_enhance=100,
            map_activation="tanh",
            enhance_activation="sigmoid",
            seed=experiment,
            epochs=20_000,
            learning_rate=0.01,
            l2=1e-3,
            smooth_l1_delta=1.0,
        )
        model.fit(train_x, train_y, train_length=60)
        metrics = model.evaluate(iter_test_data(SIM_TEST_FOLDER))
        row: dict[str, object] = {
            "timestamp": timestamp,
            "method": "SAGD-BLS",
            "dataset": "SimData",
            "sampling": "battnn_loader_style",
            "experiment": experiment,
            "seed": experiment,
            "n_train": 30,
            "train_length": 60,
            "n_map": 100,
            "n_enhance": 100,
            "map_activation": "tanh",
            "enhance_activation": "sigmoid",
            "epochs": 20_000,
            "learning_rate": 0.01,
            "l2": 1e-3,
            "initial_loss": model.training_summary_["initial_loss"],
            "final_loss": model.training_summary_["final_loss"],
            "train_files": "|".join(train_files),
            **metrics,
        }
        rows.append(row)
        print(
            f"SAGD-BLS BattNN-style split experiment={experiment}: "
            f"MAE={metrics['MAE']:.6f}, MAPE={metrics['MAPE']:.6f}, RMSE={metrics['RMSE']:.6f}"
        )

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as file:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    arr = np.array([[float(row[name]) for name in ("MAE", "MAPE", "RMSE")] for row in rows])
    print(
        "Audit summary: "
        f"MAE={arr[:, 0].mean():.6f} +/- {arr[:, 0].std():.6f}, "
        f"MAPE={arr[:, 1].mean():.6f} +/- {arr[:, 1].std():.6f}, "
        f"RMSE={arr[:, 2].mean():.6f} +/- {arr[:, 2].std():.6f}"
    )
    print(f"saved results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
