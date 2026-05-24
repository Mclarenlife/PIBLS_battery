"""Run SAGD-BLS on raw XJTU `.mat` battery cycles.

The first XJTU industrial test uses Batch-5 random-walk discharge batteries.
Each cycle's negative-current discharge segments are concatenated, converted to
positive discharge current, resampled to a coarser time step, and used for
single-cycle current-to-voltage prediction.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from sagd_bls_battery import SAGDBLS


XJTU_ROOT = Path("XJTU battery dataset")
RESULTS_CSV = Path(r"results\sagd_bls_xjtu_batch5.csv")


@dataclass
class CycleSequence:
    battery: str
    cycle: int
    current: np.ndarray
    voltage: np.ndarray
    duration_min: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAGD-BLS on raw XJTU battery `.mat` data.")
    parser.add_argument("--batch", default="Batch-5")
    parser.add_argument("--root", type=Path, default=XJTU_ROOT)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--data-seed", type=int, default=2022)
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--train-length", type=int, default=60)
    parser.add_argument("--resample-minutes", type=float, default=0.5)
    parser.add_argument("--min-test-length", type=int, default=20)
    parser.add_argument("--context", choices=["none", "cycle", "cycle_voltage"], default="none")
    parser.add_argument("--early-voltage-points", type=int, default=5)
    parser.add_argument("--activation", nargs=2, default=["tanh", "sigmoid"], metavar=("MAP", "ENHANCE"))
    parser.add_argument("--n-map", type=int, default=100)
    parser.add_argument("--n-enhance", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1.0)
    return parser.parse_args()


def extract_batch_sequences(
    batch_dir: Path,
    resample_minutes: float,
    min_length: int,
) -> list[CycleSequence]:
    sequences: list[CycleSequence] = []
    for mat_path in sorted(batch_dir.glob("*.mat")):
        sequences.extend(extract_battery_sequences(mat_path, resample_minutes, min_length))
    return sequences


def extract_battery_sequences(
    mat_path: Path,
    resample_minutes: float,
    min_length: int,
) -> list[CycleSequence]:
    mat = loadmat(mat_path)
    data = mat["data"]
    rows: list[CycleSequence] = []
    for cycle_idx in range(data.shape[1]):
        extracted = extract_cycle_discharge(data[0][cycle_idx], resample_minutes)
        if extracted is None:
            continue
        current, voltage, duration_min = extracted
        if current.size >= min_length:
            rows.append(
                CycleSequence(
                    battery=mat_path.stem,
                    cycle=cycle_idx + 1,
                    current=current,
                    voltage=voltage,
                    duration_min=duration_min,
                )
            )
    return rows


def extract_cycle_discharge(cycle_data, resample_minutes: float) -> tuple[np.ndarray, np.ndarray, float] | None:
    relative_time = np.asarray(cycle_data[1]).reshape(-1).astype(np.float64)
    voltage = np.asarray(cycle_data[2]).reshape(-1).astype(np.float64)
    current = np.asarray(cycle_data[3]).reshape(-1).astype(np.float64)
    if relative_time.size < 2:
        return None

    reset_indices = np.where(relative_time == 0)[0]
    if reset_indices.size == 0 or reset_indices[0] != 0:
        reset_indices = np.insert(reset_indices, 0, 0)
    boundaries = np.r_[reset_indices, relative_time.size]

    time_parts: list[np.ndarray] = []
    current_parts: list[np.ndarray] = []
    voltage_parts: list[np.ndarray] = []
    offset = 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start + 1:
            continue
        segment_current = current[start:end]
        if float(np.nanmean(segment_current)) >= -0.05:
            continue
        segment_time = relative_time[start:end] - relative_time[start]
        segment_voltage = voltage[start:end]
        valid = np.isfinite(segment_time) & np.isfinite(segment_current) & np.isfinite(segment_voltage)
        if valid.sum() < 2:
            continue
        segment_time = segment_time[valid]
        segment_current = segment_current[valid]
        segment_voltage = segment_voltage[valid]
        order = np.argsort(segment_time)
        segment_time = segment_time[order]
        segment_current = segment_current[order]
        segment_voltage = segment_voltage[order]
        unique_time, unique_idx = np.unique(segment_time, return_index=True)
        if unique_time.size < 2:
            continue
        segment_time = unique_time
        segment_current = segment_current[unique_idx]
        segment_voltage = segment_voltage[unique_idx]

        time_parts.append(offset + segment_time)
        current_parts.append(-segment_current)  # match BattNN convention: discharge current is positive
        voltage_parts.append(segment_voltage)

        dt = np.median(np.diff(segment_time))
        offset = float(time_parts[-1][-1] + max(dt, 1e-6))

    if not time_parts:
        return None

    time = np.concatenate(time_parts)
    discharge_current = np.concatenate(current_parts)
    discharge_voltage = np.concatenate(voltage_parts)
    if time[-1] < resample_minutes:
        return None

    target_time = np.arange(0.0, time[-1] + 1e-9, resample_minutes)
    if target_time.size < 2:
        return None
    resampled_current = np.interp(target_time, time, discharge_current)
    resampled_voltage = np.interp(target_time, time, discharge_voltage)
    return resampled_current, resampled_voltage, float(time[-1])


def choose_train_sequences(
    sequences: list[CycleSequence],
    held_out_battery: str,
    n_train: int,
    train_length: int,
    data_seed: int,
) -> list[CycleSequence]:
    candidates = [
        seq
        for seq in sequences
        if seq.battery != held_out_battery and seq.current.size >= train_length
    ]
    rng = random.Random(data_seed)
    rng.shuffle(candidates)
    if len(candidates) < n_train:
        raise RuntimeError(
            f"Only {len(candidates)} train sequences available for held-out {held_out_battery}; "
            f"need {n_train}."
        )
    return candidates[:n_train]


def evaluate_split(
    args: argparse.Namespace,
    sequences: list[CycleSequence],
    held_out_battery: str,
    seed: int,
) -> dict[str, object]:
    train_sequences = choose_train_sequences(
        sequences,
        held_out_battery=held_out_battery,
        n_train=args.n_train,
        train_length=args.train_length,
        data_seed=args.data_seed,
    )
    test_sequences = [
        seq
        for seq in sequences
        if seq.battery == held_out_battery and seq.current.size >= args.min_test_length
    ]
    if not test_sequences:
        raise RuntimeError(f"No test sequences for held-out battery {held_out_battery}.")

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
    model.fit(
        [seq.current for seq in train_sequences],
        [seq.voltage for seq in train_sequences],
        train_length=args.train_length,
        contexts=[context_features(seq, args) for seq in train_sequences] if args.context != "none" else None,
    )
    if args.context == "none":
        metrics = model.evaluate((seq.current, seq.voltage) for seq in test_sequences)
    else:
        metrics = model.evaluate((seq.current, seq.voltage, context_features(seq, args)) for seq in test_sequences)
    return {
        "method": "SAGD-BLS",
        "dataset": f"XJTU-{args.batch}",
        "held_out_battery": held_out_battery,
        "seed": seed,
        "data_seed": args.data_seed,
        "map_activation": args.activation[0],
        "enhance_activation": args.activation[1],
        "n_map": args.n_map,
        "n_enhance": args.n_enhance,
        "n_train": args.n_train,
        "train_length": args.train_length,
        "resample_minutes": args.resample_minutes,
        "min_test_length": args.min_test_length,
        "context": args.context,
        "early_voltage_points": args.early_voltage_points,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "l2": args.l2,
        "smooth_l1_delta": args.delta,
        "train_cycles": "|".join(f"{seq.battery}:{seq.cycle}" for seq in train_sequences),
        "test_cycles": len(test_sequences),
        "mean_test_duration_min": float(np.mean([seq.duration_min for seq in test_sequences])),
        "initial_loss": model.training_summary_["initial_loss"],
        "final_loss": model.training_summary_["final_loss"],
        **metrics,
    }


def context_features(seq: CycleSequence, args: argparse.Namespace) -> np.ndarray:
    cycle_norm = seq.cycle / 500.0
    duration_norm = seq.duration_min / 60.0
    if args.context == "cycle":
        return np.array([cycle_norm, duration_norm], dtype=np.float64)
    if args.context == "cycle_voltage":
        k = min(max(1, args.early_voltage_points), seq.voltage.size)
        early_voltage = seq.voltage[:k]
        return np.array(
            [
                cycle_norm,
                duration_norm,
                float(early_voltage[0]),
                float(early_voltage.mean()),
                float(early_voltage[-1] - early_voltage[0]),
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported context mode: {args.context}")


def save_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> None:
    arr = np.array([[float(row[m]) for m in ("MAE", "MAPE", "RMSE")] for row in rows])
    print(
        "XJTU SAGD-BLS summary: "
        f"MAE={arr[:, 0].mean():.6f} +/- {arr[:, 0].std():.6f}, "
        f"MAPE={arr[:, 1].mean():.6f} +/- {arr[:, 1].std():.6f}, "
        f"RMSE={arr[:, 2].mean():.6f} +/- {arr[:, 2].std():.6f}"
    )
    for battery in sorted({str(row["held_out_battery"]) for row in rows}):
        subset = [row for row in rows if row["held_out_battery"] == battery]
        vals = np.array([float(row["MAE"]) for row in subset])
        print(f"{battery}: MAE={vals.mean():.6f} +/- {vals.std():.6f}")


def main() -> None:
    args = parse_args()
    batch_dir = args.root / args.batch
    timestamp = datetime.now().isoformat(timespec="seconds")

    extraction_min_length = max(args.min_test_length, 2)
    sequences = extract_batch_sequences(batch_dir, args.resample_minutes, extraction_min_length)
    batteries = sorted({seq.battery for seq in sequences})
    print(
        f"loaded {len(sequences)} discharge cycles from {batch_dir}; "
        f"batteries={len(batteries)}; resample={args.resample_minutes} min"
    )

    rows: list[dict[str, object]] = []
    for held_out_battery in batteries:
        eligible_train = [
            seq for seq in sequences if seq.battery != held_out_battery and seq.current.size >= args.train_length
        ]
        if len(eligible_train) < args.n_train:
            print(f"skip {held_out_battery}: only {len(eligible_train)} eligible training cycles")
            continue
        for seed in args.seeds:
            row = evaluate_split(args, sequences, held_out_battery, seed)
            row["timestamp"] = timestamp
            rows.append(row)
            print(
                f"held_out={held_out_battery} seed={seed}: "
                f"MAE={row['MAE']:.6f}, MAPE={row['MAPE']:.6f}, RMSE={row['RMSE']:.6f}, "
                f"test_cycles={row['test_cycles']}"
            )

    if not rows:
        raise RuntimeError("No XJTU experiments were run.")
    save_rows(rows, args.results_csv)
    summarize(rows)
    print(f"saved results: {args.results_csv}")


if __name__ == "__main__":
    main()
