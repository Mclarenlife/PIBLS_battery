"""Tune BattNN adapter hyperparameters on extracted XJTU cycles.

The original BattNN paper does not provide an XJTU raw-mat benchmark. This
script makes the adapter comparison stronger by selecting BattNN hyperparameters
on validation batteries before rerunning the leave-one-battery-out test.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import random
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import scienceplots  # noqa: F401
import numpy as np
import torch

from run_sagd_bls_xjtu import XJTU_ROOT, CycleSequence, choose_train_sequences, extract_batch_sequences


ROOT = Path(__file__).resolve().parent
BATTNN_DIR = ROOT / "BattNN"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class BattNNXJTUConfig:
    V0: float = 4.2
    x0: tuple[float, float, float] = (8000.0, 0.0, 0.0)
    dt: float = 1.0
    VEOD: float = 3.0
    Rp: float = 6000.0
    Rs: float = 1.0
    Csp: float = 40.0
    Cs: float = 800.0
    batch_size: int = 30
    seq_len: int = 60
    device: str = "cpu"
    epoch: int = 2000
    lr: float = 2e-2
    weight_decay: float = 5e-4
    model_name: str = "BattNN"
    save_model: None = None


PHYS_PROFILES: dict[str, BattNNXJTUConfig] = {
    "lab": BattNNXJTUConfig(V0=4.2, x0=(8000.0, 0.0, 0.0), VEOD=3.0, Rp=6000.0, Rs=1.0, Csp=40.0, Cs=800.0),
    "nasa": BattNNXJTUConfig(V0=4.2, x0=(8000.0, 0.0, 0.0), VEOD=3.0, Rp=1000.0, Rs=0.5, Csp=15.0, Cs=500.0),
    "sim": BattNNXJTUConfig(V0=4.183, x0=(7856.3254, 0.0, 0.0), VEOD=3.0, Rp=10000.0, Rs=0.1, Csp=10.0, Cs=400.0),
    "paper": BattNNXJTUConfig(V0=4.2, x0=(8000.0, 0.0, 0.0), VEOD=3.0, Rp=10000.0, Rs=0.0538926, Csp=14.8223, Cs=234.387),
    "soft": BattNNXJTUConfig(V0=4.2, x0=(6000.0, 0.0, 0.0), VEOD=2.5, Rp=3000.0, Rs=0.2, Csp=20.0, Cs=300.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune BattNN adapter on XJTU batches.")
    parser.add_argument("--batches", nargs="+", default=["Batch-5", "Batch-1"])
    parser.add_argument("--root", type=Path, default=XJTU_ROOT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--data-seed", type=int, default=2022)
    parser.add_argument("--torch-seed", type=int, default=2022)
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--train-length", type=int, default=60)
    parser.add_argument("--resample-minutes", type=float, default=0.5)
    parser.add_argument("--min-test-length", type=int, default=20)
    parser.add_argument("--validation-count", type=int, default=2)
    parser.add_argument("--profiles", nargs="+", default=["lab", "nasa", "sim", "paper"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 5e-4])
    parser.add_argument("--search-experiments", type=int, default=1)
    parser.add_argument("--final-experiments", type=int, default=5)
    parser.add_argument("--epochs-search", type=int, default=1200)
    parser.add_argument("--epochs-final", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--verbose-train", action="store_true")
    return parser.parse_args()


def import_battnn():
    sys.path.insert(0, str(BATTNN_DIR))
    from BattNN import BattNN

    return BattNN


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = true - pred
    mse = float(np.mean(err**2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err) / np.maximum(np.abs(true), 1e-12))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }


def train_model(BattNN, config: BattNNXJTUConfig, train_sequences: list[CycleSequence], seed: int, patience: int, verbose: bool):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    model = BattNN(config)
    train_x = np.asarray([seq.current[: config.seq_len] for seq in train_sequences], dtype=np.float32)
    train_y = np.asarray([seq.voltage[: config.seq_len] for seq in train_sequences], dtype=np.float32)
    model.get_data(x=torch.from_numpy(train_x), label=torch.from_numpy(train_y))

    best_loss = float("inf")
    best_state = None
    stale = 0
    losses: list[float] = []
    output = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with output:
        for _ in range(config.epoch):
            loss = model.train_one_epoch(print_per=10_000_000)
            model.scheduler.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale > patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "train_epochs_run": len(losses),
        "initial_train_loss": losses[0] if losses else float("nan"),
        "best_train_loss": best_loss,
        "final_train_loss": losses[-1] if losses else float("nan"),
    }


def evaluate(model, sequences: Iterable[CycleSequence]) -> dict[str, float]:
    sequence_list = list(sequences)
    if not sequence_list:
        raise ValueError("Cannot evaluate on an empty sequence list.")
    rows = []
    torch.nn.Module.train(model, False)
    with torch.no_grad():
        for same_length in group_by_length(sequence_list):
            currents = np.asarray([seq.current for seq in same_length], dtype=np.float32)
            voltages = [seq.voltage for seq in same_length]
            pred = predict_batch(model, currents)
            for true_voltage, pred_voltage in zip(voltages, pred):
                rows.append(metrics(true_voltage, pred_voltage))
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("MAE", "MAPE", "MSE", "RMSE")
    } | {"n_sequences": float(len(rows))}


def group_by_length(sequences: list[CycleSequence]) -> Iterable[list[CycleSequence]]:
    buckets: dict[int, list[CycleSequence]] = {}
    for seq in sequences:
        buckets.setdefault(int(seq.current.size), []).append(seq)
    for length in sorted(buckets):
        yield buckets[length]


def predict_batch(model, currents: np.ndarray) -> np.ndarray:
    original_init_x = model.init_x
    batch_size = currents.shape[0]
    model.init_x = torch.tensor(model.config.x0, device=model.device, dtype=torch.float32).repeat(batch_size, 1)
    try:
        pred, _ = model.predict(torch.from_numpy(currents.astype(np.float32)))
    finally:
        model.init_x = original_init_x
    return pred.detach().cpu().numpy()


def candidate_configs(args: argparse.Namespace) -> list[tuple[str, BattNNXJTUConfig]]:
    candidates = []
    for profile in args.profiles:
        if profile not in PHYS_PROFILES:
            raise ValueError(f"Unknown profile {profile!r}; available: {sorted(PHYS_PROFILES)}")
        base = PHYS_PROFILES[profile]
        for lr in args.lrs:
            for weight_decay in args.weight_decays:
                config = replace(
                    base,
                    batch_size=args.n_train,
                    seq_len=args.train_length,
                    epoch=args.epochs_search,
                    lr=lr,
                    weight_decay=weight_decay,
                )
                candidates.append((profile, config))
    return candidates


def choose_validation_batteries(batteries: list[str], count: int) -> list[str]:
    if count <= 0:
        raise ValueError("validation-count must be positive.")
    return batteries[: min(count, len(batteries))]


def choose_search_train_sequences(
    sequences: list[CycleSequence],
    validation_battery: str,
    n_train: int,
    train_length: int,
    data_seed: int,
) -> list[CycleSequence]:
    candidates = [
        seq
        for seq in sequences
        if seq.battery != validation_battery and seq.current.size >= train_length
    ]
    rng = random.Random(data_seed)
    rng.shuffle(candidates)
    if len(candidates) < n_train:
        raise RuntimeError(f"Only {len(candidates)} train cycles for validation battery {validation_battery}.")
    return candidates[:n_train]


def config_fields(prefix: str, profile: str, config: BattNNXJTUConfig) -> dict[str, object]:
    return {
        f"{prefix}_profile": profile,
        f"{prefix}_V0": config.V0,
        f"{prefix}_x0_qb": config.x0[0],
        f"{prefix}_VEOD": config.VEOD,
        f"{prefix}_Rp": config.Rp,
        f"{prefix}_Rs": config.Rs,
        f"{prefix}_Csp": config.Csp,
        f"{prefix}_Cs": config.Cs,
        f"{prefix}_lr": config.lr,
        f"{prefix}_weight_decay": config.weight_decay,
        f"{prefix}_epochs": config.epoch,
    }


def tune_batch(args: argparse.Namespace, BattNN, batch: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    root = args.root if args.root.is_absolute() else ROOT / args.root
    sequences = extract_batch_sequences(root / batch, args.resample_minutes, max(args.min_test_length, 2))
    if not sequences:
        raise RuntimeError(f"No XJTU sequences extracted for {batch}.")
    batteries = sorted({seq.battery for seq in sequences})
    validation_batteries = choose_validation_batteries(batteries, args.validation_count)
    candidates = candidate_configs(args)
    batch_name = batch.lower().replace("-", "")
    search_path = args.results_dir / f"battnn_xjtu_{batch_name}_tuning_search.csv"
    final_path = args.results_dir / f"battnn_xjtu_{batch_name}_tuned.csv"
    for path in (search_path, final_path):
        if path.exists():
            path.unlink()
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(
        f"{batch}: {len(sequences)} cycles, batteries={len(batteries)}, "
        f"validation={validation_batteries}, candidates={len(candidates)}"
    )

    search_rows: list[dict[str, object]] = []
    for candidate_id, (profile, config) in enumerate(candidates, start=1):
        for validation_battery in validation_batteries:
            train_sequences = choose_search_train_sequences(
                sequences,
                validation_battery=validation_battery,
                n_train=args.n_train,
                train_length=args.train_length,
                data_seed=args.data_seed,
            )
            validation_sequences = [
                seq
                for seq in sequences
                if seq.battery == validation_battery and seq.current.size >= args.min_test_length
            ]
            for experiment in range(1, args.search_experiments + 1):
                seed = args.torch_seed + candidate_id * 1000 + experiment
                model, train_summary = train_model(BattNN, config, train_sequences, seed, args.patience, args.verbose_train)
                result = evaluate(model, validation_sequences)
                row = {
                    "timestamp": timestamp,
                    "stage": "search",
                    "method": "BattNN",
                    "dataset": f"XJTU-{batch}",
                    "candidate_id": candidate_id,
                    "validation_battery": validation_battery,
                    "experiment": experiment,
                    "torch_seed": seed,
                    "data_seed": args.data_seed,
                    "n_train": args.n_train,
                    "train_length": args.train_length,
                    "resample_minutes": args.resample_minutes,
                    "min_test_length": args.min_test_length,
                    **config_fields("config", profile, config),
                    **train_summary,
                    **result,
                }
                search_rows.append(row)
                append_row(row, search_path)
                print(
                    f"{batch} search {candidate_id:02d}/{len(candidates)} {profile} "
                    f"lr={config.lr:g} wd={config.weight_decay:g} val={validation_battery}: "
                    f"MAE={result['MAE']:.6f}"
                )

    best_row = min(
        aggregate_search_rows(search_rows),
        key=lambda row: (float(row["validation_MAE_mean"]), float(row["validation_MAE_std"])),
    )
    best_candidate_id = int(best_row["candidate_id"])
    best_profile, best_search_config = candidates[best_candidate_id - 1]
    final_config = replace(best_search_config, epoch=args.epochs_final)
    print(
        f"{batch} best candidate={best_candidate_id} profile={best_profile} "
        f"lr={final_config.lr:g} wd={final_config.weight_decay:g} "
        f"validation MAE={float(best_row['validation_MAE_mean']):.6f} +/- "
        f"{float(best_row['validation_MAE_std']):.6f}"
    )

    final_rows: list[dict[str, object]] = []
    for held_out_battery in batteries:
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
        for experiment in range(1, args.final_experiments + 1):
            seed = args.torch_seed + experiment
            model, train_summary = train_model(BattNN, final_config, train_sequences, seed, args.patience, args.verbose_train)
            result = evaluate(model, test_sequences)
            row = {
                "timestamp": timestamp,
                "stage": "final",
                "method": "BattNN tuned",
                "dataset": f"XJTU-{batch}",
                "held_out_battery": held_out_battery,
                "experiment": experiment,
                "torch_seed": seed,
                "data_seed": args.data_seed,
                "n_train": args.n_train,
                "train_length": args.train_length,
                "resample_minutes": args.resample_minutes,
                "min_test_length": args.min_test_length,
                "selected_candidate_id": best_candidate_id,
                "selected_validation_MAE_mean": best_row["validation_MAE_mean"],
                "selected_validation_MAE_std": best_row["validation_MAE_std"],
                "test_cycles": len(test_sequences),
                **config_fields("config", best_profile, final_config),
                **train_summary,
                **result,
            }
            final_rows.append(row)
            append_row(row, final_path)
            print(
                f"{batch} tuned held_out={held_out_battery} exp={experiment}: "
                f"MAE={result['MAE']:.6f}, MAPE={result['MAPE']:.6f}, RMSE={result['RMSE']:.6f}"
            )

    return search_rows, final_rows


def aggregate_search_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidate_ids = sorted({int(row["candidate_id"]) for row in rows})
    aggregates: list[dict[str, object]] = []
    for candidate_id in candidate_ids:
        subset = [row for row in rows if int(row["candidate_id"]) == candidate_id]
        mae = np.array([float(row["MAE"]) for row in subset], dtype=np.float64)
        first = subset[0]
        aggregates.append(
            {
                "candidate_id": candidate_id,
                "validation_MAE_mean": float(mae.mean()),
                "validation_MAE_std": float(mae.std()),
                "validation_runs": len(subset),
                **{key: first[key] for key in first if key.startswith("config_")},
            }
        )
    return aggregates


def save_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_row(row: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = sorted(row.keys())
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_final(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        subset = [row for row in rows if str(row["dataset"]) == dataset]
        arr = np.array([[float(row[m]) for m in ("MAE", "MAPE", "RMSE")] for row in subset], dtype=np.float64)
        first = subset[0]
        summary_rows.append(
            {
                "dataset": dataset,
                "method": "BattNN tuned",
                "n": len(subset),
                "MAE_mean": float(arr[:, 0].mean()),
                "MAE_std": float(arr[:, 0].std()),
                "MAPE_mean": float(arr[:, 1].mean()),
                "MAPE_std": float(arr[:, 1].std()),
                "RMSE_mean": float(arr[:, 2].mean()),
                "RMSE_std": float(arr[:, 2].std()),
                "selected_candidate_id": first["selected_candidate_id"],
                "selected_validation_MAE_mean": first["selected_validation_MAE_mean"],
                "selected_validation_MAE_std": first["selected_validation_MAE_std"],
                **{key: first[key] for key in first if key.startswith("config_")},
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    BattNN = import_battnn()

    all_search_rows: list[dict[str, object]] = []
    all_final_rows: list[dict[str, object]] = []
    for batch in args.batches:
        search_rows, final_rows = tune_batch(args, BattNN, batch)
        all_search_rows.extend(search_rows)
        all_final_rows.extend(final_rows)
        batch_name = batch.lower().replace("-", "")
        save_rows(search_rows, args.results_dir / f"battnn_xjtu_{batch_name}_tuning_search.csv")
        save_rows(final_rows, args.results_dir / f"battnn_xjtu_{batch_name}_tuned.csv")

    summary_rows = summarize_final(all_final_rows)
    save_rows(summary_rows, args.results_dir / "battnn_xjtu_tuned_summary.csv")
    for row in summary_rows:
        print(
            f"{row['dataset']} tuned summary: "
            f"MAE={float(row['MAE_mean']):.6f} +/- {float(row['MAE_std']):.6f}, "
            f"MAPE={float(row['MAPE_mean']):.6f} +/- {float(row['MAPE_std']):.6f}, "
            f"RMSE={float(row['RMSE_mean']):.6f} +/- {float(row['RMSE_std']):.6f}"
        )


if __name__ == "__main__":
    main()
