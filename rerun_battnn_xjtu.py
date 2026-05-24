"""Run BattNN on the same extracted XJTU Batch-5 splits as SAGD-BLS."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import scienceplots  # noqa: F401
import numpy as np
import torch

from run_sagd_bls_xjtu import XJTU_ROOT, choose_train_sequences, extract_batch_sequences


ROOT = Path(__file__).resolve().parent
BATTNN_DIR = ROOT / "BattNN"
RESULTS_CSV = ROOT / "results" / "battnn_xjtu_batch5.csv"


@dataclass
class BattNNXJTUArgs:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BattNN baseline on extracted XJTU Batch-5 cycles.")
    parser.add_argument("--batch", default="Batch-5")
    parser.add_argument("--root", type=Path, default=XJTU_ROOT)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--experiments", type=int, default=5)
    parser.add_argument("--torch-seed", type=int, default=2022)
    parser.add_argument("--data-seed", type=int, default=2022)
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--train-length", type=int, default=60)
    parser.add_argument("--resample-minutes", type=float, default=0.5)
    parser.add_argument("--min-test-length", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--verbose-train", action="store_true")
    return parser.parse_args()


def import_battnn():
    os.chdir(BATTNN_DIR)
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


def evaluate(model, test_sequences) -> dict[str, float]:
    rows = []
    torch.nn.Module.train(model, False)
    with torch.no_grad():
        for seq in test_sequences:
            c_tensor = torch.from_numpy(seq.current.astype(np.float32))
            pred, _ = model.predict(c_tensor.view(1, -1))
            pred = pred.detach().cpu().numpy()[0]
            rows.append(metrics(seq.voltage, pred))
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("MAE", "MAPE", "MSE", "RMSE")
    } | {"n_sequences": float(len(rows))}


def train_and_eval(BattNN, config: BattNNXJTUArgs, train_sequences, test_sequences, verbose: bool):
    model = BattNN(config)
    train_x = np.asarray([seq.current[: config.seq_len] for seq in train_sequences], dtype=np.float32)
    train_y = np.asarray([seq.voltage[: config.seq_len] for seq in train_sequences], dtype=np.float32)
    model.get_data(x=torch.from_numpy(train_x), label=torch.from_numpy(train_y))
    if verbose:
        model.train()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            model.train()
    if model.best_state is not None:
        model.load_state_dict(model.best_state["net"])
    return evaluate(model, test_sequences)


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
        "XJTU BattNN summary: "
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
    if not args.results_csv.is_absolute():
        args.results_csv = ROOT / args.results_csv
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    root = args.root if args.root.is_absolute() else ROOT / args.root
    batch_dir = root / args.batch
    sequences = extract_batch_sequences(
        batch_dir,
        resample_minutes=args.resample_minutes,
        min_length=max(args.min_test_length, 2),
    )
    batteries = sorted({seq.battery for seq in sequences})
    print(f"loaded {len(sequences)} XJTU cycles from {batch_dir}; batteries={len(batteries)}")
    if not sequences:
        raise RuntimeError(f"No XJTU cycles were extracted from {batch_dir}.")

    BattNN = import_battnn()

    rows: list[dict[str, object]] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
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
        for experiment in range(1, args.experiments + 1):
            seed = args.torch_seed + experiment
            torch.manual_seed(seed)
            config = BattNNXJTUArgs(
                batch_size=args.n_train,
                seq_len=args.train_length,
                epoch=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            result = train_and_eval(BattNN, config, train_sequences, test_sequences, args.verbose_train)
            row = {
                "timestamp": timestamp,
                "method": "BattNN",
                "dataset": f"XJTU-{args.batch}",
                "held_out_battery": held_out_battery,
                "experiment": experiment,
                "torch_seed": seed,
                "data_seed": args.data_seed,
                "n_train": args.n_train,
                "train_length": args.train_length,
                "resample_minutes": args.resample_minutes,
                "min_test_length": args.min_test_length,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "test_cycles": len(test_sequences),
                "torch_version": torch.__version__,
                **result,
            }
            rows.append(row)
            print(
                f"held_out={held_out_battery} experiment={experiment}: "
                f"MAE={row['MAE']:.6f}, MAPE={row['MAPE']:.6f}, RMSE={row['RMSE']:.6f}"
            )

    save_rows(rows, args.results_csv)
    summarize(rows)
    print(f"saved results: {args.results_csv}")


if __name__ == "__main__":
    main()
