"""Create summary tables and representative SAGD-BLS prediction plots."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_sagd_bls_battnn_datasets import nasa_splits
from run_sagd_bls_sim import iter_test_data, load_train_data
from run_sagd_bls_xjtu import choose_train_sequences, extract_batch_sequences
from sagd_bls_battery import SAGDBLS


RESULTS_DIR = Path("results")
FIGURE_DIR = RESULTS_DIR / "figures"
SUMMARY_CSV = RESULTS_DIR / "overall_battnn_vs_sagd_summary.csv"
SUMMARY_MD = RESULTS_DIR / "overall_battnn_vs_sagd_summary.md"
XJTU_VARIANT_CSV = RESULTS_DIR / "xjtu_context_variant_summary.csv"
XJTU_BATCH5_SUMMARY_CSV = RESULTS_DIR / "xjtu_batch5_battnn_vs_sagd_summary.csv"
XJTU_BATCH1_SUMMARY_CSV = RESULTS_DIR / "xjtu_batch1_battnn_vs_sagd_summary.csv"
BATTNN_XJTU_TUNED_SUMMARY_CSV = RESULTS_DIR / "battnn_xjtu_tuned_summary.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def mean_std(rows: list[dict[str, str]], metric: str) -> tuple[float, float]:
    values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std())


def build_summary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    batt_sim = read_csv(RESULTS_DIR / "battnn_sim_reproduced.csv")
    sagd_sim = [
        row
        for row in read_csv(RESULTS_DIR / "sagd_bls_sim_results.csv")
        if row["map_activation"] == "tanh" and row["enhance_activation"] == "sigmoid"
    ]
    add_pair(rows, "SimData", "BattNN reproduced", batt_sim, "SAGD-BLS tanh/sigmoid", sagd_sim)

    batt_lab_nasa = read_csv(RESULTS_DIR / "battnn_lab_nasa_reproduced.csv")
    sagd_lab_nasa = read_csv(RESULTS_DIR / "sagd_bls_battnn_datasets.csv")
    batt_lab = [row for row in batt_lab_nasa if row["dataset"] == "Lab" and float(row["MAE"]) < 0.1]
    sagd_lab = [row for row in sagd_lab_nasa if row["dataset"] == "Lab"]
    add_pair(rows, "LabData (local B1-B6)", "BattNN reproduced clean", batt_lab, "SAGD-BLS tanh/sigmoid", sagd_lab)

    batt_nasa = [row for row in batt_lab_nasa if row["dataset"] == "NASA"]
    sagd_nasa = [row for row in sagd_lab_nasa if row["dataset"] == "NASA"]
    add_pair(rows, "NASAData", "BattNN reproduced", batt_nasa, "SAGD-BLS tanh/sigmoid", sagd_nasa)

    batt_xjtu = read_csv(preferred_xjtu_battnn_path("batch5"))
    sagd_xjtu = read_csv(RESULTS_DIR / "sagd_bls_xjtu_batch5_v2_cycle.csv")
    add_pair(rows, "XJTU Batch-5", preferred_xjtu_battnn_label("batch5"), batt_xjtu, "SAGD-BLS + cycle context", sagd_xjtu)

    batch1_batt = preferred_xjtu_battnn_path("batch1")
    batch1_sagd = RESULTS_DIR / "sagd_bls_xjtu_batch1_v2_cycle.csv"
    if batch1_batt.exists() and batch1_sagd.exists():
        add_pair(
            rows,
            "XJTU Batch-1",
            preferred_xjtu_battnn_label("batch1"),
            read_csv(batch1_batt),
            "SAGD-BLS + cycle context",
            read_csv(batch1_sagd),
        )
    return rows


def build_xjtu_variant_rows() -> list[dict[str, object]]:
    batch5_batt = preferred_xjtu_battnn_path("batch5")
    batch1_batt = preferred_xjtu_battnn_path("batch1")
    variants = [
        (
            "XJTU Batch-5",
            "current-only",
            batch5_batt,
            RESULTS_DIR / "sagd_bls_xjtu_batch5.csv",
        ),
        (
            "XJTU Batch-5",
            "cycle context",
            batch5_batt,
            RESULTS_DIR / "sagd_bls_xjtu_batch5_v2_cycle.csv",
        ),
        (
            "XJTU Batch-5",
            "cycle + early-voltage context",
            batch5_batt,
            RESULTS_DIR / "sagd_bls_xjtu_batch5_v2_cycle_voltage.csv",
        ),
        (
            "XJTU Batch-1",
            "current-only",
            batch1_batt,
            RESULTS_DIR / "sagd_bls_xjtu_batch1.csv",
        ),
        (
            "XJTU Batch-1",
            "cycle context",
            batch1_batt,
            RESULTS_DIR / "sagd_bls_xjtu_batch1_v2_cycle.csv",
        ),
        (
            "XJTU Batch-1",
            "cycle + early-voltage context",
            batch1_batt,
            RESULTS_DIR / "sagd_bls_xjtu_batch1_v2_cycle_voltage.csv",
        ),
        (
            "XJTU Batch-1",
            "cycle + early-voltage context, l2=1e-2",
            batch1_batt,
            RESULTS_DIR / "sagd_bls_xjtu_batch1_v2_cycle_voltage_l2_1e-2.csv",
        ),
    ]

    rows: list[dict[str, object]] = []
    for dataset, variant, batt_path, sagd_path in variants:
        if not batt_path.exists() or not sagd_path.exists():
            continue
        batt_rows = read_csv(batt_path)
        sagd_rows = read_csv(sagd_path)
        batt_mae, batt_mae_std = mean_std(batt_rows, "MAE")
        sagd_mae, sagd_mae_std = mean_std(sagd_rows, "MAE")
        batt_mape, batt_mape_std = mean_std(batt_rows, "MAPE")
        sagd_mape, sagd_mape_std = mean_std(sagd_rows, "MAPE")
        batt_rmse, batt_rmse_std = mean_std(batt_rows, "RMSE")
        sagd_rmse, sagd_rmse_std = mean_std(sagd_rows, "RMSE")
        rows.append(
            {
                "dataset": dataset,
                "sagd_variant": variant,
                "battnn_MAE_mean": batt_mae,
                "battnn_MAE_std": batt_mae_std,
                "sagd_MAE_mean": sagd_mae,
                "sagd_MAE_std": sagd_mae_std,
                "MAE_reduction_percent": (batt_mae - sagd_mae) / batt_mae * 100.0,
                "battnn_MAPE_mean": batt_mape,
                "battnn_MAPE_std": batt_mape_std,
                "sagd_MAPE_mean": sagd_mape,
                "sagd_MAPE_std": sagd_mape_std,
                "MAPE_reduction_percent": (batt_mape - sagd_mape) / batt_mape * 100.0,
                "battnn_RMSE_mean": batt_rmse,
                "battnn_RMSE_std": batt_rmse_std,
                "sagd_RMSE_mean": sagd_rmse,
                "sagd_RMSE_std": sagd_rmse_std,
                "RMSE_reduction_percent": (batt_rmse - sagd_rmse) / batt_rmse * 100.0,
                "sagd_rows": len(sagd_rows),
            }
        )
    return rows


def preferred_xjtu_battnn_path(batch_key: str) -> Path:
    tuned = {
        "batch5": RESULTS_DIR / "battnn_xjtu_batch5_tuned.csv",
        "batch1": RESULTS_DIR / "battnn_xjtu_batch1_tuned.csv",
    }[batch_key]
    if tuned.exists():
        return tuned
    return {
        "batch5": RESULTS_DIR / "battnn_xjtu_batch5.csv",
        "batch1": RESULTS_DIR / "battnn_xjtu_batch1.csv",
    }[batch_key]


def preferred_xjtu_battnn_label(batch_key: str) -> str:
    return "BattNN tuned adapter" if preferred_xjtu_battnn_path(batch_key).name.endswith("_tuned.csv") else "BattNN adapter"


def add_pair(
    rows: list[dict[str, object]],
    dataset: str,
    batt_label: str,
    batt_rows: list[dict[str, str]],
    sagd_label: str,
    sagd_rows: list[dict[str, str]],
) -> None:
    batt_mae, batt_mae_std = mean_std(batt_rows, "MAE")
    sagd_mae, sagd_mae_std = mean_std(sagd_rows, "MAE")
    batt_mape, batt_mape_std = mean_std(batt_rows, "MAPE")
    sagd_mape, sagd_mape_std = mean_std(sagd_rows, "MAPE")
    batt_rmse, batt_rmse_std = mean_std(batt_rows, "RMSE")
    sagd_rmse, sagd_rmse_std = mean_std(sagd_rows, "RMSE")
    rows.append(
        {
            "dataset": dataset,
            "battnn_label": batt_label,
            "sagd_label": sagd_label,
            "battnn_n": len(batt_rows),
            "sagd_n": len(sagd_rows),
            "battnn_MAE_mean": batt_mae,
            "battnn_MAE_std": batt_mae_std,
            "sagd_MAE_mean": sagd_mae,
            "sagd_MAE_std": sagd_mae_std,
            "MAE_reduction_percent": (batt_mae - sagd_mae) / batt_mae * 100.0,
            "battnn_MAPE_mean": batt_mape,
            "battnn_MAPE_std": batt_mape_std,
            "sagd_MAPE_mean": sagd_mape,
            "sagd_MAPE_std": sagd_mape_std,
            "MAPE_reduction_percent": (batt_mape - sagd_mape) / batt_mape * 100.0,
            "battnn_RMSE_mean": batt_rmse,
            "battnn_RMSE_std": batt_rmse_std,
            "sagd_RMSE_mean": sagd_rmse,
            "sagd_RMSE_std": sagd_rmse_std,
            "RMSE_reduction_percent": (batt_rmse - sagd_rmse) / batt_rmse * 100.0,
        }
    )


def save_summary(rows: list[dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = list(rows[0].keys())
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# BattNN vs SAGD-BLS Summary",
        "",
        "| Dataset | BattNN MAE | SAGD-BLS MAE | MAE Reduction |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | "
            f"{row['battnn_MAE_mean']:.6f} +/- {row['battnn_MAE_std']:.6f} | "
            f"{row['sagd_MAE_mean']:.6f} +/- {row['sagd_MAE_std']:.6f} | "
            f"{row['MAE_reduction_percent']:.1f}% |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_xjtu_variant_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with XJTU_VARIANT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metric_pair_summary(
    dataset: str,
    sagd_label: str,
    sagd_rows: list[dict[str, str]],
    batt_label: str,
    batt_rows: list[dict[str, str]],
    path: Path,
) -> None:
    fieldnames = ["dataset", "method", "n", "MAE_mean", "MAE_std", "MAPE_mean", "MAPE_std", "RMSE_mean", "RMSE_std"]
    output_rows = []
    for label, rows in [(sagd_label, sagd_rows), (batt_label, batt_rows)]:
        mae, mae_std = mean_std(rows, "MAE")
        mape, mape_std = mean_std(rows, "MAPE")
        rmse, rmse_std = mean_std(rows, "RMSE")
        output_rows.append(
            {
                "dataset": dataset,
                "method": label,
                "n": len(rows),
                "MAE_mean": mae,
                "MAE_std": mae_std,
                "MAPE_mean": mape,
                "MAPE_std": mape_std,
                "RMSE_mean": rmse,
                "RMSE_std": rmse_std,
            }
        )

    sagd_mae, _ = mean_std(sagd_rows, "MAE")
    batt_mae, _ = mean_std(batt_rows, "MAE")
    sagd_mape, _ = mean_std(sagd_rows, "MAPE")
    batt_mape, _ = mean_std(batt_rows, "MAPE")
    sagd_rmse, _ = mean_std(sagd_rows, "RMSE")
    batt_rmse, _ = mean_std(batt_rows, "RMSE")
    output_rows.append(
        {
            "dataset": dataset,
            "method": "SAGD_vs_BattNN_reduction_percent",
            "n": len(sagd_rows),
            "MAE_mean": (batt_mae - sagd_mae) / batt_mae * 100.0,
            "MAE_std": "",
            "MAPE_mean": (batt_mape - sagd_mape) / batt_mape * 100.0,
            "MAPE_std": "",
            "RMSE_mean": (batt_rmse - sagd_rmse) / batt_rmse * 100.0,
            "RMSE_std": "",
        }
    )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def save_xjtu_best_pair_summaries() -> None:
    save_metric_pair_summary(
        "XJTU-Batch-5",
        "SAGD-BLS_tanh_sigmoid_cycle_context",
        read_csv(RESULTS_DIR / "sagd_bls_xjtu_batch5_v2_cycle.csv"),
        preferred_xjtu_battnn_label("batch5").replace(" ", "_"),
        read_csv(preferred_xjtu_battnn_path("batch5")),
        XJTU_BATCH5_SUMMARY_CSV,
    )


def save_battnn_xjtu_tuned_summary() -> None:
    tuned_paths = [
        RESULTS_DIR / "battnn_xjtu_batch5_tuned.csv",
        RESULTS_DIR / "battnn_xjtu_batch1_tuned.csv",
    ]
    rows: list[dict[str, object]] = []
    for path in tuned_paths:
        if not path.exists():
            continue
        data = read_csv(path)
        mae, mae_std = mean_std(data, "MAE")
        mape, mape_std = mean_std(data, "MAPE")
        rmse, rmse_std = mean_std(data, "RMSE")
        first = data[0]
        rows.append(
            {
                "dataset": first["dataset"],
                "method": "BattNN tuned adapter",
                "n": len(data),
                "MAE_mean": mae,
                "MAE_std": mae_std,
                "MAPE_mean": mape,
                "MAPE_std": mape_std,
                "RMSE_mean": rmse,
                "RMSE_std": rmse_std,
                "selected_candidate_id": first.get("selected_candidate_id", ""),
                "selected_validation_MAE_mean": first.get("selected_validation_MAE_mean", ""),
                "config_profile": first.get("config_profile", ""),
                "config_lr": first.get("config_lr", ""),
                "config_weight_decay": first.get("config_weight_decay", ""),
                "config_epochs": first.get("config_epochs", ""),
            }
        )
    if rows:
        save_rows_generic(rows, BATTNN_XJTU_TUNED_SUMMARY_CSV)


def save_rows_generic(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_metric_pair_summary(
        "XJTU-Batch-1",
        "SAGD-BLS_tanh_sigmoid_cycle_context",
        read_csv(RESULTS_DIR / "sagd_bls_xjtu_batch1_v2_cycle.csv"),
        preferred_xjtu_battnn_label("batch1").replace(" ", "_"),
        read_csv(preferred_xjtu_battnn_path("batch1")),
        XJTU_BATCH1_SUMMARY_CSV,
    )


def train_default_sagd(currents, voltages, train_length=60, seed=2, contexts=None) -> SAGDBLS:
    model = SAGDBLS(
        n_map=100,
        n_enhance=100,
        map_activation="tanh",
        enhance_activation="sigmoid",
        seed=seed,
        epochs=20_000,
        learning_rate=0.01,
        l2=1e-3,
        smooth_l1_delta=1.0,
    )
    model.fit(currents, voltages, train_length=train_length, contexts=contexts)
    return model


def cycle_context(seq) -> np.ndarray:
    return np.array([seq.cycle / 500.0, seq.duration_min / 60.0], dtype=np.float64)


def plot_prediction(title: str, current: np.ndarray, voltage: np.ndarray, pred: np.ndarray, path: Path) -> None:
    x = np.arange(voltage.size)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(x, voltage, label="True voltage", linewidth=1.8)
    axes[0].plot(x, pred, label="SAGD-BLS prediction", linewidth=1.5)
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, current, color="#2f6f9f", linewidth=1.2)
    axes[1].set_ylabel("Current (A)")
    axes[1].set_xlabel("Resampled step")
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_prediction_plots() -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    sim_train_x, sim_train_y, _ = load_train_data(Path(r"BattNN\data\SimData\current=[2, 8] len=[60, 200] train"), 30, 60)
    sim_model = train_default_sagd(sim_train_x, sim_train_y, seed=2)
    sim_current, sim_voltage = next(iter_test_data(Path(r"BattNN\data\SimData\current=[2, 8] len=[60, 200] test")))
    sim_pred = sim_model.predict(sim_current)
    path = FIGURE_DIR / "sagd_sim_representative.png"
    plot_prediction("SimData representative cycle", sim_current, sim_voltage, sim_pred, path)
    paths.append(path)

    nasa_dataset, nasa_split, nasa_train_x, nasa_train_y, nasa_test = next(nasa_splits(Path(r"BattNN\data\NASA11"), 30, 60))
    nasa_model = train_default_sagd(nasa_train_x, nasa_train_y, seed=2)
    nasa_current, nasa_voltage = nasa_test[0]
    nasa_pred = nasa_model.predict(nasa_current)
    path = FIGURE_DIR / "sagd_nasa_rw3_representative.png"
    plot_prediction(f"{nasa_dataset} {nasa_split} representative cycle", nasa_current, nasa_voltage, nasa_pred, path)
    paths.append(path)

    xjtu_sequences = extract_batch_sequences(Path(r"XJTU battery dataset\Batch-5"), resample_minutes=0.5, min_length=20)
    train_sequences = choose_train_sequences(xjtu_sequences, "RW_battery-1", 30, 60, 2022)
    xjtu_model = train_default_sagd(
        [seq.current for seq in train_sequences],
        [seq.voltage for seq in train_sequences],
        seed=2,
        contexts=[cycle_context(seq) for seq in train_sequences],
    )
    xjtu_test = next(seq for seq in xjtu_sequences if seq.battery == "RW_battery-1")
    xjtu_pred = xjtu_model.predict(xjtu_test.current, context=cycle_context(xjtu_test))
    path = FIGURE_DIR / "sagd_xjtu_batch5_rw1_representative.png"
    plot_prediction("XJTU Batch-5 RW_battery-1 representative cycle", xjtu_test.current, xjtu_test.voltage, xjtu_pred, path)
    paths.append(path)

    batch1_sequences = extract_batch_sequences(Path(r"XJTU battery dataset\Batch-1"), resample_minutes=0.5, min_length=20)
    batch1_train = choose_train_sequences(batch1_sequences, "2C_battery-1", 30, 60, 2022)
    batch1_model = train_default_sagd(
        [seq.current for seq in batch1_train],
        [seq.voltage for seq in batch1_train],
        seed=2,
        contexts=[cycle_context(seq) for seq in batch1_train],
    )
    batch1_test = next(seq for seq in batch1_sequences if seq.battery == "2C_battery-1")
    batch1_pred = batch1_model.predict(batch1_test.current, context=cycle_context(batch1_test))
    path = FIGURE_DIR / "sagd_xjtu_batch1_2c1_representative.png"
    plot_prediction("XJTU Batch-1 2C_battery-1 representative cycle", batch1_test.current, batch1_test.voltage, batch1_pred, path)
    paths.append(path)

    return paths


def main() -> None:
    summary_rows = build_summary_rows()
    save_summary(summary_rows)
    save_xjtu_variant_summary(build_xjtu_variant_rows())
    save_xjtu_best_pair_summaries()
    save_battnn_xjtu_tuned_summary()
    plot_paths = make_prediction_plots()
    print(f"saved summary: {SUMMARY_CSV}")
    print(f"saved markdown: {SUMMARY_MD}")
    print(f"saved XJTU variant summary: {XJTU_VARIANT_CSV}")
    print(f"saved XJTU Batch-5 summary: {XJTU_BATCH5_SUMMARY_CSV}")
    print(f"saved XJTU Batch-1 summary: {XJTU_BATCH1_SUMMARY_CSV}")
    print(f"saved BattNN XJTU tuned summary: {BATTNN_XJTU_TUNED_SUMMARY_CSV}")
    for path in plot_paths:
        print(f"saved figure: {path}")


if __name__ == "__main__":
    main()
