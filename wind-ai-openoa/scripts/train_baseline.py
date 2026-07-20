#!/usr/bin/env python3
"""Train and evaluate a CPU-friendly wind-turbine power baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_ai.evaluation import regression_metrics, summarize_predictions
from wind_ai.features import (
    EXPERIMENT_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_experiment_features,
    build_power_features,
)
from wind_ai.paths import MODELS_DIR, RESULTS_DIR


def save_grouped_evaluation_tables(results: pd.DataFrame) -> list[Path]:
    """Save reusable evaluation tables for report and diagnostic analysis."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    if "Wind_turbine_name" in results.columns:
        path = RESULTS_DIR / "evaluation_by_turbine.csv"
        summarize_predictions(results, "Wind_turbine_name").to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        artifacts.append(path)

    if "Ws_avg" in results.columns:
        wind_speed_results = results.copy()
        wind_speed_results["wind_speed_bin"] = pd.cut(
            wind_speed_results["Ws_avg"],
            bins=[0, 3, 5, 8, 12, 15, 40],
            labels=["0-3", "3-5", "5-8", "8-12", "12-15", "15-40"],
            right=False,
            include_lowest=True,
        )
        path = RESULTS_DIR / "evaluation_by_wind_speed_bin.csv"
        summarize_predictions(wind_speed_results, "wind_speed_bin").to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        artifacts.append(path)

    power_results = results.copy()
    power_results["power_bin"] = pd.cut(
        power_results["actual_kw"],
        bins=[-100, 100, 500, 1000, 1500, 1900, 2500],
        labels=[
            "-100-100",
            "100-500",
            "500-1000",
            "1000-1500",
            "1500-1900",
            "1900-2500",
        ],
        right=False,
        include_lowest=True,
    )
    path = RESULTS_DIR / "evaluation_by_power_bin.csv"
    summarize_predictions(power_results, "power_bin").to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )
    artifacts.append(path)

    if "Date_time" in results.columns:
        monthly_results = results.copy()
        monthly_results["month"] = (
            pd.to_datetime(monthly_results["Date_time"]).dt.to_period("M").astype(str)
        )
        path = RESULTS_DIR / "evaluation_by_month.csv"
        summarize_predictions(monthly_results, "month").to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        artifacts.append(path)

    if "era5_ws_100m" in results.columns:
        era5_results = results.copy()
        era5_results["era5_wind_speed_bin"] = pd.cut(
            era5_results["era5_ws_100m"],
            bins=[0, 3, 5, 8, 12, 15, 40],
            labels=["0-3", "3-5", "5-8", "8-12", "12-15", "15-40"],
            right=False,
            include_lowest=True,
        )
        path = RESULTS_DIR / "evaluation_by_era5_wind_speed_bin.csv"
        summarize_predictions(era5_results, "era5_wind_speed_bin").to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        artifacts.append(path)

    return artifacts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the wind-power baseline model."
    )
    parser.add_argument(
        "--feature-set",
        choices=["enhanced", "scada"],
        default="enhanced",
        help=(
            "Feature set used for training. 'enhanced' uses SCADA, ERA5, "
            "MERRA2 and asset metadata; 'scada' keeps the SCADA-only baseline."
        ),
    )
    return parser.parse_args()


def save_evaluation_artifacts(
    test: pd.DataFrame,
    predictions,
) -> pd.DataFrame:
    """Save prediction results and evaluation tables."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        {
            "actual_kw": test[TARGET_COLUMN].to_numpy(),
            "predicted_kw": predictions,
        }
    )

    results["residual_kw"] = (
        results["actual_kw"] - results["predicted_kw"]
    )
    results["absolute_error_kw"] = results["residual_kw"].abs()

    # Preserve useful identifying columns when build_power_features keeps them.
    metadata_columns = [
        column
        for column in [
            "Date_time",
            "Wind_turbine_name",
            "Ws_avg",
            "era5_ws_100m",
            "merra2_ws_50m",
            "asset_rated_power_kw",
        ]
        if column in test.columns
    ]

    if metadata_columns:
        metadata = test[metadata_columns].reset_index(drop=True)
        results = pd.concat([metadata, results], axis=1)

    results.to_csv(
        RESULTS_DIR / "power_baseline_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grouped_artifacts = save_grouped_evaluation_tables(results)

    results.attrs["grouped_artifacts"] = grouped_artifacts
    return results


def main() -> int:
    args = parse_args()

    try:
        from wind_ai.openoa_loader import load_dataframes
    except ModuleNotFoundError as exc:
        if exc.name == "openoa":
            print(
                "Missing dependency: OpenOA. Run `pip install -r requirements.txt` "
                "or create the Conda environment from `environment.yml` first.",
                file=sys.stderr,
            )
            return 1
        raise

    frames = load_dataframes()
    scada = frames["scada"]

    # Ensure the split is genuinely chronological.
    if "Date_time" in scada.columns:
        scada = scada.copy()
        scada["Date_time"] = pd.to_datetime(
            scada["Date_time"],
            errors="coerce",
        )

        sort_columns = ["Date_time"]

        if "Wind_turbine_name" in scada.columns:
            sort_columns.append("Wind_turbine_name")

        scada = scada.sort_values(sort_columns).reset_index(drop=True)

    if args.feature_set == "enhanced":
        dataset = build_experiment_features(frames)
        feature_columns = EXPERIMENT_FEATURE_COLUMNS
        data_sources = [
            "scada",
            "era5_reanalysis",
            "merra2_reanalysis",
            "asset_metadata",
        ]
    else:
        dataset = build_power_features(scada)
        feature_columns = FEATURE_COLUMNS
        data_sources = ["scada"]

    # Keep chronological order after feature engineering as well.
    if "Date_time" in dataset.columns:
        dataset = dataset.sort_values("Date_time").reset_index(drop=True)

    # Time-ordered split avoids leaking future observations into training.
    split = int(len(dataset) * 0.8)

    if split <= 0 or split >= len(dataset):
        raise ValueError(
            f"Dataset is too small for an 80/20 split: {len(dataset)} rows."
        )

    train = dataset.iloc[:split].copy()
    test = dataset.iloc[split:].copy()

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    )

    model.fit(
        train[feature_columns],
        train[TARGET_COLUMN],
    )

    predictions = model.predict(test[feature_columns])

    test_metrics = regression_metrics(test[TARGET_COLUMN], pd.Series(predictions))

    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae_kw": test_metrics["mae_kw"],
        "rmse_kw": test_metrics["rmse_kw"],
        "r2": test_metrics["r2"],
        "mean_residual_kw": test_metrics["mean_residual_kw"],
        "median_absolute_error_kw": test_metrics["median_absolute_error_kw"],
        "p95_absolute_error_kw": test_metrics["p95_absolute_error_kw"],
        "max_absolute_error_kw": test_metrics["max_absolute_error_kw"],
        "feature_set": args.feature_set,
        "features": feature_columns,
        "data_sources": data_sources,
        "raw_scada_rows": int(len(scada)),
        "feature_rows": int(len(dataset)),
        "dropped_rows_after_feature_engineering": int(len(scada) - len(dataset)),
    }

    if "Date_time" in train.columns:
        metrics["train_start"] = str(train["Date_time"].min())
        metrics["train_end"] = str(train["Date_time"].max())
        metrics["test_start"] = str(test["Date_time"].min())
        metrics["test_end"] = str(test["Date_time"].max())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODELS_DIR / "power_baseline.joblib",
    )

    metrics_path = RESULTS_DIR / "power_baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = save_evaluation_artifacts(
        test,
        predictions,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\nGenerated artifacts:")
    print(f"- Model:      {MODELS_DIR / 'power_baseline.joblib'}")
    print(f"- Metrics:    {metrics_path}")
    print(
        f"- Predictions:{RESULTS_DIR / 'power_baseline_predictions.csv'}"
    )

    grouped_artifacts = results.attrs.get("grouped_artifacts", [])
    if grouped_artifacts:
        print("- Grouped evaluation tables:")
        for artifact in grouped_artifacts:
            print(f"  - {artifact}")

    print("\nResidual summary:")
    print(results["residual_kw"].describe().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
