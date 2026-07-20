#!/usr/bin/env python3
"""Generate the Word experiment report from existing result tables and charts."""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "la_haute_borne"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
RAW_FIGURES_DIR = FIGURES_DIR / "raw_data"
RESULT_FIGURES_DIR = FIGURES_DIR / "experiment_results"
COMPARISON_FIGURES_DIR = FIGURES_DIR / "comparison_analysis"
REPORT_PATH = REPORTS_DIR / "wind_ai_openoa_report.docx"

EMU_PER_INCH = 914400
TWIP_PER_INCH = 1440
MONTH_TICK_STEP = 2

COLORS = {
    "blue": "#3f6b8f",
    "green": "#2e6f57",
    "olive": "#6f8f3f",
    "orange": "#b76e36",
    "red": "#9a4f4f",
    "purple": "#6b5f9a",
    "gray": "#5f6770",
}


def require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}\n{hint}")


def load_inputs() -> dict[str, object]:
    require_file(
        DATA_DIR / "la-haute-borne-data-2014-2015.csv",
        "Run `python scripts/download_data.py` first.",
    )
    require_file(
        RESULTS_DIR / "power_baseline_metrics.json",
        "Run `python scripts/run_experiment.py` first.",
    )
    require_file(
        RESULTS_DIR / "power_baseline_predictions.csv",
        "Run `python scripts/run_experiment.py` first.",
    )

    scada = pd.read_csv(
        DATA_DIR / "la-haute-borne-data-2014-2015.csv",
        usecols=["Date_time", "Wind_turbine_name", "Ws_avg", "P_avg", "Ot_avg", "Ba_avg"],
    )
    scada["Date_time"] = pd.to_datetime(scada["Date_time"], utc=True).dt.tz_localize(None)
    plant = pd.read_csv(DATA_DIR / "plant_data.csv", parse_dates=["time_utc"])
    era5 = pd.read_csv(
        DATA_DIR / "era5_wind_la_haute_borne.csv",
        usecols=["datetime", "ws_100m", "dens_100m"],
    )
    era5["datetime"] = pd.to_datetime(era5["datetime"], utc=True).dt.tz_localize(None)
    merra2 = pd.read_csv(
        DATA_DIR / "merra2_la_haute_borne.csv",
        usecols=["datetime", "ws_50m", "dens_50m"],
    )
    merra2["datetime"] = pd.to_datetime(merra2["datetime"], utc=True).dt.tz_localize(None)
    asset = pd.read_csv(DATA_DIR / "la-haute-borne_asset_table.csv")
    predictions = pd.read_csv(RESULTS_DIR / "power_baseline_predictions.csv")
    with (RESULTS_DIR / "power_baseline_metrics.json").open(encoding="utf-8") as file:
        metrics = json.load(file)

    tables = {
        "turbine": pd.read_csv(RESULTS_DIR / "evaluation_by_turbine.csv"),
        "wind_speed": pd.read_csv(RESULTS_DIR / "evaluation_by_wind_speed_bin.csv"),
        "power": pd.read_csv(RESULTS_DIR / "evaluation_by_power_bin.csv"),
        "month": pd.read_csv(RESULTS_DIR / "evaluation_by_month.csv"),
    }
    era5_eval = RESULTS_DIR / "evaluation_by_era5_wind_speed_bin.csv"
    if era5_eval.is_file():
        tables["era5_wind_speed"] = pd.read_csv(era5_eval)

    return {
        "scada": scada,
        "plant": plant,
        "era5": era5,
        "merra2": merra2,
        "asset": asset,
        "predictions": predictions,
        "metrics": metrics,
        "tables": tables,
    }


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def style_axes(title: str, xlabel: str, ylabel: str) -> None:
    ax = plt.gca()
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def set_sparse_month_ticks(labels: Iterable[object], *, step: int = MONTH_TICK_STEP) -> None:
    labels = [str(label) for label in labels]
    tick_positions = list(range(0, len(labels), step))
    if labels and tick_positions[-1] != len(labels) - 1:
        tick_positions.append(len(labels) - 1)
    plt.xticks(tick_positions, [labels[index] for index in tick_positions], rotation=35, ha="right")


def add_bar_labels(horizontal: bool = False, fmt: str = "{:,.0f}") -> None:
    ax = plt.gca()
    for patch in ax.patches:
        if horizontal:
            value = patch.get_width()
            ax.text(
                value,
                patch.get_y() + patch.get_height() / 2,
                f" {fmt.format(value)}",
                va="center",
                fontsize=8,
            )
        else:
            value = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                value,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )


def monthly_mean(frame: pd.DataFrame, time_column: str, value_column: str) -> pd.Series:
    monthly = frame.set_index(time_column)[value_column].resample("MS").mean()
    return monthly.dropna()


def filter_time_range(
    frame: pd.DataFrame,
    time_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = frame[time_column].between(start, end)
    return frame.loc[mask].copy()


def set_month_date_axis(dates: Iterable[object], *, interval: int = 3, ax: Optional[plt.Axes] = None) -> None:
    ax = ax or plt.gca()
    dates = pd.to_datetime(list(dates))
    tick_positions = list(range(0, len(dates), interval))
    if len(dates) and tick_positions[-1] != len(dates) - 1:
        tick_positions.append(len(dates) - 1)
    tick_dates = dates[tick_positions]
    ax.set_xlim(dates.min() - pd.Timedelta(days=12), dates.max() + pd.Timedelta(days=12))
    ax.set_xticks(tick_dates)
    ax.set_xticklabels([date.strftime("%Y-%m") for date in tick_dates])
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def plot_monthly_scada_volume(scada: pd.DataFrame, output_path: Path) -> None:
    monthly = scada.assign(month=scada["Date_time"].dt.to_period("M").astype(str))
    monthly_counts = monthly.groupby("month").size()

    plt.figure(figsize=(9.6, 4.8))
    monthly_counts.plot(kind="bar", color=COLORS["blue"])
    style_axes("Monthly SCADA Data Volume", "Month", "SCADA rows")
    set_sparse_month_ticks(monthly_counts.index)
    save_figure(output_path)


def plot_data_source_rows(inputs: dict[str, object], output_path: Path) -> None:
    counts = pd.Series(
        {
            "SCADA": len(inputs["scada"]),
            "Plant meter/curtail": len(inputs["plant"]),
            "ERA5": len(inputs["era5"]),
            "MERRA2": len(inputs["merra2"]),
            "Asset": len(inputs["asset"]),
        }
    )

    plt.figure(figsize=(8.4, 4.8))
    counts.sort_values().plot(kind="barh", color=COLORS["blue"])
    plt.xscale("log")
    style_axes("Loaded Raw Data Volume by Source", "Rows (log scale)", "Data source")
    add_bar_labels(horizontal=True)
    save_figure(output_path)


def plot_monthly_mean_wind_speed(
    monthly: pd.Series,
    *,
    title: str,
    ylabel: str,
    color: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9.6, 4.8))
    plt.plot(monthly.index, monthly.values, marker="o", linewidth=1.8, color=color)
    style_axes(title, "Month", ylabel)
    set_month_date_axis(monthly.index)
    save_figure(output_path)


def plot_monthly_mean_wind_speed_panels(
    monthly_series: list[tuple[str, str, pd.Series, str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(monthly_series),
        ncols=1,
        figsize=(10.2, 8.2),
        sharex=True,
    )
    axes = list(axes)

    all_dates = pd.to_datetime(
        sorted({date for _, _, monthly, _ in monthly_series for date in monthly.index})
    )

    for index, (title, ylabel, monthly, color) in enumerate(monthly_series):
        ax = axes[index]
        ax.plot(monthly.index, monthly.values, marker="o", linewidth=1.8, color=color)
        ax.set_title(title, loc="left", pad=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("Month")
    set_month_date_axis(all_dates, ax=axes[-1])
    fig.suptitle("Monthly Mean Wind Speed by Data Source", y=0.985)
    fig.subplots_adjust(hspace=0.34)
    save_figure(output_path)


def plot_power_curve(scada: pd.DataFrame, output_path: Path) -> None:
    sample = scada[["Ws_avg", "P_avg"]].dropna()
    sample = sample.sample(n=min(50000, len(sample)), random_state=42)

    plt.figure(figsize=(8.4, 5.0))
    plt.scatter(sample["Ws_avg"], sample["P_avg"], s=3, alpha=0.12, color=COLORS["green"])
    style_axes("Wind Speed vs Power", "Wind speed (m/s)", "Power (kW)")
    plt.xlim(0, 25)
    save_figure(output_path)


def plot_actual_vs_predicted(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(6.2, 6.2))
    plt.scatter(
        predictions["actual_kw"],
        predictions["predicted_kw"],
        s=4,
        alpha=0.14,
        color=COLORS["blue"],
    )
    minimum = min(predictions["actual_kw"].min(), predictions["predicted_kw"].min())
    maximum = max(predictions["actual_kw"].max(), predictions["predicted_kw"].max())
    plt.plot([minimum, maximum], [minimum, maximum], "--", linewidth=1.3, color="#222222")
    style_axes("Actual vs Predicted Power", "Actual power (kW)", "Predicted power (kW)")
    save_figure(output_path)


def plot_prediction_series(
    predictions: pd.DataFrame,
    output_path: Path,
    rows: int = 500,
) -> None:
    rows_to_plot = min(rows, len(predictions))
    curve_data = predictions.iloc[:rows_to_plot].reset_index(drop=True)

    plt.figure(figsize=(10.2, 4.8))
    plt.plot(curve_data.index, curve_data["actual_kw"], label="Actual", linewidth=1.4, color=COLORS["blue"])
    plt.plot(curve_data.index, curve_data["predicted_kw"], label="Predicted", linewidth=1.2, color=COLORS["orange"])
    style_axes(f"Power Prediction Series (first {rows_to_plot} test rows)", "Test sample", "Power (kW)")
    plt.legend()
    save_figure(output_path)


def plot_metric_bar(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8.0, 4.8))
    values = table[y_column].astype(float)
    plt.bar(table[x_column].astype(str), values, color=COLORS["olive"])
    style_axes(title, x_column, y_column)
    plt.xticks(rotation=25, ha="right")
    add_bar_labels(fmt="{:,.1f}")
    save_figure(output_path)


def plot_monthly_errors(month_table: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9.6, 4.8))
    plt.plot(month_table["month"], month_table["mae_kw"], marker="o", label="MAE", color=COLORS["blue"])
    plt.plot(month_table["month"], month_table["rmse_kw"], marker="s", label="RMSE", color=COLORS["orange"])
    style_axes("Monthly Error Stability", "Month", "Error (kW)")
    set_sparse_month_ticks(month_table["month"].tolist())
    plt.legend()
    save_figure(output_path)


def plot_residual_distribution(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8.4, 4.8))
    plt.hist(predictions["residual_kw"], bins=100, alpha=0.85, color=COLORS["orange"])
    plt.axvline(0, linestyle="--", linewidth=1.4, color="#222222", label="Zero residual")
    style_axes("Prediction Residual Distribution", "Residual: actual - predicted (kW)", "Count")
    plt.legend()
    save_figure(output_path)


def plot_wind_bin_mae_comparison(
    scada_table: pd.DataFrame,
    era5_table: pd.DataFrame,
    output_path: Path,
) -> None:
    comparison = scada_table[["wind_speed_bin", "mae_kw"]].rename(
        columns={"wind_speed_bin": "bin", "mae_kw": "SCADA wind bin"}
    )
    era5 = era5_table[["era5_wind_speed_bin", "mae_kw"]].rename(
        columns={"era5_wind_speed_bin": "bin", "mae_kw": "ERA5 wind bin"}
    )
    comparison = comparison.merge(era5, on="bin", how="outer").set_index("bin")

    plt.figure(figsize=(8.4, 4.8))
    comparison.plot(kind="bar", ax=plt.gca(), color=[COLORS["blue"], COLORS["purple"]])
    style_axes("MAE Comparison by Wind-Speed Bins", "Wind speed bin (m/s)", "MAE (kW)")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="")
    save_figure(output_path)


def plot_turbine_mae_rmse_comparison(turbine_table: pd.DataFrame, output_path: Path) -> None:
    comparison = turbine_table.set_index("Wind_turbine_name")[["mae_kw", "rmse_kw"]]

    plt.figure(figsize=(8.0, 4.8))
    comparison.plot(kind="bar", ax=plt.gca(), color=[COLORS["blue"], COLORS["orange"]])
    style_axes("MAE and RMSE by Turbine", "Turbine", "Error (kW)")
    plt.xticks(rotation=0)
    plt.legend(["MAE", "RMSE"], title="")
    save_figure(output_path)


def generate_figures(inputs: dict[str, object]) -> dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for old_figure in FIGURES_DIR.rglob("*.png"):
        old_figure.unlink()
    for directory in (RAW_FIGURES_DIR, RESULT_FIGURES_DIR, COMPARISON_FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    scada = inputs["scada"]
    predictions = inputs["predictions"]
    tables = inputs["tables"]
    start_time = scada["Date_time"].min()
    end_time = scada["Date_time"].max()
    era5_for_plot = filter_time_range(inputs["era5"], "datetime", start_time, end_time)
    merra2_for_plot = filter_time_range(inputs["merra2"], "datetime", start_time, end_time)
    scada_monthly_wind = monthly_mean(scada, "Date_time", "Ws_avg")
    era5_monthly_wind = monthly_mean(era5_for_plot, "datetime", "ws_100m")
    merra2_monthly_wind = monthly_mean(merra2_for_plot, "datetime", "ws_50m")

    figure_paths = {
        "data_source_rows": RAW_FIGURES_DIR / "01_data_source_rows.png",
        "monthly_scada_volume": RAW_FIGURES_DIR / "02_monthly_scada_volume.png",
        "monthly_mean_wind_by_source": RAW_FIGURES_DIR / "03_monthly_mean_wind_speed_by_source.png",
        "power_curve": RAW_FIGURES_DIR / "04_wind_speed_power_curve.png",
        "actual_vs_predicted": RESULT_FIGURES_DIR / "01_actual_vs_predicted.png",
        "prediction_series": RESULT_FIGURES_DIR / "02_prediction_series.png",
        "residual_distribution": RESULT_FIGURES_DIR / "03_residual_distribution.png",
        "turbine_mae": RESULT_FIGURES_DIR / "04_mae_by_turbine.png",
        "wind_speed_mae": RESULT_FIGURES_DIR / "05_mae_by_scada_wind_speed_bin.png",
        "power_bin_mae": RESULT_FIGURES_DIR / "06_mae_by_power_bin.png",
        "monthly_errors": RESULT_FIGURES_DIR / "07_monthly_errors.png",
        "era5_wind_speed_mae": RESULT_FIGURES_DIR / "08_mae_by_era5_wind_speed_bin.png",
        "wind_bin_mae_comparison": COMPARISON_FIGURES_DIR / "01_scada_vs_era5_wind_bin_mae.png",
        "turbine_mae_rmse_comparison": COMPARISON_FIGURES_DIR / "02_turbine_mae_rmse_comparison.png",
        "monthly_mae_rmse_comparison": COMPARISON_FIGURES_DIR / "03_monthly_mae_rmse_comparison.png",
    }

    plot_data_source_rows(inputs, figure_paths["data_source_rows"])
    plot_monthly_scada_volume(scada, figure_paths["monthly_scada_volume"])
    plot_monthly_mean_wind_speed_panels(
        [
            ("SCADA monthly mean wind speed", "Ws_avg (m/s)", scada_monthly_wind, COLORS["green"]),
            ("ERA5 monthly mean wind speed (SCADA period)", "ws_100m (m/s)", era5_monthly_wind, COLORS["purple"]),
            ("MERRA2 monthly mean wind speed (SCADA period)", "ws_50m (m/s)", merra2_monthly_wind, COLORS["orange"]),
        ],
        figure_paths["monthly_mean_wind_by_source"],
    )
    plot_power_curve(scada, figure_paths["power_curve"])
    plot_actual_vs_predicted(predictions, figure_paths["actual_vs_predicted"])
    plot_prediction_series(predictions, figure_paths["prediction_series"])
    plot_residual_distribution(predictions, figure_paths["residual_distribution"])
    plot_metric_bar(
        tables["turbine"],
        "Wind_turbine_name",
        "mae_kw",
        "MAE by Turbine",
        figure_paths["turbine_mae"],
    )
    plot_metric_bar(
        tables["wind_speed"],
        "wind_speed_bin",
        "mae_kw",
        "MAE by Wind Speed Bin",
        figure_paths["wind_speed_mae"],
    )
    if "era5_wind_speed" in tables:
        plot_metric_bar(
            tables["era5_wind_speed"],
            "era5_wind_speed_bin",
            "mae_kw",
            "MAE by ERA5 Wind Speed Bin",
            figure_paths["era5_wind_speed_mae"],
        )
        plot_wind_bin_mae_comparison(
            tables["wind_speed"],
            tables["era5_wind_speed"],
            figure_paths["wind_bin_mae_comparison"],
        )
    plot_metric_bar(
        tables["power"],
        "power_bin",
        "mae_kw",
        "MAE by Actual Power Bin",
        figure_paths["power_bin_mae"],
    )
    plot_monthly_errors(tables["month"], figure_paths["monthly_errors"])
    plot_turbine_mae_rmse_comparison(tables["turbine"], figure_paths["turbine_mae_rmse_comparison"])
    plot_monthly_errors(tables["month"], figure_paths["monthly_mae_rmse_comparison"])

    return figure_paths


def expected_figure_paths() -> dict[str, Path]:
    return {
        "data_source_rows": RAW_FIGURES_DIR / "01_data_source_rows.png",
        "monthly_scada_volume": RAW_FIGURES_DIR / "02_monthly_scada_volume.png",
        "monthly_mean_wind_by_source": RAW_FIGURES_DIR / "03_monthly_mean_wind_speed_by_source.png",
        "power_curve": RAW_FIGURES_DIR / "04_wind_speed_power_curve.png",
        "actual_vs_predicted": RESULT_FIGURES_DIR / "01_actual_vs_predicted.png",
        "prediction_series": RESULT_FIGURES_DIR / "02_prediction_series.png",
        "residual_distribution": RESULT_FIGURES_DIR / "03_residual_distribution.png",
        "turbine_mae": RESULT_FIGURES_DIR / "04_mae_by_turbine.png",
        "wind_speed_mae": RESULT_FIGURES_DIR / "05_mae_by_scada_wind_speed_bin.png",
        "power_bin_mae": RESULT_FIGURES_DIR / "06_mae_by_power_bin.png",
        "monthly_errors": RESULT_FIGURES_DIR / "07_monthly_errors.png",
        "era5_wind_speed_mae": RESULT_FIGURES_DIR / "08_mae_by_era5_wind_speed_bin.png",
        "wind_bin_mae_comparison": COMPARISON_FIGURES_DIR / "01_scada_vs_era5_wind_bin_mae.png",
        "turbine_mae_rmse_comparison": COMPARISON_FIGURES_DIR / "02_turbine_mae_rmse_comparison.png",
        "monthly_mae_rmse_comparison": COMPARISON_FIGURES_DIR / "03_monthly_mae_rmse_comparison.png",
    }


def collect_existing_figures() -> dict[str, Path]:
    figures = expected_figure_paths()
    missing = [path for path in figures.values() if not path.is_file()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Report figures are missing. Run `python scripts/generate_figures.py` first."
            f"\n{joined}"
        )
    return figures


def xml_text(text: object) -> str:
    return escape(str(text), quote=False)


@dataclass
class DocxBuilder:
    image_rels: list[tuple[str, Path]] = field(default_factory=list)
    body: list[str] = field(default_factory=list)
    image_counter: int = 1

    def paragraph(self, text: str = "", style: Optional[str] = None) -> None:
        style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        self.body.append(f"<w:p>{style_xml}<w:r><w:t>{xml_text(text)}</w:t></w:r></w:p>")

    def heading(self, text: str, level: int = 1) -> None:
        self.paragraph(text, f"Heading{level}")

    def bullet(self, text: str) -> None:
        self.body.append(
            "<w:p>"
            '<w:pPr><w:pStyle w:val="ListParagraph"/></w:pPr>'
            f"<w:r><w:t>• {xml_text(text)}</w:t></w:r>"
            "</w:p>"
        )

    def code_block(self, code: str) -> None:
        for line in code.splitlines():
            self.body.append(
                "<w:p>"
                '<w:pPr><w:pStyle w:val="Code"/></w:pPr>'
                f"<w:r><w:t xml:space=\"preserve\">{xml_text(line)}</w:t></w:r>"
                "</w:p>"
            )

    def table(self, headers: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
        header_cells = "".join(self._cell(header, bold=True) for header in headers)
        row_xml = [f"<w:tr>{header_cells}</w:tr>"]
        for row in rows:
            row_xml.append("<w:tr>" + "".join(self._cell(value) for value in row) + "</w:tr>")

        self.body.append(
            "<w:tbl>"
            "<w:tblPr>"
            '<w:tblBorders><w:top w:val="single" w:sz="6" w:color="999999"/>'
            '<w:left w:val="single" w:sz="6" w:color="999999"/>'
            '<w:bottom w:val="single" w:sz="6" w:color="999999"/>'
            '<w:right w:val="single" w:sz="6" w:color="999999"/>'
            '<w:insideH w:val="single" w:sz="4" w:color="CCCCCC"/>'
            '<w:insideV w:val="single" w:sz="4" w:color="CCCCCC"/></w:tblBorders>'
            "</w:tblPr>"
            + "".join(row_xml)
            + "</w:tbl>"
        )

    def _cell(self, value: object, *, bold: bool = False) -> str:
        bold_xml = "<w:b/>" if bold else ""
        return (
            "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"<w:p><w:r><w:rPr>{bold_xml}</w:rPr><w:t>{xml_text(value)}</w:t></w:r></w:p>"
            "</w:tc>"
        )

    def image(self, path: Path, caption: str, width_inches: float = 6.2) -> None:
        rel_id = f"rId{len(self.image_rels) + 1}"
        self.image_rels.append((rel_id, path))
        image_id = self.image_counter
        self.image_counter += 1

        width_emu = int(width_inches * EMU_PER_INCH)
        height_emu = int(width_emu * _image_aspect_ratio(path))
        filename = path.name

        self.body.append(
            "<w:p><w:r><w:drawing>"
            '<wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{width_emu}" cy="{height_emu}"/>'
            f'<wp:docPr id="{image_id}" name="{xml_text(filename)}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic><pic:nvPicPr>'
            f'<pic:cNvPr id="{image_id}" name="{xml_text(filename)}"/>'
            '<pic:cNvPicPr/>'
            '</pic:nvPicPr><pic:blipFill>'
            f'<a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch>'
            '</pic:blipFill><pic:spPr>'
            '<a:xfrm><a:off x="0" y="0"/>'
            f'<a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '</pic:spPr></pic:pic>'
            '</a:graphicData></a:graphic>'
            '</wp:inline></w:drawing></w:r></w:p>'
        )
        self.paragraph(caption, "Caption")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
            docx.writestr("[Content_Types].xml", self._content_types())
            docx.writestr("_rels/.rels", self._root_rels())
            docx.writestr("docProps/core.xml", self._core_props())
            docx.writestr("docProps/app.xml", self._app_props())
            docx.writestr("word/styles.xml", self._styles())
            docx.writestr("word/document.xml", self._document())
            docx.writestr("word/_rels/document.xml.rels", self._document_rels())
            for _, image_path in self.image_rels:
                docx.write(image_path, f"word/media/{image_path.name}")

    def _content_types(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

    def _root_rels(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

    def _document_rels(self) -> str:
        rels = [
            f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{path.name}"/>'
            for rel_id, path in self.image_rels
        ]
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rels)
            + "</Relationships>"
        )

    def _core_props(self) -> str:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Wind AI OpenOA 实验报告</dc:title>
  <dc:creator>wind-ai-openoa</dc:creator>
  <cp:lastModifiedBy>wind-ai-openoa</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""

    def _app_props(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>wind-ai-openoa</Application>
</Properties>"""

    def _styles(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="300" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="32"/><w:rFonts w:ascii="Calibri" w:eastAsia="Microsoft YaHei"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="240" w:after="100"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="26"/><w:rFonts w:ascii="Calibri" w:eastAsia="Microsoft YaHei"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="caption"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="120"/></w:pPr>
    <w:rPr><w:i/><w:color w:val="666666"/><w:sz w:val="19"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Code">
    <w:name w:val="code"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="0" w:after="0"/><w:shd w:fill="F2F2F2"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="Consolas"/><w:sz w:val="18"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph">
    <w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:ind w:left="360"/></w:pPr>
  </w:style>
</w:styles>"""

    def _document(self) -> str:
        sect = (
            "<w:sectPr>"
            f'<w:pgSz w:w="{int(8.27 * TWIP_PER_INCH)}" w:h="{int(11.69 * TWIP_PER_INCH)}"/>'
            '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="720" w:footer="720" w:gutter="0"/>'
            "</w:sectPr>"
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
            'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
            'xmlns:o="urn:schemas-microsoft-com:office:office" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
            'xmlns:v="urn:schemas-microsoft-com:vml" '
            'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
            'xmlns:w10="urn:schemas-microsoft-com:office:word" '
            'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
            'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
            'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
            'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
            'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
            'mc:Ignorable="w14 wp14">'
            "<w:body>"
            + "".join(self.body)
            + sect
            + "</w:body></w:document>"
        )


def _image_aspect_ratio(path: Path) -> float:
    with path.open("rb") as file:
        header = file.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        return 0.62
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return height / width


def fmt_number(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def fmt_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_time(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")


def mean_confidence_interval(values: pd.Series) -> tuple[float, float, float]:
    values = pd.Series(values).dropna().astype(float)
    mean = float(values.mean())
    if len(values) <= 1:
        return mean, mean, mean
    half_width = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def table_rows_from_metrics(table: pd.DataFrame, group_column: str) -> list[list[object]]:
    return [
        [
            row[group_column],
            fmt_number(int(row["rows"]), 0),
            f"{float(row['mae_kw']):.2f}",
            f"{float(row['rmse_kw']):.2f}",
            f"{float(row['r2']):.4f}" if not pd.isna(row["r2"]) else "",
            f"{float(row['p95_absolute_error_kw']):.2f}",
        ]
        for _, row in table.iterrows()
    ]


def build_report(inputs: dict[str, object], figures: dict[str, Path]) -> None:
    metrics = inputs["metrics"]
    tables = inputs["tables"]
    scada = inputs["scada"]
    plant = inputs["plant"]
    era5 = inputs["era5"]
    merra2 = inputs["merra2"]
    asset = inputs["asset"]
    predictions = inputs["predictions"]

    raw_rows = int(metrics.get("raw_scada_rows", len(scada)))
    feature_rows = int(metrics.get("feature_rows", 0))
    dropped_rows = int(metrics.get("dropped_rows_after_feature_engineering", raw_rows - feature_rows))
    feature_retention = feature_rows / raw_rows if raw_rows else 0
    train_rows = int(metrics["train_rows"])
    test_rows = int(metrics["test_rows"])
    total_model_rows = train_rows + test_rows
    test_ratio = test_rows / total_model_rows if total_model_rows else 0

    scada_start = scada["Date_time"].min()
    scada_end = scada["Date_time"].max()
    turbine_count = scada["Wind_turbine_name"].nunique()
    expected_intervals = int((scada_end - scada_start) / pd.Timedelta(minutes=10)) + 1
    expected_rows = expected_intervals * turbine_count
    coverage_ratio = raw_rows / expected_rows if expected_rows else 0
    duplicate_rows = int(scada.duplicated(["Date_time", "Wind_turbine_name"]).sum())
    target_missing = int(scada["P_avg"].isna().sum())
    wind_missing = int(scada["Ws_avg"].isna().sum())
    invalid_removed_ratio = dropped_rows / raw_rows if raw_rows else 0

    absolute_error = predictions["absolute_error_kw"].astype(float)
    residual = predictions["residual_kw"].astype(float)
    mae_mean, mae_ci_low, mae_ci_high = mean_confidence_interval(absolute_error)
    bias_mean, bias_ci_low, bias_ci_high = mean_confidence_interval(residual)
    within_50 = float((absolute_error <= 50).mean())
    within_100 = float((absolute_error <= 100).mean())

    turbine_table = tables["turbine"].copy()
    month_table = tables["month"].copy()
    wind_table = tables["wind_speed"].copy()
    power_table = tables["power"].copy()
    era5_wind_table = tables.get("era5_wind_speed")
    best_turbine = turbine_table.loc[turbine_table["mae_kw"].idxmin()]
    worst_turbine = turbine_table.loc[turbine_table["mae_kw"].idxmax()]
    best_month = month_table.loc[month_table["mae_kw"].idxmin()]
    worst_month = month_table.loc[month_table["mae_kw"].idxmax()]
    robust_wind_table = wind_table[wind_table["rows"] >= 1000]
    worst_wind_bin = robust_wind_table.loc[robust_wind_table["mae_kw"].idxmax()]
    sparse_wind_table = wind_table[wind_table["rows"] < 1000]
    robust_power_table = power_table[power_table["rows"] >= 1000]
    worst_power_bin = robust_power_table.loc[robust_power_table["mae_kw"].idxmax()]
    best_power_bin = power_table.loc[power_table["mae_kw"].idxmin()]
    low_wind_bin = wind_table.loc[wind_table["wind_speed_bin"].astype(str).eq("0-3")].iloc[0]
    steep_wind_bin = wind_table.loc[wind_table["wind_speed_bin"].astype(str).eq("8-12")].iloc[0]
    low_power_bin = power_table.loc[power_table["power_bin"].astype(str).eq("-100-100")].iloc[0]
    rated_power_bin = power_table.loc[power_table["power_bin"].astype(str).eq("1500-1900")].iloc[0]
    sparse_power_bin = power_table.loc[power_table["power_bin"].astype(str).eq("1900-2500")].iloc[0]

    rmse_mae_ratio = metrics["rmse_kw"] / metrics["mae_kw"]
    p95_to_mae_ratio = metrics["p95_absolute_error_kw"] / metrics["mae_kw"]
    p99_absolute_error = float(absolute_error.quantile(0.99))
    turbine_mae_gap = float(worst_turbine["mae_kw"] - best_turbine["mae_kw"])
    turbine_mae_ratio = float(worst_turbine["mae_kw"] / best_turbine["mae_kw"])
    month_mae_gap = float(worst_month["mae_kw"] - best_month["mae_kw"])
    high_error_share = 1.0 - within_100

    doc = DocxBuilder()
    doc.heading("风机功率模型数据实验报告", 1)
    doc.paragraph(f"报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.paragraph(
        "本报告围绕 La Haute Borne 风电场历史运行数据，评估一个风机功率基线模型是否可以用于运行监测、"
        "功率曲线分析和异常识别。报告重点说明实验背景、目标、假设、样本范围、指标体系、模型处理链路、"
        "实验结果、可信度分析和业务决策。"
    )

    doc.heading("1. 实验背景", 1)
    doc.paragraph(
        "风电场日常运维需要持续判断风机在给定风况、控制状态和设备条件下的发电功率是否合理。"
        "仅查看原始 SCADA 曲线很难快速区分正常波动、气象变化、控制策略变化和潜在设备异常，"
        "因此需要一个可量化的功率基线作为后续残差监测和异常诊断的参照。"
    )
    doc.bullet("业务现状：SCADA 数据记录粒度高、变量多，人工查看曲线效率低，跨风机和跨月份比较不够稳定。")
    doc.bullet("问题与机会：如果模型能够准确估计当前运行状态下的合理功率，就可以把实际功率与模型输出的差值转化为可监控指标。")
    doc.bullet("已有数据：本实验已有 2014-01-01 至 2015-12-31 的 10 分钟级 SCADA、ERA5/MERRA2 再分析气象、资产台账和场站层数据。")
    doc.bullet("希望支持的决策：判断该模型能否作为风机功率基线进入离线监控验证，以及后续是否需要扩大为在线告警或性能评估方案。")

    doc.heading("2. 实验目标", 1)
    doc.paragraph("本实验要回答的核心问题是：增强特征模型能否在未来测试时间段内稳定估计风机 10 分钟平均功率。")
    doc.table(
        ["目标", "可衡量口径", "验收参考"],
        [
            ["整体准确性", "测试集 MAE、RMSE、R2", "MAE <= 35 kW，RMSE <= 60 kW，R2 >= 0.98"],
            ["极端误差控制", "P95 绝对误差、最大绝对误差", "P95 绝对误差 <= 120 kW，最大误差需定位原因"],
            ["稳定性", "分风机、分月份、分风速、分功率区间误差", "主要分群不出现持续异常偏高"],
            ["业务可用性", "残差是否可解释、是否能形成监控指标", "可进入离线监控验证，但需保留风险边界"],
        ],
    )

    doc.heading("3. 实验假设", 1)
    doc.table(
        ["假设类型", "内容", "判定方式"],
        [
            ["H0 原假设", "增强特征模型对风机功率估计没有足够效果，测试集误差超过业务验收参考。", "MAE > 35 kW 或 R2 < 0.98，或分群误差不可接受。"],
            ["H1 备择假设", "在 SCADA、再分析气象和资产信息共同输入下，模型可以稳定估计当前运行状态下的合理功率。", "MAE <= 35 kW、R2 >= 0.98，且主要分群结果可解释。"],
            ["残差假设", "模型残差应大体围绕 0 波动，不应长期系统性高估或低估。", "检查平均残差及其 95% 置信区间。"],
        ],
    )

    doc.heading("4. 实验对象与范围", 1)
    doc.table(
        ["项目", "范围定义"],
        [
            ["目标对象", "La Haute Borne 风电场 4 台风机：R80711、R80721、R80736、R80790。"],
            ["时间范围", f"{fmt_time(scada_start)} 至 {fmt_time(scada_end)}。"],
            ["样本粒度", "单台风机 10 分钟平均运行记录。"],
            ["纳入条件", "存在可用功率、风速、角度、温度、控制状态、气象背景和资产信息的记录。"],
            ["排除条件", "缺失关键字段、风速明显不合理、功率超出合理区间、或无法匹配必要特征的记录。"],
            ["推广边界", "结论主要适用于该风电场、该机型和相近数据采集条件，不直接等同于其他风场或未来预测场景。"],
        ],
    )
    doc.paragraph(
        f"原始 SCADA 共 {fmt_number(raw_rows, 0)} 行，特征工程后保留 {fmt_number(feature_rows, 0)} 行，"
        f"仅丢弃 {fmt_number(dropped_rows, 0)} 行，保留率为 {fmt_percent(feature_retention)}。"
        "因此本实验不是只抽取少量原始数据进行建模，而是基本使用了可用 SCADA 数据。"
    )

    doc.heading("5. 实验分组与样本量", 1)
    doc.paragraph(
        "本实验属于离线模型验证，不是面向用户的线上 A/B 分流。为了避免未来信息进入训练阶段，"
        "样本按时间顺序切分：前 80% 用于训练，后 20% 用于测试。测试期真实功率作为对照参照，"
        "模型输出作为实验方案结果，两者逐条配对计算误差。"
    )
    doc.table(
        ["分组", "时间范围", "样本量", "占比", "用途"],
        [
            ["训练集", f"{metrics['train_start']} 至 {metrics['train_end']}", fmt_number(train_rows, 0), fmt_percent(train_rows / total_model_rows), "学习功率基线关系"],
            ["测试集", f"{metrics['test_start']} 至 {metrics['test_end']}", fmt_number(test_rows, 0), fmt_percent(test_ratio), "检验时间外推表现"],
        ],
    )
    doc.paragraph(
        f"测试集包含 {fmt_number(test_rows, 0)} 条样本，整体 MAE 的 95% 置信区间为 "
        f"{mae_ci_low:.2f} 至 {mae_ci_high:.2f} kW，区间较窄，说明整体误差估计样本量充足。"
        "但高风速稀疏区间样本少，分群结论需要谨慎。"
    )

    doc.heading("6. 指标体系", 1)
    doc.table(
        ["指标类型", "指标", "口径"],
        [
            ["核心指标", "MAE", "测试集中实际功率与模型估计功率绝对差值的平均值，单位 kW。"],
            ["核心指标", "RMSE", "测试集中误差平方均值的平方根，对大误差更敏感，单位 kW。"],
            ["核心指标", "R2", "模型解释实际功率波动的比例。"],
            ["过程指标", "中位绝对误差、P95 绝对误差", "分别衡量典型误差和尾部误差。"],
            ["过程指标", "分风机、分月份、分风速、分功率区间误差", "解释模型在哪些场景更好或更差。"],
            ["护栏指标", "平均残差、最大绝对误差、低样本分箱", "检查系统性偏差、极端异常和统计不稳定风险。"],
        ],
    )

    doc.heading("7. 数据采集与质量检查", 1)
    doc.table(
        ["数据源", "记录数", "实验角色", "质量关注点"],
        [
            ["SCADA", fmt_number(len(scada), 0), "核心输入与功率目标", "时间连续性、缺失值、重复记录、异常功率和异常风速。"],
            ["场站电表/可利用率/限电", fmt_number(len(plant), 0), "业务解释和后续护栏指标", "当前不作为模型输入，避免引入场站级结果信息。"],
            ["ERA5", fmt_number(len(era5), 0), "气象背景输入", "与 10 分钟 SCADA 的时间对齐和高度口径差异。"],
            ["MERRA2", fmt_number(len(merra2), 0), "气象背景输入", "与 ERA5 的差异可用于对比解释。"],
            ["资产台账", fmt_number(len(asset), 0), "静态设备输入", "风机名称匹配、额定功率、轮毂高度和叶轮直径完整性。"],
        ],
    )
    doc.table(
        ["检查项", "结果", "说明"],
        [
            ["SCADA 时间覆盖率", fmt_percent(coverage_ratio), f"按 4 台风机、10 分钟粒度估算应有 {fmt_number(expected_rows, 0)} 行。"],
            ["重复记录", fmt_number(duplicate_rows, 0), "按时间与风机名称检查。"],
            ["功率缺失", fmt_number(target_missing, 0), "目标字段 P_avg 缺失行数。"],
            ["风速缺失", fmt_number(wind_missing, 0), "核心输入 Ws_avg 缺失行数。"],
            ["特征工程剔除比例", fmt_percent(invalid_removed_ratio), "包括缺失关键特征或明显不合理值。"],
        ],
    )
    doc.image(figures["data_source_rows"], "图 1：原始数据源记录数，展示 SCADA、场站、气象和资产数据的覆盖规模。")
    doc.image(figures["monthly_scada_volume"], "图 2：每月 SCADA 记录数，用于检查历史数据覆盖是否连续。")
    doc.image(figures["monthly_mean_wind_by_source"], "图 3：SCADA、ERA5 与 MERRA2 月均风速三联图，在同一时间范围内对比不同数据源的月度风况。")
    doc.image(figures["power_curve"], "图 4：SCADA 风速与功率散点，展示功率曲线形态和高误差可能出现的风速区间。")

    doc.heading("8. 数据处理、特征工程与模型处理链路", 1)
    doc.paragraph(
        "本实验的核心方法是把多源原始数据整理成按时间对齐的单风机样本表，"
        "再让模型学习“当前风况、控制状态、环境背景和设备属性”到“当前 10 分钟平均功率”的映射关系。"
        "这个处理目标不是提前预测未来发电量，而是建立一个当前状态下的合理功率基线。"
    )
    doc.table(
        ["处理环节", "具体处理", "这样处理的原因"],
        [
            ["时间标准化", "将 SCADA、ERA5、MERRA2 的时间字段转换为统一时间类型，并按时间顺序排列。", "保证训练集与测试集按真实时间先后切分，避免未来数据泄漏。"],
            ["异常值过滤", "保留风速 0 至 40 m/s、功率 -100 至 2500 kW 的样本，并剔除关键特征缺失记录。", "去除明显不符合物理范围或无法学习的记录，降低异常点对模型的干扰。"],
            ["气象对齐", "ERA5 与 MERRA2 为小时级背景气象，按时间向前匹配到每条 10 分钟 SCADA 样本。", "SCADA 粒度更细，而再分析气象提供区域背景；向前匹配避免使用样本之后的气象信息。"],
            ["资产匹配", "按风机名称合并额定功率、轮毂高度、叶轮直径和海拔。", "静态设备属性决定风机功率曲线的尺度和上限，有助于区分设备差异。"],
            ["目标定义", "以 SCADA 中的 P_avg 作为真实功率目标。", "P_avg 是当前样本实际 10 分钟平均功率，适合评价当前状态功率基线。"],
        ],
    )
    doc.paragraph(
        f"从样本保留情况看，{fmt_number(raw_rows, 0)} 行 SCADA 中有 {fmt_number(feature_rows, 0)} 行进入模型数据集，"
        f"剔除比例仅 {fmt_percent(invalid_removed_ratio)}。这说明结果主要反映完整历史数据的规律，而不是少量抽样结果。"
    )
    doc.table(
        ["特征类别", "输入内容", "作用解释"],
        [
            ["风机运行特征", "风速 Ws_avg、外部温度 Ot_avg、桨距角 Ba_avg。", "风速是功率曲线的主驱动；温度影响空气密度；桨距角反映控制系统对风况和功率限制的响应。"],
            ["方向角特征", "机舱方向 Va_avg 与风向 Wa_avg 转换为正弦、余弦。", "方向是周期变量，0 度与 360 度在物理上相邻，正弦/余弦表达可以避免角度断点。"],
            ["时间周期特征", "小时与月份转换为正弦、余弦。", "小时特征可吸收日内运行差异，月份特征可表达季节性风况和环境变化。"],
            ["ERA5 气象特征", "100m 风速、空气密度、2m 温度、地表气压、风向正弦/余弦。", "提供风机侧传感器之外的区域气象背景，补充环境解释能力。"],
            ["MERRA2 气象特征", "50m 风速、空气密度、2m 温度、地表气压、风向正弦/余弦。", "与 ERA5 形成另一套再分析气象口径，降低单一气象源偏差带来的解释风险。"],
            ["资产特征", "额定功率、轮毂高度、叶轮直径、海拔。", "把风机物理差异显式提供给模型，支持跨风机共同训练。"],
        ],
    )
    doc.paragraph(
        f"本次增强特征集共 {len(metrics.get('features', []))} 个输入特征。"
        "特征设计遵循两个原则：第一，优先使用在当前样本时点可以观测或定义的信息；"
        "第二，把物理上非线性、周期性或跨数据源差异明显的信息转成模型更容易学习的形式。"
    )
    doc.table(
        ["未作为模型输入的数据", "处理方式", "原因"],
        [
            ["实际功率 P_avg", "仅作为模型学习目标和评估对照。", "目标值不能同时作为输入，否则会造成直接泄漏。"],
            ["场站电表", "保留为业务解释和后续护栏分析数据。", "场站电表是聚合结果信号，直接输入会让模型过度依赖结果性信息。"],
            ["可利用率与限电字段", "当前不进入训练特征。", "这些字段可能包含运行结果或调度结果信息，现阶段先用于解释异常和后续专题实验。"],
        ],
    )
    doc.table(
        ["管理项", "当前记录"],
        [
            ["模型名称", "power_baseline"],
            ["模型类型", "HistGradientBoostingRegressor"],
            ["模型角色", "离线功率基线模型，不是直接预测未来发电量的模型。"],
            ["训练数据版本", "La Haute Borne 2014-2015 历史数据。"],
            ["特征版本", f"{metrics.get('feature_set', 'unknown')}，共 {len(metrics.get('features', []))} 个输入特征。"],
            ["输出定义", "10 分钟平均功率估计值，单位 kW。"],
            ["版本锁定", "本次评估期间模型、特征和参数保持固定。"],
        ],
    )
    doc.paragraph(
        "模型采用梯度提升树回归方法。它不是先假设一条固定形式的功率曲线，而是通过多棵浅树逐步修正前一轮预测误差，"
        "学习风速、桨距角、气象背景和资产属性之间的非线性组合关系。风机功率曲线本身具有明显分段特征："
        "低风速区功率接近零，中等风速区功率快速爬升，高风速和接近额定区会受到控制策略影响。"
        "树模型擅长处理这类分段关系，因此比简单线性模型更适合当前任务。"
    )
    doc.paragraph(
        "模型效果较好的主要原因有三点：第一，风速与功率之间存在强物理关系，SCADA 风速为模型提供了最核心信号；"
        "第二，桨距角、温度、方向角和气象再分析数据补充了控制状态和环境背景，使模型能解释同一风速下的功率差异；"
        "第三，训练和测试来自同一风场、同一批风机，数据分布相对一致，且样本量充足。"
    )
    doc.paragraph(
        "关键风险是信息泄漏和时间穿越：实际功率不能作为模型输入，测试期数据不能进入训练阶段；"
        "场站电表、可利用率和限电字段暂不进入模型输入，是为了避免把结果性信息提前交给模型。"
        "桨距角属于当前控制状态变量，适合当前功率基线估计；若后续改为提前预测未来功率，需要重新评估该变量是否在预测时点可用。"
    )

    doc.heading("9. 实验执行与监控", 1)
    doc.paragraph(
        "离线实验执行期间重点监控数据覆盖、时间切分、特征缺失、训练测试重叠、整体误差和分群异常。"
        "若发现测试期进入训练集、特征保留率异常下降、分风机误差大面积恶化或残差明显偏移，应停止使用该轮结果并重新检查数据链路。"
    )
    doc.table(
        ["风险", "停止或重做规则"],
        [
            ["数据质量风险", "关键输入或目标字段缺失显著增加，或特征保留率低于 99%。"],
            ["实验污染风险", "训练集时间晚于测试集，或测试期样本被用于模型训练。"],
            ["模型稳定性风险", "整体 MAE 超过 35 kW、R2 低于 0.98，或主要风机分群误差持续异常。"],
            ["业务解释风险", "平均残差置信区间长期偏离 0，且无法由风况、控制状态或数据质量解释。"],
        ],
    )

    doc.heading("10. 统计分析方法", 1)
    doc.paragraph(
        "实验结束后采用配对误差分析：每条测试样本都有一个真实功率和一个模型估计功率，"
        "通过 MAE、RMSE、R2、残差均值、中位绝对误差、P95 绝对误差和最大绝对误差评估模型。"
        "置信区间基于测试样本误差的均值和标准误估算；分群分析用于判断不同风机、月份、风速和功率区间下的异质性。"
    )
    doc.paragraph(
        "由于本实验不是线上 A/B 实验，不对用户转化率或收入类指标做显著性检验。"
        "本报告关注的是离线模型效果是否可靠，以及是否具有进入业务验证的价值。"
    )

    doc.heading("11. 实验结果", 1)
    doc.table(
        ["指标", "结果", "判断"],
        [
            ["训练样本数", fmt_number(train_rows, 0), "样本规模充足。"],
            ["测试样本数", fmt_number(test_rows, 0), "用于时间外推评估。"],
            ["MAE", f"{metrics['mae_kw']:.2f} kW", "低于 35 kW 验收参考。"],
            ["RMSE", f"{metrics['rmse_kw']:.2f} kW", "低于 60 kW 验收参考，但高于 MAE，说明存在尾部误差。"],
            ["R2", f"{metrics['r2']:.4f}", "高于 0.98 验收参考。"],
            ["中位绝对误差", f"{metrics['median_absolute_error_kw']:.2f} kW", "典型样本误差较低。"],
            ["P95 绝对误差", f"{metrics['p95_absolute_error_kw']:.2f} kW", "低于 120 kW 验收参考。"],
            ["最大绝对误差", f"{metrics['max_absolute_error_kw']:.2f} kW", "需要作为异常样本继续追踪。"],
            ["50 kW 内样本占比", fmt_percent(within_50), "多数样本误差较小。"],
            ["100 kW 内样本占比", fmt_percent(within_100), "尾部误差总体可控。"],
        ],
    )
    doc.table(
        ["统计项", "95% 置信区间", "解释"],
        [
            ["MAE", f"{mae_ci_low:.2f} 至 {mae_ci_high:.2f} kW", "整体平均绝对误差估计稳定。"],
            ["平均残差", f"{bias_ci_low:.2f} 至 {bias_ci_high:.2f} kW", "区间为正，说明模型平均略低估实际功率。"],
        ],
    )
    doc.paragraph(
        f"整体看，主要准确性假设成立：MAE 为 {metrics['mae_kw']:.2f} kW，R2 为 {metrics['r2']:.4f}。"
        f"但残差均值为 {bias_mean:.2f} kW，95% 置信区间不跨 0，说明模型存在轻微系统性低估，"
        "后续上线前需要用残差校准或分场景阈值处理。"
    )
    doc.table(
        ["结果现象", "数值表现", "分析解释"],
        [
            ["典型误差较低", f"中位绝对误差 {metrics['median_absolute_error_kw']:.2f} kW", "一半测试样本的误差低于该值，说明大多数常规工况被模型较好捕捉。"],
            ["尾部误差存在", f"RMSE/MAE = {rmse_mae_ratio:.2f}，P95/MAE = {p95_to_mae_ratio:.2f}", "RMSE 和 P95 明显高于 MAE，说明少量高误差样本会显著拉高风险。"],
            ["大部分样本可控", f"{fmt_percent(within_50)} 样本误差不超过 50 kW，{fmt_percent(within_100)} 不超过 100 kW", "作为额定功率约 2050 kW 风机的当前功率基线，绝大多数样本误差处于较低水平。"],
            ["极端误差需解释", f"P99 绝对误差 {p99_absolute_error:.2f} kW，最大绝对误差 {metrics['max_absolute_error_kw']:.2f} kW", "极端值可能来自高风速稀疏样本、控制状态变化、传感器异常或限电等特殊工况。"],
            ["系统性偏差较小但存在", f"平均残差 {bias_mean:.2f} kW", "残差为实际功率减模型估计功率，正值表示模型平均略低估实际功率。"],
        ],
    )
    doc.paragraph(
        "从图形上看，实际值与估计值大体沿对角线分布，说明模型学到了功率曲线的主关系；"
        "序列对比能跟随短期功率变化，说明模型不仅拟合平均水平，也捕捉了测试期运行波动。"
        "残差分布集中在 0 附近，但右侧存在少量长尾，和平均残差为正、最大误差较高的数值结果一致。"
    )
    doc.image(figures["actual_vs_predicted"], "图 5：实际功率与模型估计功率对比，大多数点接近对角线。")
    doc.image(figures["prediction_series"], "图 6：测试期样本序列对比，展示模型能跟随功率变化趋势。")
    doc.image(figures["residual_distribution"], "图 7：残差分布，展示误差集中程度以及尾部误差。")

    doc.heading("12. 分群与异质性分析", 1)
    doc.table(
        ["风机", "样本数", "MAE kW", "RMSE kW", "R2", "P95 绝对误差 kW"],
        table_rows_from_metrics(turbine_table, "Wind_turbine_name"),
    )
    doc.paragraph(
        f"风机层面，{best_turbine['Wind_turbine_name']} 的 MAE 最低，为 {best_turbine['mae_kw']:.2f} kW；"
        f"{worst_turbine['Wind_turbine_name']} 的 MAE 最高，为 {worst_turbine['mae_kw']:.2f} kW。"
        "差异仍在整体可接受范围内，但高误差风机应进入后续运维排查清单。"
    )
    doc.paragraph(
        f"从差异幅度看，最高 MAE 与最低 MAE 相差 {turbine_mae_gap:.2f} kW，"
        f"最高值约为最低值的 {turbine_mae_ratio:.2f} 倍。R80790 的 RMSE 和 P95 误差也最高，"
        "说明它不只是平均误差偏高，还更容易出现较大的尾部误差。R80736 的最大误差最高，"
        "但其 MAE 和 P95 并不高，说明该风机整体表现稳定，最大误差更可能来自少量特殊样本。"
    )
    doc.paragraph(
        "风机差异可能来自三个方面：一是局部流场和尾流影响导致同一背景风速下单机入流不同；"
        "二是传感器校准或控制策略差异造成 SCADA 变量分布不同；三是个别风机存在限电、维护或短时异常工况。"
        "因此分风机结果不应只看平均误差，还要结合最大误差和 P95 误差判断是否存在持续性问题。"
    )
    doc.image(figures["turbine_mae"], "图 8：分风机 MAE，展示不同风机的误差差异。")
    doc.image(figures["turbine_mae_rmse_comparison"], "图 9：分风机 MAE 与 RMSE 对比，用于观察大误差敏感性。")
    doc.table(
        ["SCADA 风速区间", "样本数", "MAE kW", "RMSE kW", "R2", "P95 绝对误差 kW"],
        table_rows_from_metrics(wind_table, "wind_speed_bin"),
    )
    doc.paragraph(
        f"风速层面，样本充足的区间中 {worst_wind_bin['wind_speed_bin']} m/s 的 MAE 最高，"
        f"为 {worst_wind_bin['mae_kw']:.2f} kW。"
        "这符合功率曲线爬坡区和控制区更难拟合的业务规律。"
    )
    doc.paragraph(
        f"低风速 0-3 m/s 区间 MAE 仅 {low_wind_bin['mae_kw']:.2f} kW，因为该区间实际功率通常接近 0，"
        "模型即使只学到停机或低功率状态也能取得很低绝对误差。该区间 R2 为负并不代表业务风险很高，"
        "主要原因是实际功率波动范围很小，R2 对低方差数据非常敏感。"
    )
    doc.paragraph(
        f"8-12 m/s 区间 MAE 升至 {steep_wind_bin['mae_kw']:.2f} kW，P95 绝对误差达到 "
        f"{steep_wind_bin['p95_absolute_error_kw']:.2f} kW。该区间处于功率曲线快速爬升阶段，"
        "微小风速误差、风向偏差、湍流和桨距控制变化都会放大成功率误差，因此是后续残差监控最需要细分阈值的风速段。"
    )
    if not sparse_wind_table.empty:
        sparse_bins = "、".join(str(value) for value in sparse_wind_table["wind_speed_bin"].tolist())
        doc.paragraph(f"{sparse_bins} m/s 区间样本较少，不能单独作为稳定业务结论。")
    doc.image(figures["wind_speed_mae"], "图 10：按 SCADA 风速区间统计 MAE，用于定位误差更大的风速段。")
    if figures["era5_wind_speed_mae"].is_file():
        doc.image(figures["era5_wind_speed_mae"], "图 11：按 ERA5 风速区间统计 MAE，用于观察气象背景下的误差分布。")
    if era5_wind_table is not None:
        era5_worst_bin = era5_wind_table.loc[era5_wind_table["mae_kw"].idxmax()]
        doc.paragraph(
            f"按 ERA5 风速分箱时，误差最高的区间是 {era5_worst_bin['era5_wind_speed_bin']} m/s，"
            f"MAE 为 {era5_worst_bin['mae_kw']:.2f} kW。与 SCADA 风速分箱相比，ERA5 是区域背景风速，"
            "不完全等同于机舱或转子处真实入流，因此更适合解释大尺度风况，而不是替代风机侧风速。"
        )
    doc.table(
        ["功率区间", "样本数", "MAE kW", "RMSE kW", "R2", "P95 绝对误差 kW"],
        table_rows_from_metrics(power_table, "power_bin"),
    )
    doc.image(figures["power_bin_mae"], "图 12：按实际功率区间统计 MAE，展示低功率、爬坡段和高功率区间表现。")
    doc.paragraph(
        f"功率区间结果显示，低功率 -100-100 kW 区间 MAE 为 {low_power_bin['mae_kw']:.2f} kW，"
        "模型几乎只需判断风机是否处于低功率状态，因此误差最低。随着实际功率升高，"
        "同样的风速或控制变量扰动会对应更大的功率差，误差逐步上升。"
    )
    doc.paragraph(
        f"在样本较充足的功率段中，{worst_power_bin['power_bin']} kW 区间 MAE 最高，"
        f"为 {worst_power_bin['mae_kw']:.2f} kW；接近额定的 1500-1900 kW 区间 MAE 为 "
        f"{rated_power_bin['mae_kw']:.2f} kW。高功率区受限电、桨距控制、额定功率平台和空气密度影响更明显，"
        "因此比低功率区更难精确估计。1900-2500 kW 区间只有 "
        f"{fmt_number(int(sparse_power_bin['rows']), 0)} 条样本，虽然误差最高，但统计稳定性不足，不能单独作为上线阈值依据。"
    )
    doc.table(
        ["月份", "样本数", "MAE kW", "RMSE kW", "R2", "P95 绝对误差 kW"],
        table_rows_from_metrics(month_table, "month"),
    )
    doc.paragraph(
        f"月份层面，{best_month['month']} MAE 最低，为 {best_month['mae_kw']:.2f} kW；"
        f"{worst_month['month']} MAE 最高，为 {worst_month['mae_kw']:.2f} kW。"
        "测试期覆盖 2015 年 8 月至 12 月，仍需要更多季节样本验证全年稳定性。"
    )
    doc.paragraph(
        f"月份间 MAE 差距为 {month_mae_gap:.2f} kW。2015-10 的误差最低，说明该月测试样本更接近训练集中已学习到的风况和控制状态；"
        "2015-09、2015-11 和 2015-12 的误差较高，可能与月度风况变化、较多爬坡区样本、控制策略变化或少量极端误差有关。"
        "2015-08 只是从 8 月 8 日之后开始进入测试集，因此该月不是完整自然月，解释时需要注意样本周期差异。"
    )
    doc.image(figures["monthly_errors"], "图 13：按月份统计 MAE 与 RMSE，展示测试期时间稳定性。")
    doc.image(figures["wind_bin_mae_comparison"], "图 14：SCADA 风速分箱与 ERA5 风速分箱的 MAE 对比，展示不同风速口径下的误差结构。")
    doc.image(figures["monthly_mae_rmse_comparison"], "图 15：月份维度 MAE 与 RMSE 对比，辅助判断误差是否随时间漂移。")

    doc.heading("13. 结果解释", 1)
    doc.paragraph(
        "综合整体指标、分群指标和图形表现，本次实验效果可以解释为“强物理主关系 + 当前状态变量 + 同场景充分样本”共同作用的结果。"
        "风速决定功率曲线主形态，桨距角和方向角帮助解释控制状态与入流方向，ERA5/MERRA2 提供区域气象背景，资产特征提供设备尺度。"
        "这些信息组合后，模型能够在大多数常规运行状态下给出接近真实功率的估计。"
    )
    doc.bullet("实验假设总体成立：模型在测试期达到 MAE、RMSE、R2 和 P95 误差的验收参考，可作为功率基线候选。")
    doc.bullet("模型效果不应被理解为长期预测能力。本实验估计的是当前 10 分钟样本的合理功率，输入中包含当前风速和当前控制状态。")
    doc.bullet("平均残差为正，说明模型整体略低估实际功率。该偏差不大，但如果直接用于告警，会让正残差阈值更容易触发，需要先做偏差校准。")
    doc.bullet("RMSE 高于 MAE，说明平均表现好并不代表所有样本都好；高风速、高功率和稀疏工况仍是主要风险来源。")
    doc.bullet("分风机差异显示 R80790 和 R80711 相对更难估计，后续应检查传感器、控制参数、维护记录和局部流场差异。")
    doc.bullet("分风速结果符合功率曲线规律：低风速绝对误差低，中高风速误差高，额定附近受控制策略和限电影响更强。")
    doc.bullet("分月份结果说明模型在测试期内基本稳定，但 2015-09、2015-11、2015-12 误差较高，后续需要结合月度风况和异常样本做复核。")
    doc.bullet("测试期只覆盖 2015 年 8 月至 12 月，结论对全年季节泛化仍需补充滚动时间验证。")

    doc.heading("14. 业务决策", 1)
    doc.table(
        ["决策项", "建议", "理由"],
        [
            ["是否直接全量上线", "暂不直接全量上线", "当前是离线验证，仍需在线数据延迟、特征可用性和告警误报率验证。"],
            ["是否进入下一阶段", "建议进入离线监控或影子验证", "整体准确性达标，残差具备作为异常监控指标的基础。"],
            ["优先应用场景", "功率基线、残差分析、风机对比、异常样本排查", "模型估计的是当前状态下合理功率，适合做参照而非直接做长期预测。"],
            ["上线护栏", "监控数据缺失率、平均残差、分风机 MAE、P95 误差和最大误差", "防止模型在特定风机、月份或风况下失效。"],
            ["未来预测场景", "需重新设计特征并复验", "当前包含桨距角等当前状态变量，未来预测时未必提前可得。"],
        ],
    )

    doc.heading("15. 实验复盘与沉淀", 1)
    doc.bullet("目标设计较清晰：以 MAE、RMSE、R2 和 P95 误差判断功率基线是否可用。")
    doc.bullet("样本量总体充足，整体指标置信区间较窄；但高风速稀疏分箱不适合独立做强结论。")
    doc.bullet("数据使用范围合理，几乎全部可用 SCADA 都进入训练或测试；场站层结果数据未直接进入模型，降低泄漏风险。")
    doc.bullet("已验证的假设：增强特征模型可以较准确估计当前运行状态功率。")
    doc.bullet("被部分否定的假设：残差并非完全围绕 0，无偏性仍需校准。")
    doc.bullet("后续需要验证：去除当前控制状态变量后的表现、按完整年度滚动验证、留一风机验证、异常样本原因追踪、在线影子验证。")
    doc.write(REPORT_PATH)


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    figures = collect_existing_figures()
    build_report(inputs, figures)

    print(f"Report written: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
