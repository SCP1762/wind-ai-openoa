"""Feature preparation for the first power-prediction baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "Ws_avg",
    "Va_avg_sin",
    "Va_avg_cos",
    "Wa_avg_sin",
    "Wa_avg_cos",
    "Ot_avg",
    "Ba_avg",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]
ENRICHED_FEATURE_COLUMNS = [
    "era5_ws_100m",
    "era5_dens_100m",
    "era5_t_2m",
    "era5_surf_pres",
    "era5_winddirection_sin",
    "era5_winddirection_cos",
    "merra2_ws_50m",
    "merra2_dens_50m",
    "merra2_temp_2m",
    "merra2_surface_pressure",
    "merra2_winddirection_sin",
    "merra2_winddirection_cos",
    "asset_rated_power_kw",
    "asset_hub_height_m",
    "asset_rotor_diameter_m",
    "asset_elevation_m",
]
EXPERIMENT_FEATURE_COLUMNS = FEATURE_COLUMNS + ENRICHED_FEATURE_COLUMNS
TARGET_COLUMN = "P_avg"


def build_power_features(scada: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Date_time",
        "Wind_turbine_name",
        "Ws_avg",
        "Va_avg",
        "Wa_avg",
        "Ot_avg",
        "Ba_avg",
        "P_avg",
    }
    missing = sorted(required.difference(scada.columns))
    if missing:
        raise ValueError(f"SCADA 数据缺少字段：{missing}")

    frame = scada.loc[:, sorted(required)].copy()
    frame["Date_time"] = pd.to_datetime(frame["Date_time"])

    for angle in ("Va_avg", "Wa_avg"):
        radians = np.deg2rad(frame[angle])
        frame[f"{angle}_sin"] = np.sin(radians)
        frame[f"{angle}_cos"] = np.cos(radians)

    hour = frame["Date_time"].dt.hour + frame["Date_time"].dt.minute / 60.0
    month = frame["Date_time"].dt.month
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Remove obviously impossible values before model training.
    frame = frame.loc[
        frame["Ws_avg"].between(0, 40)
        & frame["P_avg"].between(-100, 2500)
    ].dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    return frame.sort_values("Date_time").reset_index(drop=True)


def build_experiment_features(frames: dict[str, object]) -> pd.DataFrame:
    """Build the enriched experiment table from loaded OpenOA dataframes.

    The model uses turbine-level SCADA, hourly ERA5/MERRA2 weather context, and
    static asset metadata. Plant meter, availability, and curtailment data are
    intentionally kept for reporting/diagnostics rather than direct model
    features, because they are plant-level contemporaneous outcome signals.
    """
    scada = frames["scada"]
    if not isinstance(scada, pd.DataFrame):
        raise TypeError("frames['scada'] must be a pandas DataFrame")

    dataset = build_power_features(scada)
    reanalysis = frames.get("reanalysis", {})
    if not isinstance(reanalysis, dict):
        raise TypeError("frames['reanalysis'] must be a dictionary of dataframes")

    dataset = _merge_reanalysis(dataset, reanalysis, "era5", "datetime")
    dataset = _merge_reanalysis(dataset, reanalysis, "merra2", "datetime")
    dataset = _merge_asset_metadata(dataset, frames.get("asset"))

    return (
        dataset.dropna(subset=EXPERIMENT_FEATURE_COLUMNS + [TARGET_COLUMN])
        .sort_values("Date_time")
        .reset_index(drop=True)
    )


def _merge_reanalysis(
    dataset: pd.DataFrame,
    reanalysis: dict[str, pd.DataFrame],
    name: str,
    time_column: str,
) -> pd.DataFrame:
    source = reanalysis.get(name)
    if source is None:
        raise ValueError(f"reanalysis data missing source: {name}")

    source = source.copy().reset_index(drop=True)
    source[time_column] = pd.to_datetime(source[time_column])

    if name == "era5":
        selected = source[
            [
                time_column,
                "ws_100m",
                "dens_100m",
                "t_2m",
                "surf_pres",
                "winddirection_deg",
            ]
        ].rename(
            columns={
                "ws_100m": "era5_ws_100m",
                "dens_100m": "era5_dens_100m",
                "t_2m": "era5_t_2m",
                "surf_pres": "era5_surf_pres",
                "winddirection_deg": "era5_winddirection_deg",
            }
        )
        direction_column = "era5_winddirection_deg"
    elif name == "merra2":
        selected = source[
            [
                time_column,
                "ws_50m",
                "dens_50m",
                "temp_2m",
                "surface_pressure",
                "winddirection_deg",
            ]
        ].rename(
            columns={
                "ws_50m": "merra2_ws_50m",
                "dens_50m": "merra2_dens_50m",
                "temp_2m": "merra2_temp_2m",
                "surface_pressure": "merra2_surface_pressure",
                "winddirection_deg": "merra2_winddirection_deg",
            }
        )
        direction_column = "merra2_winddirection_deg"
    else:
        raise ValueError(f"Unsupported reanalysis source: {name}")

    selected = selected.sort_values(time_column).drop_duplicates(time_column)
    direction_radians = np.deg2rad(selected[direction_column])
    selected[f"{name}_winddirection_sin"] = np.sin(direction_radians)
    selected[f"{name}_winddirection_cos"] = np.cos(direction_radians)

    merged = pd.merge_asof(
        dataset.sort_values("Date_time"),
        selected.sort_values(time_column),
        left_on="Date_time",
        right_on=time_column,
        direction="backward",
        tolerance=pd.Timedelta("1h"),
    )
    return merged.drop(columns=[time_column, direction_column])


def _merge_asset_metadata(dataset: pd.DataFrame, asset: object) -> pd.DataFrame:
    if not isinstance(asset, pd.DataFrame):
        raise TypeError("frames['asset'] must be a pandas DataFrame")

    asset_columns = asset[
        [
            "Wind_turbine_name",
            "Rated_power",
            "Hub_height_m",
            "Rotor_diameter_m",
            "elevation_m",
        ]
    ].rename(
        columns={
            "Rated_power": "asset_rated_power_kw",
            "Hub_height_m": "asset_hub_height_m",
            "Rotor_diameter_m": "asset_rotor_diameter_m",
            "elevation_m": "asset_elevation_m",
        }
    )
    return dataset.merge(asset_columns, on="Wind_turbine_name", how="left")
