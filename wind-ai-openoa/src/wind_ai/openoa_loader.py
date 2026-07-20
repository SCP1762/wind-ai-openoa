"""Load the OpenOA ENGIE example data into a PlantData object.

The cleaning steps follow the official OpenOA `examples/project_ENGIE.py`
workflow, but are kept in this project so later ML code does not depend on the
OpenOA repository layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openoa.plant import PlantData
from openoa.utils import filters
from openoa.utils import met_data_processing as met

from .paths import PLANT_METADATA_PATH, RAW_DATA_DIR

REQUIRED_FILES = (
    "la-haute-borne-data-2014-2015.csv",
    "plant_data.csv",
    "merra2_la_haute_borne.csv",
    "era5_wind_la_haute_borne.csv",
    "la-haute-borne_asset_table.csv",
)


def assert_dataset_exists(data_dir: Path = RAW_DATA_DIR) -> None:
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "数据尚未准备完成。请先运行 `python scripts/download_data.py`。"
            f"\n缺少文件：\n  - {joined}"
        )


def clean_scada(scada_file: Path) -> pd.DataFrame:
    """Read and minimally clean 10-minute turbine SCADA data."""
    scada = pd.read_csv(scada_file)
    scada["Date_time"] = pd.to_datetime(scada["Date_time"], utc=True).dt.tz_localize(None)
    scada = scada.drop_duplicates(
        subset=["Date_time", "Wind_turbine_name"], keep="first"
    ).copy()

    scada = scada.loc[scada["Ot_avg"].between(-15.0, 45.0)].copy()

    sensor_cols = ["Ba_avg", "P_avg", "Ws_avg", "Va_avg", "Ot_avg", "Ya_avg", "Wa_avg"]
    for turbine_id in scada["Wind_turbine_name"].dropna().unique():
        turbine_mask = scada["Wind_turbine_name"] == turbine_id
        turbine_data = scada.loc[turbine_mask]

        vane_flags = filters.unresponsive_flag(turbine_data, 3, col=["Va_avg"])
        bad_vane_index = vane_flags.index[vane_flags["Va_avg"]]
        scada.loc[bad_vane_index, sensor_cols] = np.nan

        temp_flags = filters.unresponsive_flag(turbine_data, 20, col=["Ot_avg"])
        bad_temp_index = temp_flags.index[temp_flags["Ot_avg"]]
        scada.loc[bad_temp_index, "Ot_avg"] = np.nan

    scada["Ba_avg"] = scada["Ba_avg"] % 360
    scada.loc[scada["Ba_avg"] > 180, "Ba_avg"] -= 360

    # 10-minute average power (kW) converted to energy (kWh).
    scada["energy_kwh"] = scada["P_avg"] / 6.0
    return scada


def _load_meter_and_curtail(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.read_csv(data_dir / "plant_data.csv")

    meter = combined.copy()
    meter["time"] = pd.to_datetime(meter["time_utc"]).dt.tz_localize(None)
    meter = meter.drop(columns=["time_utc", "availability_kwh", "curtailment_kwh"])

    curtail = combined.copy()
    curtail["time"] = pd.to_datetime(curtail["time_utc"]).dt.tz_localize(None)
    curtail = curtail.drop(columns=["time_utc"])
    return meter, curtail


def _load_reanalysis(data_dir: Path) -> dict[str, pd.DataFrame]:
    merra2 = pd.read_csv(data_dir / "merra2_la_haute_borne.csv")
    merra2["datetime"] = pd.to_datetime(merra2["datetime"], utc=True).dt.tz_localize(None)
    merra2["winddirection_deg"] = met.compute_wind_direction(merra2["u_50"], merra2["v_50"])
    merra2 = merra2.drop(columns=["Unnamed: 0"], errors="ignore")

    era5 = pd.read_csv(data_dir / "era5_wind_la_haute_borne.csv")
    era5 = era5.loc[:, ~era5.columns.duplicated()].copy()
    era5["datetime"] = pd.to_datetime(era5["datetime"], utc=True).dt.tz_localize(None)
    era5 = era5.set_index(pd.DatetimeIndex(era5["datetime"])).asfreq("1h")
    era5["datetime"] = era5.index
    era5["winddirection_deg"] = met.compute_wind_direction(
        era5["u_100"], era5["v_100"]
    ).values
    era5 = era5.drop(columns=["Unnamed: 0"], errors="ignore")

    return {"era5": era5, "merra2": merra2}


def load_dataframes(data_dir: Path = RAW_DATA_DIR) -> dict[str, object]:
    """Load cleaned dataframes without creating a PlantData object."""
    data_dir = Path(data_dir)
    assert_dataset_exists(data_dir)

    scada = clean_scada(data_dir / "la-haute-borne-data-2014-2015.csv")
    meter, curtail = _load_meter_and_curtail(data_dir)
    reanalysis = _load_reanalysis(data_dir)

    asset = pd.read_csv(data_dir / "la-haute-borne_asset_table.csv")
    asset["type"] = "turbine"

    return {
        "scada": scada,
        "meter": meter,
        "curtail": curtail,
        "asset": asset,
        "reanalysis": reanalysis,
    }


def load_plant(
    data_dir: Path = RAW_DATA_DIR,
    metadata_path: Path = PLANT_METADATA_PATH,
) -> PlantData:
    """Create and return a validated OpenOA PlantData object."""
    frames = load_dataframes(Path(data_dir))
    plant = PlantData(
        analysis_type="MonteCarloAEP",
        metadata=Path(metadata_path),
        scada=frames["scada"],
        meter=frames["meter"],
        curtail=frames["curtail"],
        asset=frames["asset"],
        reanalysis=frames["reanalysis"],
    )
    plant.validate()
    return plant
