import numpy as np
import pandas as pd

from wind_ai.features import (
    EXPERIMENT_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_experiment_features,
    build_power_features,
)


def test_build_power_features() -> None:
    frame = pd.DataFrame(
        {
            "Date_time": pd.date_range("2025-01-01", periods=3, freq="10min"),
            "Wind_turbine_name": ["T1"] * 3,
            "Ws_avg": [5.0, 6.0, 7.0],
            "Va_avg": [0.0, 90.0, 180.0],
            "Wa_avg": [0.0, 90.0, 180.0],
            "Ot_avg": [10.0, 11.0, 12.0],
            "Ba_avg": [1.0, 1.0, 1.0],
            "P_avg": [100.0, 200.0, 300.0],
        }
    )
    result = build_power_features(frame)
    assert len(result) == 3
    assert set(FEATURE_COLUMNS).issubset(result.columns)
    assert np.isclose(result.loc[0, "Va_avg_sin"], 0.0)


def test_build_experiment_features_merges_reanalysis_and_asset_data() -> None:
    scada = pd.DataFrame(
        {
            "Date_time": pd.date_range("2025-01-01", periods=3, freq="10min"),
            "Wind_turbine_name": ["T1"] * 3,
            "Ws_avg": [5.0, 6.0, 7.0],
            "Va_avg": [0.0, 90.0, 180.0],
            "Wa_avg": [0.0, 90.0, 180.0],
            "Ot_avg": [10.0, 11.0, 12.0],
            "Ba_avg": [1.0, 1.0, 1.0],
            "P_avg": [100.0, 200.0, 300.0],
        }
    )
    era5 = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=1, freq="h"),
            "ws_100m": [6.5],
            "dens_100m": [1.2],
            "t_2m": [283.0],
            "surf_pres": [101000.0],
            "winddirection_deg": [270.0],
        }
    )
    merra2 = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=1, freq="h"),
            "ws_50m": [5.5],
            "dens_50m": [1.21],
            "temp_2m": [284.0],
            "surface_pressure": [100900.0],
            "winddirection_deg": [180.0],
        }
    )
    asset = pd.DataFrame(
        {
            "Wind_turbine_name": ["T1"],
            "Rated_power": [2050.0],
            "Hub_height_m": [80.0],
            "Rotor_diameter_m": [82.0],
            "elevation_m": [411.0],
        }
    )

    result = build_experiment_features(
        {
            "scada": scada,
            "reanalysis": {"era5": era5, "merra2": merra2},
            "asset": asset,
        }
    )

    assert len(result) == 3
    assert set(EXPERIMENT_FEATURE_COLUMNS).issubset(result.columns)
    assert np.isclose(result.loc[0, "era5_ws_100m"], 6.5)
    assert np.isclose(result.loc[0, "merra2_ws_50m"], 5.5)
    assert np.isclose(result.loc[0, "asset_rated_power_kw"], 2050.0)
