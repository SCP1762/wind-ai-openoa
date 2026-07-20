import numpy as np
import pandas as pd

from wind_ai.evaluation import regression_metrics, summarize_predictions


def test_regression_metrics_resets_indexes_before_residuals() -> None:
    actual = pd.Series([100.0, 200.0, 300.0], index=[10, 11, 12])
    predicted = pd.Series([100.0, 210.0, 270.0], index=[0, 1, 2])

    metrics = regression_metrics(actual, predicted)

    assert metrics["rows"] == 3
    assert np.isclose(metrics["mae_kw"], 40.0 / 3.0)
    assert np.isclose(metrics["mean_residual_kw"], 20.0 / 3.0)
    assert np.isclose(metrics["max_absolute_error_kw"], 30.0)


def test_summarize_predictions_groups_error_metrics() -> None:
    results = pd.DataFrame(
        {
            "Wind_turbine_name": ["T1", "T1", "T2"],
            "actual_kw": [100.0, 200.0, 300.0],
            "predicted_kw": [110.0, 190.0, 330.0],
        }
    )

    summary = summarize_predictions(results, "Wind_turbine_name")

    assert list(summary["Wind_turbine_name"]) == ["T1", "T2"]
    assert list(summary["rows"]) == [2, 1]
    assert pd.isna(summary.loc[summary["Wind_turbine_name"] == "T2", "r2"].iloc[0])
