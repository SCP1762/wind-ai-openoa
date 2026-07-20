"""Evaluation helpers shared by scripts and notebooks."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, object]:
    """Return the core regression metrics used across experiment reports."""
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    residual = actual - predicted
    absolute_error = residual.abs()

    r2 = None
    if len(actual) >= 2 and actual.nunique(dropna=True) > 1:
        r2 = float(r2_score(actual, predicted))

    return {
        "rows": int(len(actual)),
        "mae_kw": float(mean_absolute_error(actual, predicted)),
        "rmse_kw": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": r2,
        "mean_residual_kw": float(residual.mean()),
        "median_absolute_error_kw": float(absolute_error.median()),
        "p95_absolute_error_kw": float(absolute_error.quantile(0.95)),
        "max_absolute_error_kw": float(absolute_error.max()),
    }


def summarize_predictions(results: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Build grouped error summaries for one analysis dimension."""
    summaries = []

    for group_value, group in results.groupby(group_column, observed=True, dropna=False):
        metrics = regression_metrics(group["actual_kw"], group["predicted_kw"])
        summaries.append({group_column: group_value, **metrics})

    return pd.DataFrame(summaries)
