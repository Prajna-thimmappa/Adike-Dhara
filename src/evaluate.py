"""Model evaluation for Adike Dhara.

Strict chronological evaluation:

* Time-based train/test split (never shuffled).
* Walk-forward (expanding-window) validation.
* MAE, RMSE, MAPE for every model in one comparison table.
* Results saved to ``data/processed/evaluation.csv``.

A model is "useful" only if it beats both baselines (naive and 7-day MA).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models import (
    MovingAverage7,
    NaiveBaseline,
    ProphetForecaster,
    SarimaForecaster,
    XGBoostForecaster,
    LightGBMForecaster,
    _metrics,
)

DEFAULT_MODELS = {
    "naive": NaiveBaseline,
    "ma7": MovingAverage7,
    "sarima": SarimaForecaster,
    "prophet": ProphetForecaster,
    "xgboost": XGBoostForecaster,
    "lightgbm": LightGBMForecaster,
}


def time_series_split(series: pd.Series, test_size: float = 0.2):
    """Split a datetime-indexed series into train/test without shuffling."""
    n = len(series)
    cut = int(n * (1 - test_size))
    return series.iloc[:cut], series.iloc[cut:]


def walk_forward_evaluate(series: pd.Series, model_cls, n_splits: int = 5,
                          min_train: int = 60, exog: Optional[pd.DataFrame] = None):
    """Expanding-window walk-forward validation.

    Returns per-fold metrics and the pooled out-of-sample predictions.
    """
    n = len(series)
    fold_size = n // (n_splits + 1)
    fold_metrics: list[dict] = []
    all_y_true: list[float] = []
    all_y_pred: list[float] = []

    for i in range(1, n_splits + 1):
        cut = min_train + i * fold_size
        if cut >= n:
            break
        train = series.iloc[:cut]
        test = series.iloc[cut:cut + fold_size]

        fc = model_cls()
        if exog is not None:
            # Align exog to train/test windows by position.
            train_exog = exog.iloc[:cut] if len(exog) >= cut else exog
            test_exog = exog.iloc[cut:cut + fold_size] if len(exog) > cut else None
            fc.fit(train, exog=train_exog)
            pred = fc.predict(len(test), exog=test_exog)
        else:
            fc.fit(train)
            pred = fc.predict(len(test))

        m = _metrics(test.values, pred)
        m["fold"] = i
        m["n_train"] = int(len(train))
        m["n_test"] = int(len(test))
        fold_metrics.append(m)

        all_y_true.extend(test.values.tolist())
        all_y_pred.extend(pred.tolist())

    pooled = _metrics(np.asarray(all_y_true), np.asarray(all_y_pred))
    pooled["fold"] = "pooled"
    fold_metrics.append(pooled)
    return fold_metrics


def compare_models(series: pd.Series, *, models: Optional[dict] = None,
                   n_splits: int = 5, min_train: int = 60,
                   exog: Optional[pd.DataFrame] = None,
                   output_path: str | Path = "data/processed/evaluation.csv"):
    """Run walk-forward evaluation across all models and save a comparison table."""
    if models is None:
        models = DEFAULT_MODELS

    # Interpolate missing mandi days (holidays/Sundays) so univariate models
    # receive a complete series. Missing days remain flagged in the raw CSV;
    # this step is for modeling only.
    series = series.sort_index()
    series = series.interpolate(limit_direction="both").astype(float)

    # Align exogenous features to the series index (inner join), so positional
    # slicing in walk-forward stays consistent. When exog is provided, restrict
    # the series to the same dates (complete-feature dates only).
    if exog is not None:
        exog = exog.loc[exog.index.intersection(series.index)]
        series = series.loc[exog.index]

    rows: list[dict] = []
    for name, cls in models.items():
        try:
            folds = walk_forward_evaluate(
                series, cls, n_splits=n_splits, min_train=min_train, exog=exog
            )
            pooled = next(f for f in folds if f["fold"] == "pooled")
            rows.append({
                "model": name,
                "mae": pooled["mae"],
                "rmse": pooled["rmse"],
                "mape": pooled["mape"],
                "usable": False,  # filled below
            })
        except Exception as exc:  # pragma: no cover - one model shouldn't kill the run
            rows.append({
                "model": name,
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "usable": False,
                "error": str(exc),
            })

    table = pd.DataFrame(rows)

    # A model is useful only if it beats both baselines on pooled MAE.
    baseline_mae = {}
    for b in ("naive", "ma7"):
        if b in table["model"].values:
            baseline_mae[b] = float(table.loc[table["model"] == b, "mae"].iloc[0])
    best_baseline = min(baseline_mae.values()) if baseline_mae else float("inf")

    table["usable"] = table.apply(
        lambda r: bool(pd.notna(r["mae"]) and r["mae"] < best_baseline), axis=1
    )

    table = table.sort_values("mae").reset_index(drop=True)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"[evaluate] Saved comparison table -> {out_path}")
    print(table.to_string(index=False))
    return table


def best_model(table: pd.DataFrame) -> Optional[str]:
    """Return the name of the best usable model, or None if none qualify."""
    usable = table[table["usable"] == True]
    if usable.empty:
        return None
    return str(usable.iloc[0]["model"])
