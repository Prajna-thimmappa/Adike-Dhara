"""Feature engineering for Adike Dhara arecanut price forecasting.

Builds a supervised feature matrix from cleaned daily prices:

* Lag features: modal_price at t-1, t-7, t-14, t-30.
* Rolling mean/std over trailing windows 7, 14, 30 (computed on past values
  only, so no future information leaks into the features).
* Calendar features: month, day-of-week.

Target is next-day modal price (t+1), so every row's features describe only
information available at or before day t.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATE_COL = "date"
TARGET_COL = "modal_price"
LAG_DAYS = (1, 7, 14, 30)
ROLL_WINDOWS = (7, 14, 30)


def load_processed(path: str | Path) -> pd.DataFrame:
    """Load a cleaned per-variety CSV and parse dates."""
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    return df.sort_values(DATE_COL).reset_index(drop=True)


def build_features(
    df: pd.DataFrame,
    *,
    target_ahead: int = 1,
    group_cols: tuple[str, ...] = ("market", "variety"),
    price_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create lag / rolling / calendar features with a next-day target.

    Parameters
    ----------
    df:
        Cleaned daily price frame (one row per market-variety-date).
    target_ahead:
        Days ahead to predict (1 = tomorrow).
    group_cols:
        Columns defining an independent price series.
    price_col:
        Source price column.

    Returns
    -------
    X, y
        Feature frame aligned with ``y``; rows with missing targets dropped.
    """
    df = df.copy().sort_values(list(group_cols) + [DATE_COL])

    # Calendar features from the date itself (never future information).
    df["month"] = df[DATE_COL].dt.month
    df["day_of_week"] = df[DATE_COL].dt.day_of_week

    # Lag / rolling / target computed within each (market, variety) series
    # using transform so every original column is preserved.
    gby = df.groupby(list(group_cols))[price_col]

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = gby.transform(lambda s, lag=lag: s.shift(lag))

    shifted = gby.transform(lambda s: s.shift(1))
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = gby.transform(lambda s, w=w: s.shift(1).rolling(window=w, min_periods=1).mean())
        df[f"roll_std_{w}"] = gby.transform(lambda s, w=w: s.shift(1).rolling(window=w, min_periods=1).std())

    # Target: price ``target_ahead`` days in the future (pure future value,
    # used only as the label, never as a feature).
    df["target"] = gby.transform(lambda s: s.shift(-target_ahead))

    feat = df.dropna(subset=["target"]).copy()

    feature_cols = [c for c in feat.columns
                    if c.startswith(("lag_", "roll_mean_", "roll_std_"))
                    or c in ("month", "day_of_week")]

    X = feat[feature_cols].copy()
    y = feat["target"].copy()

    # Align index back to the date for convenience.
    X[DATE_COL] = feat[DATE_COL].values
    for col in group_cols:
        X[col] = feat[col].values

    # Keep only fully observed feature rows (no NaN lags at series start).
    mask = X[feature_cols].notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    X[DATE_COL] = X[DATE_COL].reset_index(drop=True)
    for col in group_cols:
        X[col] = X[col].reset_index(drop=True)

    return X, y


def split_time(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    date_col: str = DATE_COL,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Strict chronological train/test split (no shuffling)."""
    order = np.argsort(X[date_col].values, kind="mergesort")
    n = len(X)
    cut = int(n * (1 - test_size))
    train_idx, test_idx = order[:cut], order[cut:]
    return (
        X.iloc[train_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


def walk_forward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    date_col: str = DATE_COL,
    n_splits: int = 5,
    min_train: int = 60,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window train/test index pairs in time order.

    Each fold trains on all history up to a cut point and tests on the next
    contiguous block. The train window only ever expands (never rolls), so
    forecasts are always strictly out-of-sample.
    """
    order = np.argsort(X[date_col].values, kind="mergesort")
    n = len(order)
    test_size = n // (n_splits + 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_splits + 1):
        cut = min_train + i * test_size
        if cut >= n:
            break
        train_idx = order[:cut]
        test_idx = order[cut : cut + test_size]
        splits.append((train_idx, test_idx))
    return splits
