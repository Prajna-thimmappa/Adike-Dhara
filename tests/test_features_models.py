"""Unit tests for Adike Dhara features, models, and evaluation modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import build_features, split_time, walk_forward_splits
from models import (
    MovingAverage7,
    NaiveBaseline,
    SarimaForecaster,
    ProphetForecaster,
    XGBoostForecaster,
    LightGBMForecaster,
    _mape,
    _metrics,
)


def _make_series(n=100):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    prices = 400 + np.cumsum(rng.normal(0, 2, n))
    return pd.Series(prices, index=dates)


def test_build_features_no_future_leakage():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=40, freq="D"),
            "market": ["M"] * 40,
            "variety": ["V"] * 40,
            "modal_price": np.arange(400, 440, dtype=float),
        }
    )
    X, y = build_features(df)
    assert len(X) == len(y)
    # Feature columns must not contain the target column itself.
    feat_cols = [c for c in X.columns if c not in ("date", "market", "variety")]
    assert "target" not in feat_cols
    assert "modal_price" not in feat_cols
    # No NaN in features after the dropna step.
    assert X[feat_cols].notna().all().all()
    # Target is ahead: first valid row (index 30, after lag_30 warmup)
    # targets price at index 31 = 431.
    assert y.iloc[0] == pytest.approx(431.0)


def test_split_time_chronological():
    series = _make_series(50)
    X = pd.DataFrame({"x": series.values, "date": series.index})
    y = series
    tr_X, tr_y, te_X, te_y = split_time(X, y, test_size=0.2)
    assert len(tr_X) + len(te_X) == len(X)
    # All train dates must precede all test dates.
    assert tr_X["date"].max() <= te_X["date"].min()


def test_walk_forward_splits_expanding():
    series = _make_series(60)
    X = pd.DataFrame({"x": series.values, "date": series.index})
    y = series
    splits = walk_forward_splits(X, y, n_splits=3, min_train=30)
    assert len(splits) >= 1
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()  # train before test


def test_naive_baseline():
    series = _make_series(20)
    fc = NaiveBaseline()
    fc.fit(series)
    pred = fc.predict(3)
    assert len(pred) == 3
    assert np.allclose(pred, series.iloc[-1])


def test_moving_average7():
    series = _make_series(20)
    fc = MovingAverage7()
    fc.fit(series)
    pred = fc.predict(3)
    assert len(pred) == 3
    expected = series.iloc[-7:].mean()
    assert np.allclose(pred, expected)


def test_metrics_and_mape():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])
    m = _metrics(y_true, y_pred)
    assert "mae" in m and "rmse" in m and "mape" in m
    assert m["mae"] == pytest.approx(50 / 3)
    assert 0 < m["mape"] < 100


def test_mape_zero_handling():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([10.0, 110.0])
    assert _mape(y_true, y_pred) == pytest.approx(10.0)  # only the 100 counts


def _make_price_series(n=120, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    trend = np.linspace(400, 460, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = rng.normal(0, 3, n)
    prices = trend + seasonal + noise
    return pd.Series(prices, index=dates)


def test_sarima_forecaster_fit_predict():
    series = _make_price_series(n=100)
    fc = SarimaForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    fc.fit(series)
    pred = fc.predict(7)
    assert len(pred) == 7
    assert np.all(np.isfinite(pred))


def test_sarima_forecaster_predict_interval():
    series = _make_price_series(n=100)
    fc = SarimaForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    fc.fit(series)
    mean, lower, upper = fc.predict_interval(7, alpha=0.05)
    assert len(mean) == 7
    assert len(lower) == 7
    assert len(upper) == 7
    assert np.all(lower <= mean)
    assert np.all(mean <= upper)


def test_prophet_forecaster_fit_predict():
    series = _make_price_series(n=100)
    fc = ProphetForecaster()
    fc.fit(series)
    pred = fc.predict(7)
    assert len(pred) == 7
    assert np.all(np.isfinite(pred))


def test_prophet_forecaster_predict_interval():
    series = _make_price_series(n=100)
    fc = ProphetForecaster()
    fc.fit(series)
    mean, lower, upper = fc.predict_interval(7, alpha=0.05)
    assert len(mean) == 7
    assert np.all(lower <= mean)
    assert np.all(mean <= upper)


def test_xgboost_forecaster_fit_predict():
    series = _make_price_series(n=100)
    # Build a simple exog frame aligned to series index
    exog = pd.DataFrame(
        {
            "lag_1": series.shift(1),
            "lag_7": series.shift(7),
            "roll_mean_7": series.shift(1).rolling(7, min_periods=1).mean(),
        },
        index=series.index,
    ).dropna()
    series_aligned = series.loc[exog.index]
    fc = XGBoostForecaster(n_estimators=10, max_depth=3)
    fc.fit(series_aligned, exog=exog)
    pred = fc.predict(5, exog=exog.tail(5))
    assert len(pred) == 5
    assert np.all(np.isfinite(pred))


def test_lightgbm_forecaster_fit_predict():
    series = _make_price_series(n=100)
    exog = pd.DataFrame(
        {
            "lag_1": series.shift(1),
            "lag_7": series.shift(7),
            "roll_mean_7": series.shift(1).rolling(7, min_periods=1).mean(),
        },
        index=series.index,
    ).dropna()
    series_aligned = series.loc[exog.index]
    fc = LightGBMForecaster(n_estimators=10, max_depth=3)
    fc.fit(series_aligned, exog=exog)
    pred = fc.predict(5, exog=exog.tail(5))
    assert len(pred) == 5
    assert np.all(np.isfinite(pred))
