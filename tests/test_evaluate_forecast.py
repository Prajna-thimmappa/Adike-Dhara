"""Unit tests for Adike Dhara evaluation and forecasting modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluate import best_model, compare_models, time_series_split, walk_forward_evaluate
from forecast import FORECAST_HORIZONS, forecast_variety_market
from models import NaiveBaseline, SarimaForecaster


def _make_series(n=200, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    trend = np.linspace(400, 470, n)
    seasonal = 8 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = rng.normal(0, 2, n)
    prices = trend + seasonal + noise
    return pd.Series(prices, index=dates)


def test_time_series_split_chronological():
    series = _make_series(100)
    train, test = time_series_split(series, test_size=0.2)
    assert len(train) + len(test) == len(series)
    assert train.index.max() <= test.index.min()


def test_walk_forward_evaluate_returns_folds():
    series = _make_series(120)
    folds = walk_forward_evaluate(series, NaiveBaseline, n_splits=3, min_train=60)
    assert len(folds) >= 1
    pooled = [f for f in folds if f["fold"] == "pooled"]
    assert len(pooled) == 1
    assert "mae" in pooled[0] and "rmse" in pooled[0] and "mape" in pooled[0]


def test_compare_models_saves_csv_and_marks_usable(tmp_path):
    series = _make_series(150)
    out_path = tmp_path / "evaluation.csv"
    table = compare_models(
        series,
        models={"naive": NaiveBaseline, "ma7": NaiveBaseline},
        n_splits=2,
        min_train=60,
        output_path=str(out_path),
    )
    assert out_path.exists()
    assert "model" in table.columns
    assert "mae" in table.columns
    assert "usable" in table.columns
    assert len(table) == 2


def test_best_model_returns_none_when_none_usable():
    table = pd.DataFrame(
        {
            "model": ["sarima", "prophet"],
            "mae": [500.0, 600.0],
            "rmse": [600.0, 700.0],
            "mape": [100.0, 120.0],
            "usable": [False, False],
        }
    )
    assert best_model(table) is None


def test_best_model_returns_best_usable():
    table = pd.DataFrame(
        {
            "model": ["naive", "ma7", "sarima"],
            "mae": [10.0, 8.0, 5.0],
            "rmse": [12.0, 10.0, 7.0],
            "mape": [2.0, 1.5, 1.0],
            "usable": [True, True, True],
        }
    ).sort_values("mae").reset_index(drop=True)
    assert best_model(table) == "sarima"


def test_forecast_variety_market(tmp_path):
    series = _make_series(150)
    df = pd.DataFrame(
        {
            "date": series.index,
            "variety": "Moati",
            "market": "APMC Yard, Davanagere",
            "modal_price": series.values,
        }
    )
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    processed_csv = processed_dir / "Moati_daily.csv"
    df.to_csv(processed_csv, index=False)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_csv = raw_dir / "sample_arecanut.csv"
    df.to_csv(raw_csv, index=False)

    eval_dir = tmp_path / "processed"
    eval_dir.mkdir(exist_ok=True)
    eval_csv = eval_dir / "evaluation.csv"
    eval_table = pd.DataFrame(
        {
            "model": ["naive", "ma7", "sarima"],
            "mae": [10.0, 8.0, 5.0],
            "rmse": [12.0, 10.0, 7.0],
            "mape": [2.0, 1.5, 1.0],
            "usable": [True, True, True],
        }
    ).sort_values("mae").reset_index(drop=True)
    eval_table.to_csv(eval_csv, index=False)

    out = forecast_variety_market(
        str(processed_csv),
        str(raw_csv),
        "Moati",
        "APMC Yard, Davanagere",
        horizons=(7,),
        evaluation_csv=str(eval_csv),
        output_dir=str(eval_dir),
    )

    assert len(out) == 7
    assert list(out.columns) == [
        "date",
        "variety",
        "market",
        "model",
        "forecast",
        "lower",
        "upper",
    ]
    assert (out["model"] == "sarima").all()
    assert (out["variety"] == "Moati").all()
    assert (out["market"] == "APMC Yard, Davanagere").all()
    assert np.all(np.isfinite(out["forecast"].values))
    assert np.all(np.isfinite(out["lower"].values))
    assert np.all(np.isfinite(out["upper"].values))
    assert (out["lower"] <= out["forecast"]).all()
    assert (out["forecast"] <= out["upper"]).all()


def test_forecast_horizons_constant():
    assert FORECAST_HORIZONS == (7, 14)
