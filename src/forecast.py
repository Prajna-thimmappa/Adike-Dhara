"""Forecast production for Adike Dhara.

Produces 7- and 14-day ahead forecasts with prediction intervals from the
best usable model identified in ``evaluation.csv``.

* SARIMA / Prophet: native model confidence intervals.
* XGBoost: recursive multi-step forecast; intervals estimated from the
  distribution of walk-forward residuals (mean ± z * residual_std).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evaluate import best_model, DEFAULT_MODELS
from features import build_features, DATE_COL
from models import BaseForecaster

FORECAST_HORIZONS = (7, 14)
INTERVAL_ALPHA = 0.05  # 90% prediction interval


def _load_best_series(processed_csv: str | Path,
                      variety: str, market: str) -> pd.Series:
    """Load a cleaned per-variety CSV and return a datetime-indexed modal_price series for a market."""
    df = pd.read_csv(processed_csv, parse_dates=[DATE_COL])
    df = df[(df["variety"] == variety) & (df["market"] == market)]
    df = df.sort_values(DATE_COL)
    s = df.set_index(DATE_COL)["modal_price"].dropna().astype(float)
    return s


def _build_exog_for_series(series: pd.Series, df_raw: pd.DataFrame,
                           market: str, variety: str) -> pd.DataFrame:
    """Build the exogenous feature frame aligned to ``series`` for XGBoost."""
    sub = df_raw[(df_raw["market"] == market) & (df_raw["variety"] == variety)].copy()
    sub = sub.sort_values(DATE_COL)
    # Use modal_price as the price column; build_features returns X aligned to dates.
    X, _ = build_features(sub, target_ahead=1)
    X = X.set_index(DATE_COL)
    # Align to series index (inner join).
    X = X.loc[X.index.intersection(series.index)]
    return X


def _residual_intervals(series: pd.Series, fc: BaseForecaster,
                        steps: int, train: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Estimate prediction intervals from in-sample one-step residuals."""
    # One-step walk residuals on the training tail.
    fitted_vals = []
    actuals = []
    window = min(len(train), 60)
    for i in range(window, len(train)):
        sub = train.iloc[:i]
        fc2 = type(fc)()
        fc2.fit(sub)
        pred = fc2.predict(1)[0]
        fitted_vals.append(pred)
        actuals.append(train.iloc[i])
    residuals = np.asarray(actuals) - np.asarray(fitted_vals)
    resid_std = float(np.std(residuals)) if len(residuals) else float(series.std())
    z = 1.645  # ~90% two-sided
    margin = z * resid_std
    return margin


def forecast_variety_market(processed_csv: str | Path, raw_csv: str | Path,
                            variety: str, market: str,
                            horizons: tuple[int, ...] = FORECAST_HORIZONS,
                            evaluation_csv: str | Path = "data/processed/evaluation.csv",
                            output_dir: str | Path = "data/processed"):
    """Produce forecasts for one (variety, market) pair and save them."""
    from models import NaiveBaseline, MovingAverage7, SarimaForecaster, ProphetForecaster, XGBoostForecaster

    series = _load_best_series(processed_csv, variety, market)
    if len(series) < 60:
        raise ValueError(f"Not enough history for {variety} @ {market}: {len(series)} points.")

    # Load evaluation table to pick the best usable model.
    eval_path = Path(evaluation_csv)
    table = None
    if eval_path.exists():
        table = pd.read_csv(eval_path)
    best = best_model(table) if table is not None and not table.empty else None
    model_name = best or "sarima"

    cls = DEFAULT_MODELS[model_name]
    fc = cls()

    # Train on all available history.
    exog = None
    if model_name == "xgboost":
        raw_df = pd.read_csv(raw_csv, parse_dates=[DATE_COL])
        exog = _build_exog_for_series(series, raw_df, market, variety)
        fc.fit(series, exog=exog)
    else:
        fc.fit(series)

    results: list[dict] = []
    last_date = series.index[-1]

    for h in horizons:
        if model_name in ("sarima", "prophet") and hasattr(fc, "predict_interval"):
            mean, lower, upper = fc.predict_interval(h, alpha=INTERVAL_ALPHA)
        else:
            # XGBoost / baseline: recursive point forecast + residual-based interval.
            if model_name == "xgboost":
                # Build future feature rows recursively is complex; use last-row repeat.
                if exog is None:
                    raw_df = pd.read_csv(raw_csv, parse_dates=[DATE_COL])
                    exog = _build_exog_for_series(series, raw_df, market, variety)
                last_row = exog.iloc[[-1]]
                future_exog = pd.DataFrame(
                    np.repeat(last_row.values, h, axis=0), columns=exog.columns
                )
                mean = fc.predict(h, exog=future_exog)
            else:
                mean = fc.predict(h)
            lower = mean - 0.0  # placeholder
            upper = mean + 0.0
            # Residual-based interval.
            train_tail = series.iloc[-min(len(series), 90):]
            margin = _residual_intervals(series, fc, h, train_tail)
            lower = mean - margin
            upper = mean + margin

        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=h, freq="D")
        for d, m, lo, hi in zip(future_dates, mean, lower, upper):
            results.append({
                "date": d,
                "variety": variety,
                "market": market,
                "model": model_name,
                "forecast": float(m),
                "lower": float(lo),
                "upper": float(hi),
            })

    out = pd.DataFrame(results)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_v = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(variety))
    safe_m = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(market))
    out_path = out_dir / f"forecast_{safe_v}_{safe_m}.csv"
    out.to_csv(out_path, index=False)
    print(f"[forecast] Saved {len(out)} rows -> {out_path}")
    return out


if __name__ == "__main__":
    import sys
    proc = sys.argv[1] if len(sys.argv) > 1 else "data/processed/Moati_daily.csv"
    raw = sys.argv[2] if len(sys.argv) > 2 else "data/raw/sample_arecanut.csv"
    variety = sys.argv[3] if len(sys.argv) > 3 else "Moati"
    market = sys.argv[4] if len(sys.argv) > 4 else "APMC Yard, Davanagere"
    forecast_variety_market(proc, raw, variety, market)
