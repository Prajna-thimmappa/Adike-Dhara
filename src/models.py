"""Forecasting models for Adike Dhara.

Implements, behind a common ``fit``/``predict`` interface:

* ``NaiveBaseline`` - tomorrow = today.
* ``MovingAverage7`` - 7-day trailing mean.
* ``SarimaForecaster`` - statsmodels SARIMA on the univariate series.
* ``ProphetForecaster`` - Prophet on the univariate series.
* ``XGBoostForecaster`` - XGBoost on lag / rolling / calendar features.

All forecasters expose ``fit(series)`` (univariate, datetime-indexed) and
``predict(steps)`` returning an array of length ``steps``. The XGBoost variant
also accepts an optional exogenous feature frame for multi-step recursion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class BaseForecaster:
    """Common interface for all forecasters."""

    name: str = "base"

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, series: pd.Series, steps: int,
                 exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Convenience: fit then predict."""
        self.fit(series, exog)
        return self.predict(steps, exog)


@dataclass
class FitResult:
    model_name: str
    train_mae: float
    train_rmse: float
    train_mape: float


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": _mape(y_true, y_pred),
    }


class NaiveBaseline(BaseForecaster):
    """Tomorrow's price = today's price (last observed value)."""

    name = "naive"

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "NaiveBaseline":
        s = series.dropna()
        if s.empty:
            raise ValueError("Series has no observed values.")
        self.last_value_ = float(s.iloc[-1])
        self.last_date_ = s.index[-1]
        return self

    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        return np.full(steps, self.last_value_)


class MovingAverage7(BaseForecaster):
    """7-day trailing mean forecast (constant horizon value)."""

    name = "ma7"

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "MovingAverage7":
        s = series.dropna()
        self.last_value_ = float(s.rolling(window=7, min_periods=1).mean().iloc[-1])
        return self

    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        return np.full(steps, self.last_value_)


class SarimaForecaster(BaseForecaster):
    """SARIMA wrapper using statsmodels."""

    name = "sarima"

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7)) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted = None

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "SarimaForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        s = series.dropna().astype(float)
        self._fitted = SARIMAX(
            s, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        fc = self._fitted.get_forecast(steps=steps)
        return np.asarray(fc.predicted_mean, dtype=float)

    def predict_interval(self, steps: int, alpha: float = 0.05):
        """Point forecast plus (lower, upper) confidence intervals."""
        fc = self._fitted.get_forecast(steps=steps, alpha=alpha)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(), dtype=float)
        return mean, ci[:, 0], ci[:, 1]


class ProphetForecaster(BaseForecaster):
    """Prophet wrapper. Requires ``prophet`` to be installed."""

    name = "prophet"

    def __init__(self, yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False) -> None:
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self._fitted = None
        self._last_date = None

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "ProphetForecaster":
        from prophet import Prophet
        s = series.dropna()
        self._last_date = s.index[-1]
        df = pd.DataFrame({
            "ds": s.index,
            "y": s.values.astype(float),
        })
        self._fitted = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )
        self._fitted.fit(df)
        return self

    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        future = self._fitted.make_future_dataframe(periods=steps, freq="D")
        forecast = self._fitted.predict(future)
        tail = forecast.tail(steps)
        return np.asarray(tail["yhat"].values, dtype=float)

    def predict_interval(self, steps: int, alpha: float = 0.05):
        from prophet import Prophet  # noqa: F401  (ensures import path)
        future = self._fitted.make_future_dataframe(periods=steps, freq="D")
        forecast = self._fitted.predict(future)
        tail = forecast.tail(steps)
        mean = np.asarray(tail["yhat"].values, dtype=float)
        lower = np.asarray(tail["yhat_lower"].values, dtype=float)
        upper = np.asarray(tail["yhat_upper"].values, dtype=float)
        return mean, lower, upper


class XGBoostForecaster(BaseForecaster):
    """XGBoost regressor on lag / rolling / calendar features."""

    name = "xgboost"

    def __init__(self, n_estimators: int = 300, max_depth: int = 5,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self._model = None
        self._feature_cols: list[str] = []
        self._last_row: Optional[pd.Series] = None

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "XGBoostForecaster":
        from xgboost import XGBRegressor
        if exog is None:
            raise ValueError("XGBoostForecaster requires an exogenous feature frame (exog).")
        self._feature_cols = list(exog.columns)
        X = exog.values.astype(float)
        y = series.values.astype(float)
        n = len(X)
        if n != len(y):
            n = min(n, len(y))
            X = X[:n]
            y = y[:n]
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._model.fit(X, y)
        self._last_row = exog.iloc[-1] if len(exog) else None
        return self

    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        if exog is None:
            # Recursive fallback: repeat the last known feature row (weak).
            if self._last_row is None:
                raise ValueError("exog required for multi-step XGBoost forecast.")
            exog = pd.DataFrame([self._last_row] * steps, columns=self._feature_cols)
        X = exog.values.astype(float)
        return np.asarray(self._model.predict(X), dtype=float)


def evaluate_model(fc: BaseForecaster, series: pd.Series,
                   train_end: pd.Timestamp) -> dict:
    """Fit on history up to ``train_end`` and evaluate on the immediate tail.

    Returns metrics plus in-sample and out-of-sample predictions.
    """
    train = series.loc[:train_end]
    test = series.loc[train_end:].iloc[1:]  # exclude the boundary point
    fc.fit(train)
    pred = fc.predict(len(test))
    metrics = _metrics(test.values, pred)
    metrics["n_test"] = int(len(test))
    return metrics
