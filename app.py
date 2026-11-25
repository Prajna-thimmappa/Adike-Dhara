"""Adike Dhara - Streamlit dashboard for arecanut price tracking & forecasting.

Run with:

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from features import DATE_COL, build_features, load_processed
from evaluate import best_model, compare_models, DEFAULT_MODELS
from forecast import FORECAST_HORIZONS, forecast_variety_market

# ---------------------------------------------------------------------------
# Kannada label map (toggle for main headings)
# ---------------------------------------------------------------------------
LABELS_KN = {
    "app_title": "ಅಡಿಕೆ ಧಾರಣೆ",
    "subtitle": "ಆರೆಕೇ ಬೆಲೆ ಟ್ರ್ಯಾಕರ್ & ಮುನ್ಸೂಚನೆ",
    "tagline": "ಇಂದಿನ ದರ ತಿಳಿದು ನಾಳೆಯ ಮಾರಾಟ ಯೋಜಿಸಿ",
    "latest_price": "ಇತ್ತೀಚಿನ ಬೆಲೆ",
    "history": "ಇತಿಹಾಸ ಬೆಲೆ ಚಾರ್ಟ್",
    "forecast": "ಮுன್ಸೂಚನೆ",
    "model_comparison": "ಮოდೆಲ್ ಹೋಲಿಕೆ",
    "download_csv": "CSV ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
    "footer": "ಮுன್ಸೂಚನೆಗಳು ಅಂದಾಜು ಮಾತ್ರ, ಹಣಕಾಸು ಸಲಹೆ ಅಲ್ಲ.",
}
LABELS_EN = {
    "app_title": "Adike Dhara (ಅಡಿಕೆ ಧಾರಣೆ)",
    "subtitle": "Areca Price Tracker & Forecast",
    "tagline": "Know today's rate. Plan tomorrow's sale.",
    "latest_price": "Latest Price",
    "history": "Historical Price Chart",
    "forecast": "Forecast Chart",
    "model_comparison": "Model Comparison",
    "download_csv": "Download CSV",
    "footer": "Forecasts are estimates, not financial advice. Real prices can change suddenly due to policy and market events.",
}


def _labels(use_kannada: bool) -> dict:
    return LABELS_KN if use_kannada else LABELS_EN


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_variety_csv(variety: str) -> Optional[pd.DataFrame]:
    path = Path("data/processed") / f"{variety}_daily.csv"
    if not path.exists():
        return None
    return load_processed(path)


@st.cache_data(show_spinner=False)
def _series_for(df: pd.DataFrame, market: str, variety: str) -> pd.Series:
    sub = df[(df["market"] == market) & (df["variety"] == variety)].copy()
    sub = sub.sort_values(DATE_COL)
    return sub.set_index(DATE_COL)["modal_price"].dropna().astype(float)


@st.cache_data(show_spinner=False)
def _eval_table() -> Optional[pd.DataFrame]:
    path = Path("data/processed/evaluation.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _history_chart(df: pd.DataFrame, market: str, variety: str, labels: dict) -> go.Figure:
    sub = df[(df["market"] == market) & (df["variety"] == variety)].copy()
    sub = sub.sort_values(DATE_COL)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub[DATE_COL],
            y=sub["modal_price"],
            mode="lines",
            name="Modal price",
            line=dict(color="#4C8B3A", width=2),
        )
    )
    fig.update_layout(
        title=labels["history"],
        xaxis_title="Date",
        yaxis_title="Modal price (₹/quintal)",
        hovermode="x unified",
        template="simple_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(step="all", label="All"),
        ])
    ))
    return fig


def _forecast_chart(fc_df: pd.DataFrame, labels: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fc_df["date"],
            y=fc_df["forecast"],
            mode="lines",
            name="Forecast",
            line=dict(color="#B5892B", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(fc_df["date"]) + list(fc_df["date"][::-1]),
            y=list(fc_df["upper"]) + list(fc_df["lower"][::-1]),
            fill="toself",
            fillcolor="rgba(181, 137, 43, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% interval",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=labels["forecast"],
        xaxis_title="Date",
        yaxis_title="Forecast price (₹/quintal)",
        hovermode="x unified",
        template="simple_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Adike Dhara", page_icon="🌴", layout="wide")

    use_kannada = st.sidebar.toggle("ಕನ್ನಡ / Kannada", value=False)
    labels = _labels(use_kannada)

    st.title(labels["app_title"])
    st.caption(labels["subtitle"])
    st.write(labels["tagline"])
    st.divider()

    # Load processed data
    processed_dir = Path("data/processed")
    csv_files = sorted(processed_dir.glob("*_daily.csv")) if processed_dir.exists() else []
    if not csv_files:
        st.error("No processed data found. Run `python -m src.clean data/raw/sample_arecanut.csv` first.")
        st.stop()

    varieties = sorted({p.stem.replace("_daily", "") for p in csv_files})
    variety = st.sidebar.selectbox("Variety", varieties, index=0)

    df = _load_variety_csv(variety)
    if df is None or df.empty:
        st.error(f"No data for variety: {variety}")
        st.stop()

    markets = sorted(df["market"].dropna().unique())
    market = st.sidebar.selectbox("Market", markets, index=0)

    series = _series_for(df, market, variety)
    if series.empty:
        st.error("No price series for this market/variety.")
        st.stop()

    # Latest price card
    latest = series.dropna()
    latest_date = latest.index[-1]
    latest_price = float(latest.iloc[-1])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(labels["latest_price"], f"₹{latest_price:,.2f}", str(latest_date.date()))
    if len(latest) >= 2:
        change_1d = latest_price - float(latest.iloc[-2])
        with col2:
            st.metric("Change vs yesterday", f"₹{change_1d:+.2f}")
    if len(latest) >= 8:
        change_7d = latest_price - float(latest.iloc[-8])
        with col3:
            st.metric("Change vs last week", f"₹{change_7d:+.2f}")

    st.divider()

    # Historical chart
    st.subheader(labels["history"])
    st.plotly_chart(_history_chart(df, market, variety, labels), use_container_width=True)

    # Forecast section
    st.subheader(labels["forecast"])
    colf1, colf2 = st.columns([3, 1])
    with colf2:
        horizon = st.selectbox("Horizon (days)", list(FORECAST_HORIZONS), index=0)
        run_forecast = st.button("Generate forecast")
    raw_path = Path("data/raw/sample_arecanut.csv")
    if run_forecast:
        with st.spinner("Training models and forecasting..."):
            try:
                fc_df = forecast_variety_market(
                    processed_dir / f"{variety}_daily.csv",
                    raw_path,
                    variety,
                    market,
                    horizons=(horizon,),
                )
                fc_df = fc_df[fc_df["date"] <= fc_df["date"].max()].head(horizon)
                st.plotly_chart(_forecast_chart(fc_df, labels), use_container_width=True)
                st.dataframe(fc_df[["date", "forecast", "lower", "upper", "model"]],
                             use_container_width=True, hide_index=True)
            except Exception as exc:  # pragma: no cover - UI safety net
                st.error(f"Forecast failed: {exc}")

    # Model comparison
    st.divider()
    st.subheader(labels["model_comparison"])
    eval_df = _eval_table()
    if eval_df is not None and not eval_df.empty:
        show_cols = [c for c in ("model", "mae", "rmse", "mape", "usable") if c in eval_df.columns]
        st.dataframe(eval_df[show_cols], use_container_width=True, hide_index=True)
        best = best_model(eval_df)
        if best:
            st.success(f"Best usable model: **{best}**")
        else:
            st.info("No model beats both baselines yet.")
    else:
        st.info("Run evaluation (`python -m src.evaluate`) to see the model comparison table.")

    # CSV download
    st.divider()
    st.subheader(labels["download_csv"])
    out_cols = [c for c in df.columns if c not in ("missing_day", "outlier")]
    csv_bytes = df[out_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {variety} daily prices (CSV)",
        data=csv_bytes,
        file_name=f"{variety}_daily.csv",
        mime="text/csv",
    )

    st.divider()
    st.caption(labels["footer"])


if __name__ == "__main__":
    main()
