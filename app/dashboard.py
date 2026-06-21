"""Streamlit dashboard: Bayesian one-week price forecasts across stocks.

Run with::

    STOCKTS_USE_SAMPLE=1 streamlit run app/dashboard.py   # offline demo
    streamlit run app/dashboard.py                         # live API

Features:
  * pick any tracked ticker and see its price history with a Bayesian
    posterior-predictive interval for the next week;
  * a leaderboard of the projected top 5 and bottom 5 movers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockts.bayesian import predict_interval  # noqa: E402
from stockts.data import get_history  # noqa: E402
from stockts.forecast import forecast_universe  # noqa: E402
from stockts.universe import default_universe  # noqa: E402

st.set_page_config(page_title="Bayesian Stock Forecasts", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return get_history(list(tickers), period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def run_universe(history: pd.DataFrame, horizon: int, level: float):
    ft = forecast_universe(history, horizon=horizon, level=level)
    return ft.table, ft.top5, ft.bottom5


def main() -> None:
    st.title("📈 Bayesian Stock Forecasts")
    st.caption(
        "Local-level dynamic linear model (Gibbs sampler) — posterior-predictive "
        "price interval for the next trading week."
    )

    with st.sidebar:
        st.header("Settings")
        horizon = st.slider("Horizon (trading days)", 1, 10, 5)
        level = st.select_slider("Credible level", [0.80, 0.90, 0.95, 0.99], value=0.95)
        period = st.selectbox("History window", ["6mo", "1y", "2y", "5y"], index=2)
        st.markdown("---")
        st.caption("Set `STOCKTS_USE_SAMPLE=1` to run fully offline.")

    tickers = tuple(default_universe())
    with st.spinner("Loading price history..."):
        history = load_history(tickers, period)

    if history.empty:
        st.error("No price data available.")
        return

    table, top5, bottom5 = run_universe(history, horizon, level)

    # --- Leaderboard ---
    st.subheader(f"Projected movers — next {horizon} trading days")
    c1, c2 = st.columns(2)
    fmt = {"expected_return": "{:+.2%}", "last_price": "{:.2f}",
           "forecast": "{:.2f}", "lower": "{:.2f}", "upper": "{:.2f}"}
    cols = ["ticker", "last_price", "forecast", "lower", "upper", "expected_return"]
    with c1:
        st.markdown("**Top 5 projected gainers**")
        st.dataframe(top5[cols].style.format(fmt), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Bottom 5 projected**")
        st.dataframe(bottom5[cols].style.format(fmt), hide_index=True, use_container_width=True)

    st.markdown("---")

    # --- Single-ticker detail ---
    st.subheader("Ticker detail")
    choices = list(table["ticker"]) if not table.empty else list(history.columns)
    ticker = st.selectbox("Ticker", choices, index=0)

    series = history[ticker].dropna()
    pi = predict_interval(series.to_numpy(), horizon=horizon, level=level)

    last_date = series.index[-1]
    future_dates = pd.bdate_range(last_date, periods=horizon + 1)[1:]

    fig = go.Figure()
    tail = series.tail(180)
    fig.add_trace(go.Scatter(x=tail.index, y=tail.values, name="History",
                             line=dict(color="#1f77b4")))
    # Interval band at the horizon, drawn from the last point.
    band_x = [last_date, future_dates[-1], future_dates[-1], last_date]
    band_y = [pi.last_price, pi.upper, pi.lower, pi.last_price]
    fig.add_trace(go.Scatter(x=band_x, y=band_y, fill="toself",
                             fillcolor="rgba(31,119,180,0.18)", line=dict(width=0),
                             name=f"{int(level*100)}% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[last_date, future_dates[-1]],
                             y=[pi.last_price, pi.point],
                             name="Forecast (median)",
                             line=dict(color="#ff7f0e", dash="dash")))
    fig.update_layout(height=460, hovermode="x unified",
                      yaxis_title="Price ($)", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last price", f"${pi.last_price:,.2f}")
    m2.metric(f"{horizon}-day forecast", f"${pi.point:,.2f}", f"{pi.expected_return:+.2%}")
    m3.metric("Lower bound", f"${pi.lower:,.2f}")
    m4.metric("Upper bound", f"${pi.upper:,.2f}")


if __name__ == "__main__":
    main()
