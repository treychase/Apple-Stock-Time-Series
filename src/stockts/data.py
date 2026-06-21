"""Market-data access layer.

Wraps the `yfinance` stock API to download adjusted-close price history for one
or many tickers.  All network access is funnelled through this module so the
rest of the codebase can be unit-tested by monkeypatching :func:`download_history`.

When the API is unreachable (e.g. a sandbox with no egress) callers can fall
back to :func:`load_sample_prices`, a small bundled CSV used for offline demos
and tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

__all__ = [
    "download_history",
    "latest_prices",
    "load_sample_prices",
    "SAMPLE_PATH",
]

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_prices.csv"


def download_history(
    tickers: list[str] | str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted-close price history from the stock API.

    Returns a wide DataFrame indexed by date with one column per ticker.
    Tickers that return no data are dropped.
    """
    import yfinance as yf

    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    # Normalise the (possibly MultiIndex) columns down to a per-ticker close frame.
    if isinstance(raw.columns, pd.MultiIndex):
        field = "Close" if "Close" in raw.columns.get_level_values(0) else raw.columns.levels[0][0]
        close = raw[field].copy()
    else:
        # Single ticker -> flat columns; take Close.
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.dropna(axis=1, how="all").sort_index()
    close.index = pd.to_datetime(close.index)
    return close


def latest_prices(history: pd.DataFrame) -> pd.Series:
    """Most recent non-NaN price for each ticker column."""
    return history.ffill().iloc[-1]


def load_sample_prices() -> pd.DataFrame:
    """Load the bundled offline sample price history."""
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"sample data not found at {SAMPLE_PATH}; run scripts/generate_sample_data.py"
        )
    df = pd.read_csv(SAMPLE_PATH, index_col=0, parse_dates=True)
    return df.sort_index()


def get_history(
    tickers: list[str] | str,
    period: str = "2y",
    use_sample: bool | None = None,
) -> pd.DataFrame:
    """Convenience loader: live API unless ``use_sample`` (or env) forces sample.

    Set the environment variable ``STOCKTS_USE_SAMPLE=1`` to force the bundled
    sample data — handy for offline demos and CI.
    """
    if use_sample is None:
        use_sample = os.environ.get("STOCKTS_USE_SAMPLE", "0") == "1"
    if use_sample:
        df = load_sample_prices()
        if isinstance(tickers, str):
            tickers = [tickers]
        cols = [t.upper() for t in tickers if t.upper() in df.columns]
        return df[cols] if cols else df
    try:
        df = download_history(tickers, period=period)
        if not df.empty:
            return df
    except Exception:
        pass
    return load_sample_prices()
