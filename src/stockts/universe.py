"""The universe of tickers the dashboard tracks.

By default we track a broad set of large-cap US equities.  ``sp500_tickers``
attempts to pull the live S&P 500 constituent list from Wikipedia and falls
back to the bundled static list when there is no network access (for example
inside a locked-down CI sandbox).
"""

from __future__ import annotations

# A broad, stable set of large-cap US tickers used as the default universe and
# as the offline fallback for the S&P 500 fetch.
DEFAULT_UNIVERSE: list[str] = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "CSCO", "ACN", "INTC", "AMD", "QCOM", "TXN", "IBM", "NOW",
    "INTU", "AMAT", "MU", "ADI", "LRCX",
    # Communication / media
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI", "BX",
    "V", "MA", "PYPL",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "MDT", "CVS", "GILD",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "HD", "LOW",
    "TGT", "BKNG", "MDLZ", "CL", "EL",
    # Industrials / energy / materials
    "XOM", "CVX", "COP", "SLB", "CAT", "BA", "GE", "HON", "UPS", "RTX",
    "LMT", "DE", "MMM", "UNP", "LIN", "FCX", "NEE", "DUK",
    # Other notables
    "BRK-B", "F", "GM", "UBER", "ABNB", "SHOP", "SQ", "PLTR", "SNOW", "COIN",
]


def default_universe() -> list[str]:
    """Return the default tracking universe (a copy)."""
    return list(DEFAULT_UNIVERSE)


def sp500_tickers(timeout: int = 10) -> list[str]:
    """Return the current S&P 500 tickers, falling back to the bundled list.

    Tries to read the constituents table from Wikipedia.  Any failure
    (no network, parse error, missing dependency) falls back to
    :data:`DEFAULT_UNIVERSE` so callers always get a usable list.
    """
    try:  # pragma: no cover - network dependent
        import pandas as pd

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        symbols = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False)
        tickers = [s.strip().upper() for s in symbols if s.strip()]
        if tickers:
            return tickers
    except Exception:
        pass
    return default_universe()
