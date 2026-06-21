import pandas as pd
import pytest

from stockts import data
from stockts.universe import DEFAULT_UNIVERSE, default_universe, sp500_tickers


def test_load_sample_prices():
    df = data.load_sample_prices()
    assert not df.empty
    assert "AAPL" in df.columns
    assert df.index.is_monotonic_increasing


def test_latest_prices():
    df = data.load_sample_prices()
    latest = data.latest_prices(df)
    assert isinstance(latest, pd.Series)
    assert latest.notna().all()


def test_get_history_uses_sample_when_forced():
    df = data.get_history(["AAPL", "MSFT"], use_sample=True)
    assert list(df.columns) == ["AAPL", "MSFT"]


def test_get_history_falls_back_on_api_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(data, "download_history", boom)
    df = data.get_history(["AAPL"], use_sample=False)
    assert not df.empty  # fell back to sample data


def test_download_history_parses_multiindex(monkeypatch):
    # Simulate a yfinance multiindex frame with a Close field.
    idx = pd.bdate_range("2024-01-01", periods=5)
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["AAPL", "MSFT"]])
    fake = pd.DataFrame(1.0, index=idx, columns=cols)

    class FakeYF:
        @staticmethod
        def download(*args, **kwargs):
            return fake

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)
    out = data.download_history(["AAPL", "MSFT"])
    assert list(out.columns) == ["AAPL", "MSFT"]
    assert len(out) == 5


def test_default_universe_is_copy():
    u = default_universe()
    u.append("ZZZZ")
    assert "ZZZZ" not in DEFAULT_UNIVERSE


def test_sp500_fallback(monkeypatch):
    # Force the Wikipedia read to fail -> fallback to default universe.
    import stockts.universe as uni

    def boom(*a, **k):
        raise RuntimeError("no network")

    import pandas as _pd
    monkeypatch.setattr(_pd, "read_html", boom)
    assert sp500_tickers() == DEFAULT_UNIVERSE
