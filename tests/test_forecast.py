import pandas as pd

from stockts.forecast import forecast_universe, rank_movers


def test_forecast_universe_table(history):
    ft = forecast_universe(history, horizon=5, n_iter=300, burn_in=80)
    assert set(ft.table["ticker"]) == set(history.columns)
    expected_cols = {"ticker", "last_price", "forecast", "lower", "upper",
                     "expected_return", "level"}
    assert expected_cols.issubset(ft.table.columns)
    # sorted descending by expected_return
    er = ft.table["expected_return"].to_numpy()
    assert (er[:-1] >= er[1:]).all()


def test_top_and_bottom(history):
    ft = forecast_universe(history, horizon=5, n_iter=300, burn_in=80)
    n = len(ft.table)
    assert len(ft.top5) == min(5, n)
    assert len(ft.bottom5) == min(5, n)
    # top has the highest expected return, bottom the lowest
    assert ft.top5.iloc[0]["expected_return"] == ft.table["expected_return"].max()
    assert ft.bottom5.iloc[0]["expected_return"] == ft.table["expected_return"].min()


def test_skips_short_series(history):
    h = history.copy()
    h["SHORT"] = pd.NA
    h.iloc[-5:, h.columns.get_loc("SHORT")] = [1.0, 2.0, 3.0, 4.0, 5.0]
    ft = forecast_universe(h, horizon=5, min_obs=30, n_iter=200, burn_in=50)
    assert "SHORT" not in set(ft.table["ticker"])


def test_empty_history_returns_empty():
    ft = forecast_universe(pd.DataFrame(), horizon=5)
    assert ft.table.empty
    assert ft.top5.empty and ft.bottom5.empty


def test_rank_movers_basic():
    df = pd.DataFrame({
        "ticker": list("ABCDEFG"),
        "expected_return": [0.1, -0.2, 0.3, 0.0, -0.5, 0.05, 0.2],
    })
    top, bottom = rank_movers(df, n=3)
    assert list(top["ticker"]) == ["C", "G", "A"]
    assert list(bottom["ticker"]) == ["E", "B", "D"]
