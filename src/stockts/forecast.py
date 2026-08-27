"""Run the Bayesian model across a whole universe and rank the results.

The headline product is a table of one-week-ahead price forecasts with credible
intervals for every tracked ticker, plus the projected ``top 5`` and
``bottom 5`` names by expected return.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .bayesian import PredictionInterval, predict_interval
from .hmm import predict_interval_switching

__all__ = ["ForecastTable", "forecast_universe", "rank_movers"]


@dataclass
class ForecastTable:
    """All per-ticker forecasts plus the ranked movers."""

    table: pd.DataFrame  # one row per ticker, sorted by expected_return desc
    top5: pd.DataFrame
    bottom5: pd.DataFrame
    horizon: int
    level: float


def _row(ticker: str, pi: PredictionInterval) -> dict:
    return {
        "ticker": ticker,
        "last_price": round(pi.last_price, 4),
        "forecast": round(pi.point, 4),
        "lower": round(pi.lower, 4),
        "upper": round(pi.upper, 4),
        "expected_return": pi.expected_return,
        "level": pi.level,
    }


def forecast_universe(
    history: pd.DataFrame,
    horizon: int = 7,
    level: float = 0.95,
    min_obs: int = 30,
    n_iter: int = 1200,
    burn_in: int = 400,
    seed: int | None = 610,
    model: str = "hmm",
) -> ForecastTable:
    """Fit the Bayesian model to every column of ``history``.

    Parameters
    ----------
    history:
        Wide price frame (date index, one column per ticker).
    horizon:
        Forecast horizon in trading days (5 == one week).
    level:
        Credible level for the prediction interval.
    min_obs:
        Skip tickers with fewer than this many finite observations.
    model:
        ``"hmm"`` (default) fits the two-state Markov-switching local-level
        model, a DLM in each regime; ``"dlm"`` fits the single-regime local
        level model kept in :mod:`stockts.bayesian`.
    """
    if model not in {"hmm", "dlm"}:
        raise ValueError("model must be 'hmm' or 'dlm'")
    fit = predict_interval_switching if model == "hmm" else predict_interval
    rows: list[dict] = []
    for ticker in history.columns:
        series = history[ticker].dropna()
        if series.size < min_obs or (series <= 0).any():
            continue
        try:
            pi = fit(
                series.to_numpy(),
                horizon=horizon,
                level=level,
                n_iter=n_iter,
                burn_in=burn_in,
                seed=seed,
            )
        except Exception:
            continue
        rows.append(_row(str(ticker), pi))

    if not rows:
        empty = pd.DataFrame(
            columns=["ticker", "last_price", "forecast", "lower", "upper",
                     "expected_return", "level"]
        )
        return ForecastTable(empty, empty, empty, horizon, level)

    table = (
        pd.DataFrame(rows)
        .sort_values("expected_return", ascending=False)
        .reset_index(drop=True)
    )
    top5, bottom5 = rank_movers(table)
    return ForecastTable(table=table, top5=top5, bottom5=bottom5,
                         horizon=horizon, level=level)


def rank_movers(table: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(top_n, bottom_n)`` tickers by expected return."""
    ordered = table.sort_values("expected_return", ascending=False)
    top = ordered.head(n).reset_index(drop=True)
    bottom = ordered.tail(n).iloc[::-1].reset_index(drop=True)
    return top, bottom
