"""Generate a deterministic offline sample price dataset.

Produces ``data/sample_prices.csv`` with synthetic-but-realistic geometric
random-walk price paths for the default universe.  Used for offline demos and
as a fallback when the stock API is unreachable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_prices.csv"

import sys
sys.path.insert(0, str(ROOT / "src"))
from stockts.universe import DEFAULT_UNIVERSE  # noqa: E402


def generate(n_days: int = 504, seed: int = 610) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-06-19"), periods=n_days)
    data = {}
    for i, ticker in enumerate(DEFAULT_UNIVERSE):
        start = float(rng.uniform(20, 500))
        mu = rng.normal(0.0004, 0.0003)      # small daily drift
        sigma = rng.uniform(0.012, 0.030)    # daily volatility
        shocks = rng.normal(mu, sigma, size=n_days)
        path = start * np.exp(np.cumsum(shocks))
        data[ticker] = np.round(path, 2)
    return pd.DataFrame(data, index=dates)


def main() -> None:
    df = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT)
    print(f"wrote {OUT} ({df.shape[0]} rows x {df.shape[1]} tickers)")


if __name__ == "__main__":
    main()
