import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def gbm_prices(rng):
    """A single geometric-random-walk price series."""
    shocks = rng.normal(0.0005, 0.02, size=300)
    return 100.0 * np.exp(np.cumsum(shocks))


@pytest.fixture
def history(rng):
    """A small wide price frame with a few tickers."""
    dates = pd.bdate_range("2024-01-01", periods=200)
    data = {}
    for i, t in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        shocks = rng.normal(0.0004 + i * 0.0001, 0.015, size=200)
        data[t] = 50.0 * (i + 1) * np.exp(np.cumsum(shocks))
    return pd.DataFrame(data, index=dates)
