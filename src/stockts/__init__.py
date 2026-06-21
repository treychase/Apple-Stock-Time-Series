"""stockts - Bayesian time-series forecasting across a universe of stocks.

Public API:
    predict_interval   - one-week Bayesian prediction interval for one series
    forecast_universe  - forecast every ticker and rank top/bottom movers
    validate_history   - data-quality checks used in CI
"""

from .bayesian import GibbsResult, PredictionInterval, gibbs_local_level, predict_interval
from .forecast import ForecastTable, forecast_universe, rank_movers
from .validation import ValidationReport, assert_valid, validate_history

__version__ = "1.0.0"

__all__ = [
    "GibbsResult",
    "PredictionInterval",
    "gibbs_local_level",
    "predict_interval",
    "ForecastTable",
    "forecast_universe",
    "rank_movers",
    "ValidationReport",
    "assert_valid",
    "validate_history",
    "__version__",
]
