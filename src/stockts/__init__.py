"""stockts - Bayesian time-series forecasting across a universe of stocks.

Public API:
    predict_fan        - two-state HMM + DLM forecast, every horizon in one fit
    predict_interval   - single-regime DLM interval for one series
    forecast_universe  - forecast every ticker and rank top/bottom movers
    validate_history   - data-quality checks used in CI
"""

from .bayesian import GibbsResult, PredictionInterval, gibbs_local_level, predict_interval
from .forecast import ForecastTable, forecast_universe, rank_movers
from .hmm import (SwitchingForecast, SwitchingResult, gibbs_switching_local_level,
                  predict_fan, predict_interval_switching)
from .validation import ValidationReport, assert_valid, validate_history

__version__ = "1.0.0"

__all__ = [
    "GibbsResult",
    "PredictionInterval",
    "gibbs_local_level",
    "predict_interval",
    "SwitchingForecast",
    "SwitchingResult",
    "gibbs_switching_local_level",
    "predict_fan",
    "predict_interval_switching",
    "ForecastTable",
    "forecast_universe",
    "rank_movers",
    "ValidationReport",
    "assert_valid",
    "validate_history",
    "__version__",
]
