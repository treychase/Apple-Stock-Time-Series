# 📈 Bayesian Stock Time-Series Forecasting

[![tests](https://github.com/treychase/apple-stock-time-series/actions/workflows/tests.yml/badge.svg)](https://github.com/treychase/apple-stock-time-series/actions/workflows/tests.yml)
[![data-validation](https://github.com/treychase/apple-stock-time-series/actions/workflows/data-validation.yml/badge.svg)](https://github.com/treychase/apple-stock-time-series/actions/workflows/data-validation.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A Bayesian time-series engine and interactive dashboard that tracks an entire
**universe of stocks** from a market-data API and produces, for each one, a
**posterior-predictive interval** for the price over the **next trading week**.
The dashboard surfaces the projected **top 5** and **bottom 5** movers.

> This project began as an R/Quarto course project analysing a single stock
> (AAPL). It has been rewritten in **Python** and generalised to a full
> universe with a dashboard and CI/CD. The original R analysis is preserved
> under [`legacy/r/`](legacy/r/).

---

## Highlights

- **Bayesian model** — local-level Dynamic Linear Model fit on log prices with
  a Gibbs sampler (forward-filter/backward-sample for the latent level,
  conjugate Inverse-Gamma updates for the variances). A NumPy port of the
  original R sampler.
- **Universe-wide forecasts** — fit every ticker and rank by expected one-week
  return to get the projected top 5 / bottom 5.
- **Interactive dashboard** — Streamlit + Plotly: per-ticker history with its
  credible band, plus the movers leaderboard.
- **CI/CD** — GitHub Actions for unit tests (multi-version matrix + coverage)
  and market-data validation (also on a weekday schedule).
- **Offline-friendly** — every network call funnels through one module; a
  bundled sample dataset (`STOCKTS_USE_SAMPLE=1`) powers demos and tests with
  no API access.

## The model

On the log price `y_t`:

```
observation:  y_t     = θ_t + v_t,        v_t ~ N(0, V)
state:        θ_t     = θ_{t-1} + w_t,    w_t ~ N(0, W)
priors:       θ_0 ~ N(m0, C0),  V ~ IG(a_V, b_V),  W ~ IG(a_W, b_W)
```

The one-week forecast is the posterior predictive distribution of the price
`horizon` (default 5 trading) days ahead: for each retained posterior draw of
`(V, W, θ_last)` we propagate the random walk forward and add observation
noise, then read off the requested quantiles and map back to price space.

## Project layout

```
src/stockts/
  bayesian.py     # Gibbs-sampled local-level DLM + predict_interval
  forecast.py     # forecast_universe + top5/bottom5 ranking
  data.py         # yfinance access layer (+ offline sample fallback)
  universe.py     # tracked tickers (live S&P 500 fetch or bundled list)
  validation.py   # data-quality checks used in CI
app/dashboard.py  # Streamlit dashboard
scripts/          # generate_sample_data / run_forecasts / validate_data
tests/            # pytest suite
data/             # bundled sample_prices.csv
.github/workflows # tests.yml, data-validation.yml
legacy/r/         # original R/Quarto analysis
```

## Quickstart

```bash
pip install -r requirements.txt

# Run the dashboard (offline demo with bundled data)
STOCKTS_USE_SAMPLE=1 streamlit run app/dashboard.py

# ...or live from the stock API
streamlit run app/dashboard.py

# Forecast the whole universe to data/forecasts.json
python scripts/run_forecasts.py            # live
STOCKTS_USE_SAMPLE=1 python scripts/run_forecasts.py   # offline
```

Programmatic use:

```python
from stockts import predict_interval, forecast_universe
from stockts.data import get_history
from stockts.universe import default_universe

history = get_history(default_universe(), period="2y")
result = forecast_universe(history, horizon=5, level=0.95)
print(result.top5)      # projected best 5 over the next week
print(result.bottom5)   # projected worst 5
```

## Development

```bash
pip install -r requirements.txt
pytest --cov=stockts          # run the test suite with coverage
python scripts/validate_data.py   # run the data-validation checks
```

## License

[MIT](LICENSE) © Trey Chase, Tully Cannon
