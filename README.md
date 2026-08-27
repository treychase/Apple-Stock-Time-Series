# 📈 Bayesian Stock Time-Series Forecasting

[![tests](https://github.com/treychase/apple-stock-time-series/actions/workflows/tests.yml/badge.svg)](https://github.com/treychase/apple-stock-time-series/actions/workflows/tests.yml)
[![data-validation](https://github.com/treychase/apple-stock-time-series/actions/workflows/data-validation.yml/badge.svg)](https://github.com/treychase/apple-stock-time-series/actions/workflows/data-validation.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A Bayesian time-series engine and interactive dashboard that tracks an entire
**universe of stocks** from a market-data API and produces, for each one, a
**posterior-predictive interval** for the price over the **next trading week**.
The dashboard surfaces the projected **top 5** and **bottom 5** movers.

The forecasting model is a **two-state hidden Markov model with a dynamic
linear model in each state**: the same local-level DLM, but with its variances
switching between a calm regime and a volatile one under a hidden Markov chain.

> This project began as an R/Quarto course project analysing a single stock
> (AAPL). It has been rewritten in **Python** and generalised to a full
> universe with a dashboard and CI/CD. The original R analysis is preserved
> under [`legacy/r/`](legacy/r/).

---

## Highlights

- **Bayesian model** — a two-state Markov-switching local-level DLM fit on log
  prices with a Gibbs sampler: FFBS for the latent level with the variances the
  current regime implies, FFBS over the discrete chain for the regime path,
  conjugate Inverse-Gamma updates for each regime's variances, and Dirichlet
  updates for the transition matrix. The single-regime sampler it grew out of —
  a NumPy port of the original R code — is still there as `predict_interval`.
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

A hidden two-state Markov chain choosing between two DLMs, on the log price
`y_t`:

```
observation:  y_t     = θ_t + v_t,        v_t ~ N(0, V[S_t])
state:        θ_t     = θ_{t-1} + w_t,    w_t ~ N(0, W[S_t])
regime:       S_t | S_{t-1} ~ Categorical(P[S_{t-1}, :]),   S_t ∈ {0, 1}
priors:       θ_0 ~ N(m0, C0),  V[s], W[s] ~ IG(a, b),  P[s, :] ~ Dirichlet(α)
```

Both regimes are the same local-level DLM; what differs is how much the level
is allowed to move and how noisy the observation is. **State 0 is the calm
regime and state 1 the volatile one** — the sampler relabels whenever a draw
violates `W[0] ≤ W[1]`, because without that constraint the two states are
exchangeable and can swap mid-run, smearing every regime-specific summary into
the average of the two.

The forecast propagates the regime chain and the level forward together, so the
predictive distribution is a **mixture over regime paths** rather than a single
Gaussian: a stock that is calm today but sits in a jumpy regime carries a wider
band than one that is calm and sticky. `predict_fan` returns every horizon from
one day to `horizon` (default 7 trading days) out of a single fit, along with
the fitted level, the smoothed probability of the volatile regime at every
point in the history, each regime's daily volatility, and how persistent each
regime is.

The single-regime model is still available and still tested: pass
`model="dlm"` to `forecast_universe`, or call `predict_interval` directly.

## Project layout

```
src/stockts/
  bayesian.py     # Gibbs-sampled local-level DLM + predict_interval
  hmm.py          # two-state Markov-switching DLM + predict_fan (the default)
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
