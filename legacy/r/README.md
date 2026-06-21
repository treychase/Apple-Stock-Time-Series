# Legacy R / Quarto analysis

These files are the original STA 542 final project: a Bayesian time-series
analysis of **AAPL** stock written in R / Quarto (`.qmd`). They implement a
Gibbs-sampled Dynamic Linear Model, a Hidden Markov Model with AR(1) states,
and an ARIMA model, with a 10-year forecast comparison.

The project has since been rewritten in Python (see the repository root) to:

- track an entire **universe of stocks** via a stock API instead of just AAPL,
- serve an interactive **dashboard** with one-week Bayesian prediction intervals,
- rank the projected **top 5 / bottom 5** movers,
- add **CI/CD** unit-testing and data-validation pipelines.

These files are retained for provenance and reference only.
