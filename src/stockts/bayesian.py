"""Bayesian local-level dynamic linear model for stock price forecasting.

This is a NumPy port of the Gibbs-sampled local-level DLM that the original
project implemented in R.  The model is fit to *log* prices so that the
forecast distribution, once exponentiated, is strictly positive and
right-skewed in the way prices tend to be.

Model (on the log price ``y_t``):

    observation:  y_t   = theta_t + v_t,   v_t ~ N(0, V)
    state:        theta_t = theta_{t-1} + w_t,   w_t ~ N(0, W)

Priors:

    theta_0 ~ N(m0, C0)
    V ~ Inverse-Gamma(a_V, b_V)
    W ~ Inverse-Gamma(a_W, b_W)

The Gibbs sampler alternates a forward-filter/backward-sample (FFBS) step for
the latent level with conjugate Inverse-Gamma draws for the two variances.

The headline output is :func:`predict_interval`, which returns a Bayesian
posterior-predictive interval for the price ``horizon`` trading days ahead
(one week == 5 trading days by default).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

__all__ = ["GibbsResult", "PredictionInterval", "gibbs_local_level", "predict_interval"]


@dataclass
class GibbsResult:
    """Posterior draws produced by :func:`gibbs_local_level`."""

    V: np.ndarray  # observation variance draws, shape (n_keep,)
    W: np.ndarray  # state (evolution) variance draws, shape (n_keep,)
    theta_last: np.ndarray  # draws of the final latent level, shape (n_keep,)


@dataclass
class PredictionInterval:
    """A posterior-predictive forecast for a single ticker."""

    horizon: int
    last_price: float
    point: float  # posterior-predictive median price at the horizon
    lower: float  # lower credible bound
    upper: float  # upper credible bound
    level: float  # credible level, e.g. 0.95
    expected_return: float  # (point - last_price) / last_price

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ffbs(y: np.ndarray, V: float, W: float, m0: float, C0: float,
          rng: np.random.Generator) -> np.ndarray:
    """Forward-filter, backward-sample the latent level path.

    Returns a single sampled path ``theta`` of the same length as ``y``.
    """
    n = y.shape[0]
    m = np.empty(n)  # filtered mean
    C = np.empty(n)  # filtered variance

    a_prev, R_prev = m0, C0
    for t in range(n):
        if t > 0:
            a_prev = m[t - 1]
            R_prev = C[t - 1] + W
        else:
            R_prev = C0 + W
        Q = R_prev + V
        A = R_prev / Q
        m[t] = a_prev + A * (y[t] - a_prev)
        C[t] = R_prev - A * A * Q

    theta = np.empty(n)
    theta[-1] = rng.normal(m[-1], np.sqrt(max(C[-1], 1e-12)))
    for t in range(n - 2, -1, -1):
        h = C[t] / (C[t] + W)
        mean = h * theta[t + 1] + (1.0 - h) * m[t]
        var = max(h * C[t], 1e-12)
        theta[t] = rng.normal(mean, np.sqrt(var))
    return theta


def gibbs_local_level(
    y: np.ndarray,
    n_iter: int = 2000,
    burn_in: int = 500,
    thin: int = 1,
    seed: int | None = 610,
) -> GibbsResult:
    """Fit the local-level DLM with a Gibbs sampler.

    Parameters
    ----------
    y:
        1-D array of observations (log prices).
    n_iter, burn_in, thin:
        Total iterations, warm-up to discard, and thinning factor.
    seed:
        Seed for the random number generator (reproducibility).
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 5:
        raise ValueError("need at least 5 observations to fit the DLM")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values")

    rng = np.random.default_rng(seed)
    n = y.size

    dy = np.diff(y)
    base_var = float(np.var(dy)) if np.var(dy) > 0 else 1e-6

    # Priors
    m0 = float(y[0])
    C0 = 1e3
    a_V, b_V = 3.0, 2.0 * base_var
    a_W, b_W = 3.0, 0.5 * base_var

    # Initialise
    V = base_var
    W = 0.1 * base_var

    keep = []
    for it in range(n_iter):
        theta = _ffbs(y, V, W, m0, C0, rng)

        ss_V = float(np.sum((y - theta) ** 2))
        V = 1.0 / rng.gamma(a_V + n / 2.0, 1.0 / (b_V + ss_V / 2.0))

        ss_W = float(np.sum(np.diff(theta) ** 2))
        W = 1.0 / rng.gamma(a_W + (n - 1) / 2.0, 1.0 / (b_W + ss_W / 2.0))

        if it >= burn_in and (it - burn_in) % thin == 0:
            keep.append((V, W, theta[-1]))

    arr = np.array(keep)
    return GibbsResult(V=arr[:, 0], W=arr[:, 1], theta_last=arr[:, 2])


def predict_interval(
    prices: np.ndarray,
    horizon: int = 5,
    level: float = 0.95,
    n_iter: int = 2000,
    burn_in: int = 500,
    seed: int | None = 610,
) -> PredictionInterval:
    """Posterior-predictive price forecast ``horizon`` trading days ahead.

    Prices are modelled on the log scale; the returned bounds are prices.

    For each retained posterior draw of ``(V, W, theta_last)`` we simulate the
    random-walk level forward ``horizon`` steps and add observation noise,
    yielding a sample from the posterior predictive distribution of the future
    (log) price.  The reported point/lower/upper are the requested quantiles of
    that distribution, mapped back to price space.
    """
    prices = np.asarray(prices, dtype=float).ravel()
    prices = prices[np.isfinite(prices)]
    if prices.size < 5:
        raise ValueError("need at least 5 finite prices")
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive (log model)")

    y = np.log(prices)
    res = gibbs_local_level(y, n_iter=n_iter, burn_in=burn_in, seed=seed)

    rng = np.random.default_rng(seed)
    n_keep = res.theta_last.shape[0]

    # Simulate the level forward: theta_{t+h} = theta_t + sum w, w~N(0,W);
    # observed log price adds one more N(0,V).  Cumulative state variance over
    # ``horizon`` steps is horizon * W.
    state_noise = rng.normal(size=n_keep) * np.sqrt(horizon * res.W)
    obs_noise = rng.normal(size=n_keep) * np.sqrt(res.V)
    future_log = res.theta_last + state_noise + obs_noise
    future_price = np.exp(future_log)

    alpha = 1.0 - level
    lower = float(np.quantile(future_price, alpha / 2.0))
    upper = float(np.quantile(future_price, 1.0 - alpha / 2.0))
    point = float(np.quantile(future_price, 0.5))
    last_price = float(prices[-1])

    return PredictionInterval(
        horizon=horizon,
        last_price=last_price,
        point=point,
        lower=lower,
        upper=upper,
        level=level,
        expected_return=(point - last_price) / last_price,
    )
