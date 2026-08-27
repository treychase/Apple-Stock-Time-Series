"""Markov-switching local-level model: a two-state HMM with a DLM in each state.

The plain local-level DLM in :mod:`stockts.bayesian` carries one observation
variance and one evolution variance for the whole history, which asks a stock to
be equally volatile in a quiet June and a panicked October.  This module keeps
the same local-level structure but lets a hidden two-state Markov chain switch
between two sets of variances, so the model can say *which* regime a period was
in and price the forecast accordingly.

Model (on the log price ``y_t``, with hidden regime ``S_t`` in {0, 1}):

    observation:  y_t     = theta_t + v_t,        v_t ~ N(0, V[S_t])
    state:        theta_t = theta_{t-1} + w_t,    w_t ~ N(0, W[S_t])
    regime:       S_t | S_{t-1} ~ Categorical(P[S_{t-1}, :])

Priors:

    theta_0 ~ N(m0, C0)
    V[s], W[s] ~ Inverse-Gamma(a, b)          independently per regime
    P[s, :]    ~ Dirichlet(alpha[s, :])       favouring persistence

Regimes are ordered by evolution variance, ``W[0] <= W[1]``, and relabelled
whenever a sweep violates that.  Without the constraint the two states are
exchangeable and the sampler is free to swap them mid-run, which would smear
every regime-specific summary into the average of the two.  With it, **state 0
is the calm regime and state 1 the volatile one** in every draw.

The Gibbs sweep alternates four conditionals:

1. the level path ``theta``, by forward-filter/backward-sample with the
   time-varying variances the current regime path implies;
2. the regime path ``S``, by forward-filter/backward-sample over the discrete
   chain, where each regime's likelihood at ``t`` combines the observation
   density and the evolution density;
3. ``V[s]`` and ``W[s]``, conjugate inverse-gamma draws from the residuals
   belonging to each regime;
4. ``P``, conjugate Dirichlet draws from the transition counts.

:func:`predict_fan` is the headline output: it propagates the regime chain and
the level forward from the end of the sample, giving a posterior predictive
distribution at every horizon from one day to ``horizon`` days out.  Because the
chain can switch during those days, the forecast is a mixture over regime paths
rather than a single Gaussian, which is the point of fitting it this way.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np

from .bayesian import PredictionInterval

__all__ = [
    "SwitchingResult",
    "SwitchingForecast",
    "gibbs_switching_local_level",
    "predict_fan",
    "predict_interval_switching",
]

N_STATES = 2


@dataclass
class SwitchingResult:
    """Posterior draws from :func:`gibbs_switching_local_level`."""

    V: np.ndarray           # (n_keep, 2) observation variance by regime
    W: np.ndarray           # (n_keep, 2) evolution variance by regime
    P: np.ndarray           # (n_keep, 2, 2) transition matrix
    theta_last: np.ndarray  # (n_keep,) final latent level
    state_last: np.ndarray  # (n_keep,) final regime
    theta_mean: np.ndarray  # (n,) posterior mean level: the fitted values
    state_prob: np.ndarray  # (n,) posterior P(S_t = 1), the volatile regime


@dataclass
class SwitchingForecast:
    """Posterior predictive prices from one day out to the horizon."""

    horizon: int
    level: float
    last_price: float
    lower: list[float] = field(default_factory=list)
    point: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)
    fitted: list[float] = field(default_factory=list)      # in-sample level, prices
    state_prob: list[float] = field(default_factory=list)  # P(volatile) per day
    vol_daily: tuple[float, float] = (0.0, 0.0)   # sd of the level step, by regime
    persistence: tuple[float, float] = (0.0, 0.0)  # P[s, s], how sticky each regime is
    p_volatile_now: float = 0.0
    rmse: float = 0.0     # in-sample RMSE of the fitted level, in price units

    @property
    def expected_return(self) -> float:
        return (self.point[-1] - self.last_price) / self.last_price

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ffbs_varying(y: np.ndarray, V: np.ndarray, W: np.ndarray,
                  m0: float, C0: float, rng: np.random.Generator) -> np.ndarray:
    """FFBS for the level with a per-observation variance pair.

    ``V`` and ``W`` are length-n arrays holding the variance in force at each
    time, which is what makes this the switching version of the sampler in
    :mod:`stockts.bayesian`.
    """
    n = y.shape[0]
    m = np.empty(n)
    C = np.empty(n)
    R = np.empty(n)

    for t in range(n):
        a = m0 if t == 0 else m[t - 1]
        R[t] = (C0 if t == 0 else C[t - 1]) + W[t]
        Q = R[t] + V[t]
        A = R[t] / Q
        m[t] = a + A * (y[t] - a)
        C[t] = R[t] - A * A * Q

    theta = np.empty(n)
    theta[-1] = rng.normal(m[-1], np.sqrt(max(C[-1], 1e-12)))
    for t in range(n - 2, -1, -1):
        # the step from t to t+1 carries W[t+1]
        h = C[t] / (C[t] + W[t + 1])
        mean = h * theta[t + 1] + (1.0 - h) * m[t]
        var = max(h * C[t], 1e-12)
        theta[t] = rng.normal(mean, np.sqrt(var))
    return theta


def _sample_states(y: np.ndarray, theta: np.ndarray, V: np.ndarray, W: np.ndarray,
                   P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """FFBS over the discrete chain, given the level path and the variances."""
    n = y.shape[0]
    dtheta = np.empty(n)
    dtheta[0] = 0.0
    dtheta[1:] = np.diff(theta)

    # log p(y_t | theta_t, s) + log p(theta_t | theta_{t-1}, s)
    logp = np.empty((n, N_STATES))
    for s in range(N_STATES):
        obs = -0.5 * np.log(V[s]) - (y - theta) ** 2 / (2.0 * V[s])
        evo = -0.5 * np.log(W[s]) - dtheta ** 2 / (2.0 * W[s])
        evo[0] = 0.0            # no evolution step into the first observation
        logp[:, s] = obs + evo
    logp -= logp.max(axis=1, keepdims=True)
    lik = np.exp(logp)

    alpha = np.empty((n, N_STATES))
    stat = np.array([0.5, 0.5])
    alpha[0] = stat * lik[0]
    alpha[0] /= alpha[0].sum()
    for t in range(1, n):
        pred = alpha[t - 1] @ P
        alpha[t] = pred * lik[t]
        total = alpha[t].sum()
        alpha[t] = alpha[t] / total if total > 0 else stat

    states = np.empty(n, dtype=int)
    states[-1] = rng.random() > alpha[-1, 0]
    for t in range(n - 2, -1, -1):
        w = alpha[t] * P[:, states[t + 1]]
        w = w / w.sum() if w.sum() > 0 else stat
        states[t] = rng.random() > w[0]
    return states


def gibbs_switching_local_level(
    y: np.ndarray,
    n_iter: int = 1200,
    burn_in: int = 400,
    thin: int = 1,
    seed: int | None = 610,
) -> SwitchingResult:
    """Fit the two-state switching local-level DLM by Gibbs sampling."""
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 20:
        raise ValueError("need at least 20 observations to fit two regimes")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values")

    rng = np.random.default_rng(seed)
    n = y.size
    dy = np.diff(y)
    base = float(np.var(dy)) or 1e-6

    m0, C0 = float(y[0]), 1e3
    a_V, b_V = 3.0, 2.0 * base
    a_W, b_W = 3.0, 0.5 * base
    # a sticky prior: regimes that flip every other day are not regimes
    alpha_P = np.array([[12.0, 1.0], [1.0, 12.0]])

    # start from a crude volatility split so the two regimes begin apart
    roll = np.abs(np.concatenate([[0.0], dy]))
    states = (roll > np.median(roll)).astype(int)
    V = np.array([base, 2.0 * base])
    W = np.array([0.25 * base, 2.0 * base])
    P = np.array([[0.95, 0.05], [0.10, 0.90]])

    keep_V, keep_W, keep_P, keep_th, keep_s = [], [], [], [], []
    theta_sum = np.zeros(n)
    state_sum = np.zeros(n)
    n_keep = 0

    for it in range(n_iter):
        theta = _ffbs_varying(y, V[states], W[states], m0, C0, rng)
        states = _sample_states(y, theta, V, W, P, rng)

        resid = y - theta
        dth = np.concatenate([[0.0], np.diff(theta)])
        for s in range(N_STATES):
            mask = states == s
            k = int(mask.sum())
            ss_v = float(np.sum(resid[mask] ** 2))
            V[s] = 1.0 / rng.gamma(a_V + k / 2.0, 1.0 / (b_V + ss_v / 2.0))
            evo = mask.copy()
            evo[0] = False                      # the first point has no step
            ke = int(evo.sum())
            ss_w = float(np.sum(dth[evo] ** 2))
            W[s] = 1.0 / rng.gamma(a_W + ke / 2.0, 1.0 / (b_W + ss_w / 2.0))

        counts = np.zeros((N_STATES, N_STATES))
        np.add.at(counts, (states[:-1], states[1:]), 1.0)
        for s in range(N_STATES):
            P[s] = rng.dirichlet(alpha_P[s] + counts[s])

        # keep state 0 the calm one, so every draw means the same thing
        if W[0] > W[1]:
            V = V[::-1].copy()
            W = W[::-1].copy()
            P = P[::-1, ::-1].copy()
            states = 1 - states

        if it >= burn_in and (it - burn_in) % thin == 0:
            keep_V.append(V.copy())
            keep_W.append(W.copy())
            keep_P.append(P.copy())
            keep_th.append(theta[-1])
            keep_s.append(int(states[-1]))
            theta_sum += theta
            state_sum += states
            n_keep += 1

    return SwitchingResult(
        V=np.array(keep_V), W=np.array(keep_W), P=np.array(keep_P),
        theta_last=np.array(keep_th), state_last=np.array(keep_s, dtype=int),
        theta_mean=theta_sum / max(n_keep, 1),
        state_prob=state_sum / max(n_keep, 1),
    )


def predict_fan(
    prices: np.ndarray,
    horizon: int = 7,
    level: float = 0.95,
    n_iter: int = 1200,
    burn_in: int = 400,
    seed: int | None = 610,
) -> SwitchingForecast:
    """Posterior predictive prices for every horizon out to ``horizon`` days.

    One fit, every horizon: the regime chain and the level are propagated
    forward together, so day 1 through day ``horizon`` come from the same
    posterior and the band widens exactly as the model says it should.
    """
    prices = np.asarray(prices, dtype=float).ravel()
    prices = prices[np.isfinite(prices)]
    if prices.size < 20:
        raise ValueError("need at least 20 finite prices")
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive (log model)")
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")

    y = np.log(prices)
    res = gibbs_switching_local_level(y, n_iter=n_iter, burn_in=burn_in, seed=seed)

    rng = np.random.default_rng(seed)
    n_draw = res.theta_last.shape[0]
    theta = res.theta_last.copy()
    state = res.state_last.copy()
    paths = np.empty((n_draw, horizon))

    for h in range(horizon):
        u = rng.random(n_draw)
        state = (u > res.P[np.arange(n_draw), state, 0]).astype(int)
        w = res.W[np.arange(n_draw), state]
        v = res.V[np.arange(n_draw), state]
        theta = theta + rng.normal(0.0, np.sqrt(w))
        paths[:, h] = theta + rng.normal(0.0, np.sqrt(v))

    lo_q = (1.0 - level) / 2.0
    lower = np.exp(np.quantile(paths, lo_q, axis=0))
    point = np.exp(np.quantile(paths, 0.5, axis=0))
    upper = np.exp(np.quantile(paths, 1.0 - lo_q, axis=0))

    fitted = np.exp(res.theta_mean)
    rmse = float(np.sqrt(np.mean((prices - fitted) ** 2)))
    vol = tuple(float(np.sqrt(np.mean(res.W[:, s]))) for s in range(N_STATES))
    persist = tuple(float(np.mean(res.P[:, s, s])) for s in range(N_STATES))

    return SwitchingForecast(
        horizon=horizon, level=level, last_price=float(prices[-1]),
        lower=[float(v) for v in lower],
        point=[float(v) for v in point],
        upper=[float(v) for v in upper],
        fitted=[float(v) for v in fitted],
        state_prob=[float(v) for v in res.state_prob],
        vol_daily=vol, persistence=persist,
        p_volatile_now=float(np.mean(res.state_last)),
        rmse=rmse,
    )


def predict_interval_switching(
    prices: np.ndarray,
    horizon: int = 7,
    level: float = 0.95,
    n_iter: int = 1200,
    burn_in: int = 400,
    seed: int | None = 610,
) -> PredictionInterval:
    """The switching model behind the same interface as the plain DLM."""
    fan = predict_fan(prices, horizon=horizon, level=level,
                      n_iter=n_iter, burn_in=burn_in, seed=seed)
    return PredictionInterval(
        horizon=horizon,
        last_price=fan.last_price,
        point=fan.point[-1],
        lower=fan.lower[-1],
        upper=fan.upper[-1],
        level=level,
        expected_return=fan.expected_return,
    )
