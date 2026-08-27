import numpy as np
import pytest

from stockts.hmm import (
    SwitchingForecast,
    gibbs_switching_local_level,
    predict_fan,
    predict_interval_switching,
)


@pytest.fixture
def regime_prices():
    """A calm stretch, a volatile stretch, then calm again.

    The middle block is an order of magnitude noisier at the level, which is
    exactly the structure the two-state model is meant to find.
    """
    rng = np.random.default_rng(0)
    y = [np.log(100.0)]
    states = []
    for i in range(360):
        s = 1 if 140 <= i < 250 else 0
        states.append(s)
        y.append(y[-1] + rng.normal(0.0, 0.004 if s == 0 else 0.030))
    obs = np.array(y[1:]) + rng.normal(0.0, 0.002, 360)
    return np.exp(obs), np.array(states)


def test_shapes_and_positivity(gbm_prices):
    res = gibbs_switching_local_level(np.log(gbm_prices), n_iter=300, burn_in=100, seed=1)
    n_keep = 200
    assert res.V.shape == (n_keep, 2)
    assert res.W.shape == (n_keep, 2)
    assert res.P.shape == (n_keep, 2, 2)
    assert res.theta_last.shape == (n_keep,)
    assert res.theta_mean.shape == gbm_prices.shape
    assert res.state_prob.shape == gbm_prices.shape
    assert np.all(res.V > 0) and np.all(res.W > 0)
    assert np.all((res.state_prob >= 0) & (res.state_prob <= 1))
    # every row of every transition matrix is a distribution
    assert np.allclose(res.P.sum(axis=2), 1.0)


def test_regimes_stay_ordered(gbm_prices):
    """State 0 is the calm one in every single draw, not just on average."""
    res = gibbs_switching_local_level(np.log(gbm_prices), n_iter=300, burn_in=100, seed=3)
    assert np.all(res.W[:, 0] <= res.W[:, 1])


def test_finds_the_volatile_stretch(regime_prices):
    prices, truth = regime_prices
    res = gibbs_switching_local_level(np.log(prices), n_iter=600, burn_in=200, seed=1)
    p_vol = res.state_prob
    assert p_vol[truth == 1].mean() > 0.8
    assert p_vol[truth == 0].mean() < 0.2
    # and it recovers the gap between the two regimes' level volatility
    sd = np.sqrt(res.W.mean(axis=0))
    assert sd[1] > 4 * sd[0]


def test_regimes_are_persistent(regime_prices):
    prices, _ = regime_prices
    res = gibbs_switching_local_level(np.log(prices), n_iter=600, burn_in=200, seed=1)
    diag = res.P.mean(axis=0).diagonal()
    assert np.all(diag > 0.8)


def test_fan_widens_with_horizon(gbm_prices):
    fan = predict_fan(gbm_prices, horizon=7, n_iter=300, burn_in=100, seed=2)
    assert isinstance(fan, SwitchingForecast)
    assert len(fan.lower) == len(fan.point) == len(fan.upper) == 7
    widths = [u - l for l, u in zip(fan.lower, fan.upper)]
    assert widths[-1] > widths[0]
    for lo, pt, up in zip(fan.lower, fan.point, fan.upper):
        assert lo < pt < up
        assert lo > 0
    assert len(fan.fitted) == len(gbm_prices)
    assert len(fan.state_prob) == len(gbm_prices)
    assert fan.rmse > 0


def test_fitted_tracks_the_prices(gbm_prices):
    """The fitted level is a level, not a straight line through the mean."""
    fan = predict_fan(gbm_prices, horizon=3, n_iter=300, burn_in=100, seed=2)
    fitted = np.array(fan.fitted)
    assert fan.rmse < gbm_prices.std()
    assert np.corrcoef(fitted, gbm_prices)[0, 1] > 0.95


def test_seed_is_reproducible(gbm_prices):
    a = predict_fan(gbm_prices, horizon=4, n_iter=300, burn_in=100, seed=7)
    b = predict_fan(gbm_prices, horizon=4, n_iter=300, burn_in=100, seed=7)
    assert a.point == b.point and a.lower == b.lower and a.upper == b.upper


def test_interval_wrapper_matches_the_fan(gbm_prices):
    kw = dict(horizon=5, n_iter=300, burn_in=100, seed=5)
    pi = predict_interval_switching(gbm_prices, **kw)
    fan = predict_fan(gbm_prices, **kw)
    assert pi.horizon == 5
    assert pi.point == pytest.approx(fan.point[-1])
    assert pi.lower == pytest.approx(fan.lower[-1])
    assert pi.upper == pytest.approx(fan.upper[-1])
    assert pi.last_price == pytest.approx(float(gbm_prices[-1]))


def test_rejects_short_and_negative_series():
    with pytest.raises(ValueError):
        predict_fan(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        predict_fan(np.full(40, -1.0))
    with pytest.raises(ValueError):
        predict_fan(np.full(40, 100.0), horizon=0)
