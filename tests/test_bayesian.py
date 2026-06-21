import numpy as np
import pytest

from stockts.bayesian import gibbs_local_level, predict_interval


def test_gibbs_shapes_and_positivity(gbm_prices):
    y = np.log(gbm_prices)
    res = gibbs_local_level(y, n_iter=400, burn_in=100, seed=1)
    n_keep = 300
    assert res.V.shape == (n_keep,)
    assert res.W.shape == (n_keep,)
    assert res.theta_last.shape == (n_keep,)
    assert np.all(res.V > 0)
    assert np.all(res.W > 0)
    assert np.all(np.isfinite(res.theta_last))


def test_gibbs_requires_enough_data():
    with pytest.raises(ValueError):
        gibbs_local_level(np.array([1.0, 2.0, 3.0]))


def test_gibbs_rejects_non_finite():
    with pytest.raises(ValueError):
        gibbs_local_level(np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0]))


def test_predict_interval_ordering(gbm_prices):
    pi = predict_interval(gbm_prices, horizon=5, level=0.95, n_iter=400, burn_in=100)
    assert pi.lower < pi.point < pi.upper
    assert pi.lower > 0  # log model keeps prices positive
    assert pi.horizon == 5
    assert pi.last_price == pytest.approx(gbm_prices[-1])
    # expected_return consistent with point vs last price
    expected = (pi.point - pi.last_price) / pi.last_price
    assert pi.expected_return == pytest.approx(expected)


def test_predict_interval_reproducible(gbm_prices):
    a = predict_interval(gbm_prices, n_iter=300, burn_in=80, seed=7)
    b = predict_interval(gbm_prices, n_iter=300, burn_in=80, seed=7)
    assert a.point == pytest.approx(b.point)
    assert a.lower == pytest.approx(b.lower)
    assert a.upper == pytest.approx(b.upper)


def test_wider_level_gives_wider_interval(gbm_prices):
    narrow = predict_interval(gbm_prices, level=0.80, n_iter=400, burn_in=100, seed=3)
    wide = predict_interval(gbm_prices, level=0.99, n_iter=400, burn_in=100, seed=3)
    assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)


def test_longer_horizon_widens_interval(gbm_prices):
    short = predict_interval(gbm_prices, horizon=1, n_iter=400, burn_in=100, seed=5)
    long = predict_interval(gbm_prices, horizon=10, n_iter=400, burn_in=100, seed=5)
    assert (long.upper - long.lower) > (short.upper - short.lower)


def test_predict_interval_rejects_nonpositive():
    with pytest.raises(ValueError):
        predict_interval(np.array([1.0, -2.0, 3.0, 4.0, 5.0, 6.0]))
