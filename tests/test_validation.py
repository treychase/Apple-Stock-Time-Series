import numpy as np
import pandas as pd
import pytest

from stockts.validation import assert_valid, validate_history


def test_valid_history_passes(history):
    report = validate_history(history)
    assert report.ok, str(report)
    assert report.n_tickers == history.shape[1]
    assert report.n_rows == history.shape[0]


def test_empty_history_fails():
    report = validate_history(pd.DataFrame())
    assert not report.ok
    assert any("empty" in p for p in report.problems)


def test_too_few_rows(history):
    report = validate_history(history.head(10), min_rows=30)
    assert not report.ok
    assert any("rows" in p for p in report.problems)


def test_all_nan_column(history):
    h = history.copy()
    h["DEAD"] = np.nan
    report = validate_history(h)
    assert not report.ok
    assert any("DEAD" in p and "NaN" in p for p in report.problems)


def test_non_positive_prices(history):
    h = history.copy()
    h.iloc[0, 0] = -5.0
    report = validate_history(h)
    assert not report.ok
    assert any("non-positive" in p for p in report.problems)


def test_implausible_move(history):
    h = history.copy()
    h.iloc[50, 1] = h.iloc[49, 1] * 5  # +400% spike
    report = validate_history(h)
    assert not report.ok
    assert any("implausible" in p for p in report.problems)


def test_unsorted_index(history):
    h = history.iloc[::-1]
    report = validate_history(h)
    assert not report.ok
    assert any("sorted" in p for p in report.problems)


def test_assert_valid_raises(history):
    h = history.copy()
    h["DEAD"] = np.nan
    with pytest.raises(ValueError):
        assert_valid(h)


def test_assert_valid_passes(history):
    report = assert_valid(history)
    assert report.ok
