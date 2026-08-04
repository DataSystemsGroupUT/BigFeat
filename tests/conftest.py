"""Shared fixtures for the BigFeat characterization suite.

The generators below are pinned: each one is known to drive
``_setup_time_series`` down a specific branch. ``test_fixtures.py`` asserts
that they still do, so that if detector behaviour shifts, the fixture guard
fails loudly rather than other tests silently changing what they cover.
"""
import numpy as np
import pandas as pd
import pytest

# Small by design: the whole suite should run in seconds. These sizes were
# checked to still exercise the real code paths (periodicity detection needs
# roughly 40+ rows; the non-TS path works down to ~20).
N_ROWS = 300
FIT_KWARGS = dict(gen_size=3, iterations=2, random_state=0)


def pytest_addoption(parser):
    parser.addoption(
        "--regen-golden", action="store_true", default=False,
        help="Rewrite tests/golden.json from current behaviour instead of "
             "asserting against it. Review the resulting diff carefully.",
    )


@pytest.fixture
def regen_golden(request):
    return request.config.getoption("--regen-golden")


def _dates(n=N_ROWS, freq="D"):
    return pd.date_range("2020-01-01", periods=n, freq=freq)


@pytest.fixture
def clf_data():
    """Plain tabular classification, no datetime column."""
    rs = np.random.RandomState(0)
    X = pd.DataFrame(rs.rand(N_ROWS, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(((X["f0"] + 2 * X["f1"] - X["f2"]) > 1.0).astype(int))
    return X, y


@pytest.fixture
def reg_data():
    """Plain tabular regression, no datetime column."""
    rs = np.random.RandomState(1)
    X = pd.DataFrame(rs.rand(N_ROWS, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(3 * X["f0"] - X["f1"] + rs.randn(N_ROWS) * 0.1)
    return X, y


@pytest.fixture
def periodic_data():
    """Clean weekly+monthly seasonality -> detectors agree, 'pooled_ensemble'."""
    rs = np.random.RandomState(2)
    t = np.arange(N_ROWS)
    X = pd.DataFrame({
        "date": _dates(),
        "a": np.sin(2 * np.pi * t / 7) + rs.randn(N_ROWS) * 0.05,
        "b": np.sin(2 * np.pi * t / 30) + rs.randn(N_ROWS) * 0.05,
        "c": rs.rand(N_ROWS),
    })
    y = pd.Series(2 * X["a"].values + X["b"].values + rs.randn(N_ROWS) * 0.1)
    return X, y


@pytest.fixture
def nonstationary_data():
    """Random walk + weekly signal -> lag-1 autocorr > 0.85, restricted mode."""
    rs = np.random.RandomState(3)
    t = np.arange(N_ROWS)
    walk = np.cumsum(rs.randn(N_ROWS)) * 0.5
    X = pd.DataFrame({
        "date": _dates(),
        "a": walk + 3 * np.sin(2 * np.pi * t / 7),
        "b": np.cumsum(rs.randn(N_ROWS)) * 0.3,
        "c": walk * 0.5 + rs.randn(N_ROWS) * 0.1,
    })
    y = pd.Series(X["a"].values + rs.randn(N_ROWS) * 0.1)
    return X, y
