"""Regression tests for the periodicity detectors.

The central property: a detector must say "periodic" for periodic data and
"not periodic" for noise. All three failed that on one side or the other.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import bigfeat.bigfeat_base as bb
from bigfeat.acf_window_detector import ACFWindowDetector
from bigfeat.dft_window_detector import DFTWindowDetector
from bigfeat.lomb_scargle_window_detector import LombScargleWindowDetector

N = 365
FEATURE_COLS = ["a", "b", "c", "d"]


def _frame(kind, seed=0, n=N, freq="D"):
    rs = np.random.RandomState(seed)
    t = np.arange(n)
    if kind == "periodic":
        a = np.sin(2 * np.pi * t / 7) + rs.randn(n) * 0.05
        b = np.sin(2 * np.pi * t / 30) + rs.randn(n) * 0.05
        c = np.sin(2 * np.pi * t / 7) + rs.randn(n) * 0.05
        d = np.sin(2 * np.pi * t / 30) + rs.randn(n) * 0.05
    else:  # pure white noise
        a, b, c, d = (rs.randn(n) for _ in range(4))
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq=freq),
        "a": a, "b": b, "c": c, "d": d,
    })


def _detectors():
    return {
        "dft": DFTWindowDetector(verbose=False),
        "acf": ACFWindowDetector(verbose=False),
        "lomb_scargle": LombScargleWindowDetector(verbose=False),
    }


@pytest.mark.parametrize("name", ["dft", "acf", "lomb_scargle"])
def test_detector_finds_periodicity_in_periodic_data(name):
    """A clean 7/30-day seasonal signal must be detected as periodic.

    DFT reported periodic=False with confidence 0.244 on exactly this data.
    Its confidence was `1 - sorted[1]/sorted[0]`, and since np.sort puts the
    two largest bins adjacent -- which for a real peak are neighbouring bins
    of the SAME peak, split by spectral leakage -- strong periodicity drove
    the score toward zero.
    """
    det = _detectors()[name]
    is_periodic, confidence, _ = det.assess_periodicity(
        _frame("periodic"), "date", FEATURE_COLS)

    assert is_periodic, (
        f"{name} failed to detect a clean 7/30-day seasonal signal "
        f"(confidence={confidence:.3f})"
    )
    assert confidence > 0.3


@pytest.mark.parametrize("name", ["dft", "acf", "lomb_scargle"])
def test_detector_rejects_white_noise(name):
    """White noise must NOT be reported as periodic.

    ACF reported periodic=True with confidence 0.973 on pure noise, because
    _compute_acf ran up to lag len(series)-1. At lag 363 of a 365-sample
    series only 2 points overlap, and np.corrcoef of two points is always
    exactly +/-1. Those spurious unit correlations sorted to the top by
    height and became both the reported period and the confidence.
    """
    det = _detectors()[name]
    is_periodic, confidence, _ = det.assess_periodicity(
        _frame("noise"), "date", FEATURE_COLS)

    assert not is_periodic, (
        f"{name} reported white noise as periodic (confidence={confidence:.3f})"
    )


def test_acf_does_not_produce_spurious_unit_correlations():
    """Directly pin the short-overlap artifact in _compute_acf."""
    det = ACFWindowDetector(verbose=False)
    noise = np.random.RandomState(0).randn(365)

    acf = det._compute_acf(noise, max_lag=364)

    # Every computed lag must retain enough overlap to be meaningful. With the
    # old unbounded max_lag, |ACF| reached exactly 1.0 beyond lag 300.
    assert np.abs(acf[1:]).max() < 0.5, (
        f"white-noise ACF reached {np.abs(acf[1:]).max():.3f}; short-overlap "
        f"lags are producing spurious near-unit correlations"
    )


def test_acf_max_lag_respects_sampling_rate():
    """max_window_days is a DURATION and must be converted to sample counts.

    It was previously used directly as a lag count, so at hourly sampling a
    365-day ceiling became 365 hours (~15 days).
    """
    det = ACFWindowDetector(verbose=False, max_window_days=365)
    n = 24 * 40  # 40 days of hourly data
    t = np.arange(n)
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="h"),
        # Strong daily cycle: period is 24 SAMPLES but only 1 DAY.
        "a": np.sin(2 * np.pi * t / 24) + np.random.RandomState(0).randn(n) * 0.05,
    })

    windows, confidences = det.detect_optimal_windows(df, "date", ["a"],
                                                      sampling_rate="h")
    assert windows, "no windows detected on a strong hourly daily-cycle signal"
    # The detected period should be about a day, not about 24 days.
    assert min(w.days for w in windows) <= 7, (
        f"windows {[w.days for w in windows]} suggest lags were interpreted "
        f"as days rather than samples"
    )


def test_dft_confidence_separates_signal_from_noise():
    """The confidence metric must rank periodic data above noise."""
    det = DFTWindowDetector(verbose=False)
    _, conf_periodic, _ = det.assess_periodicity(
        _frame("periodic"), "date", FEATURE_COLS)
    _, conf_noise, _ = det.assess_periodicity(
        _frame("noise"), "date", FEATURE_COLS)

    assert conf_periodic > conf_noise, (
        f"DFT confidence does not separate signal ({conf_periodic:.3f}) from "
        f"noise ({conf_noise:.3f})"
    )
    # Previously these were 0.244 and 0.075 -- both below threshold and only
    # 0.17 apart. Require a decisive margin.
    assert conf_periodic - conf_noise > 0.2


def test_ensemble_does_not_enable_time_series_on_noise():
    """The end-to-end failure this phase set out to fix.

    With ACF voting periodic at 0.973 confidence on white noise, the default
    'auto' ensemble enabled time-series features and selected windows of
    163-363 days -- precisely the hallucinated seasonality the consensus vote
    is supposed to prevent.
    """
    df = _frame("noise", seed=0)
    y = pd.Series(np.random.RandomState(1).randn(len(df)))

    bf = bb.BigFeat(task_type="regression", enable_time_series="auto",
                    datetime_col="date", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bf.fit(df, y, gen_size=2, iterations=1, random_state=0)

    assert bf.enable_time_series is False, (
        f"time series was enabled on pure white noise with strategy "
        f"{bf.detection_strategy!r} and windows "
        f"{[w.days for w in (bf.window_sizes or [])]}"
    )


def test_ensemble_still_enables_time_series_on_periodic_data():
    """The rejection above must not come at the cost of real detections."""
    df = _frame("periodic", seed=0)
    y = pd.Series(df["a"].values * 2
                  + np.random.RandomState(1).randn(len(df)) * 0.1)

    bf = bb.BigFeat(task_type="regression", enable_time_series="auto",
                    datetime_col="date", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bf.fit(df, y, gen_size=2, iterations=1, random_state=0)

    assert bf.enable_time_series is True, \
        "time series was disabled on a clean 7/30-day seasonal signal"
    assert bf.window_sizes, "no window sizes selected"


def test_sampling_rate_is_inferred_from_the_data():
    """Detectors must not assume daily sampling.

    Detectors work in SAMPLES and convert to days via sampling_rate. The
    ensemble path never passed one, so it defaulted to 'D' and monthly or
    quarterly observations were treated as daily -- every detected period
    came out 30-90x too small.
    """
    det = DFTWindowDetector(verbose=False)
    for freq, expected in (("D", "D"), ("W", "W"), ("ME", "M"), ("QE", "Q")):
        df = pd.DataFrame({
            "date": pd.date_range("2000-01-31", periods=60, freq=freq),
            "item": ["A"] * 60,
            "v": np.arange(60, dtype=float),
        })
        got = det.infer_sampling_rate(df, "date", ["item"])
        assert got == expected, (
            f"freq={freq}: inferred {got!r}, expected {expected!r}"
        )


def test_pooled_windows_keep_the_long_scales():
    """Truncating the pooled window list must not drop every large window.

    The ensemble sorted pooled candidates ascending then took the first
    n_windows, which always discarded the LONG windows. Pooling three
    detectors reliably yields more than n_windows candidates, so on monthly
    data this left windows of 1-6 days for series sampled once a month.
    """
    rs = np.random.RandomState(0)
    n = 120
    t = np.arange(n)
    df = pd.DataFrame({
        "date": pd.date_range("2000-01-31", periods=n, freq="ME"),
        "item": ["A"] * n,
        "v": np.sin(2 * np.pi * t / 12) * 10 + t * 0.1 + rs.randn(n) * 0.2,
    })
    y = pd.Series(df["v"].values)

    bf = bb.BigFeat(task_type="regression", enable_time_series="auto",
                    datetime_col="date", groupby_cols=["item"], verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bf.fit(df, y, gen_size=2, iterations=1, random_state=0)

    if not bf.enable_time_series:
        pytest.skip("fixture did not enable time series")

    days = sorted(w.days for w in bf.window_sizes)
    assert max(days) > 30, (
        f"all pooled windows are short ({days}) for monthly data; the long "
        f"scales were truncated away"
    )
