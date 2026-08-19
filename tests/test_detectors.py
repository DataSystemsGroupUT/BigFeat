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


# ---------------------------------------------------------------------------
# ACF must recover fundamentals, not harmonics (PIPELINE_FIXES_SPEC.md Fix 1)
# ---------------------------------------------------------------------------

def _planted(periods, n=1000, seed=0, noise=0.5):
    rs = np.random.RandomState(seed)
    t = np.arange(n)
    sig = sum(10 * np.sin(2 * np.pi * t / p) for p in periods) + rs.randn(n) * noise
    return sig


def _acf_accepted_lags(sig, max_lag=400):
    det = ACFWindowDetector(verbose=False)
    acf = det._compute_acf(det._preprocess_signal(sig), max_lag=max_lag)
    lags, meta = det._find_acf_peaks(acf, min_lag=3)
    return list(lags), meta


def test_acf_returns_the_shortest_fundamental_first_with_no_harmonics():
    """The shared fixture: planted 7 and 30 days.

    Height-sorted peak picking returned [210, 91, 301] -- common multiples,
    because ACF at lcm-type lags exceeds the fundamentals (ACF(210)=0.997 vs
    ACF(7)=0.652). The achievable lag-domain contract is: the SHORTEST
    fundamental first, and no harmonic junk. The second period (30d) is NOT
    an ACF local maximum at all -- the 7d comb tooth at 28 towers over it --
    which is exactly why multi-period recovery is the ensemble's job
    (DFT/LS propose, ACF verifies; see PIPELINE_FIXES_SPEC Fixes 2 and 4).
    """
    sig = _planted([7, 30], n=730)
    lags, _ = _acf_accepted_lags(sig, max_lag=365)
    assert lags, "no fundamentals accepted"
    assert 6 <= lags[0] <= 8, f"first accepted lag {lags[0]} is not the 7d fundamental"
    assert all(l <= 60 for l in lags), (
        f"harmonic/multiple lags survived: {[l for l in lags if l > 60]}"
    )


def test_acf_single_period_reports_the_fundamental_not_a_multiple():
    """Planted [30] alone previously returned 360, 120, 330."""
    sig = _planted([30])
    lags, _ = _acf_accepted_lags(sig)
    assert lags, "no peaks accepted on a clean 30d signal"
    assert 27 <= lags[0] <= 33, f"top accepted lag {lags[0]} is not the 30d fundamental"


def test_acf_recovers_non_multiple_period_pair():
    """Planted [12, 52] previously returned 156, 312, 360 as its top-3 -- neither
    fundamental. The shortest must now lead, with its harmonics masked."""
    sig = _planted([12, 52])
    lags, _ = _acf_accepted_lags(sig)
    top3 = lags[:3]          # what detect_optimal_windows actually consumes
    assert any(10 <= l <= 14 for l in top3), f"12d missing from top-3: {top3}"
    # 52 itself is displaced by the 12d comb (teeth at 48/60); requiring it
    # here would overspecify the lag domain. What MUST hold: no multiple of
    # the accepted fundamental masquerades as a second period.
    assert not any(l in (24, 36, 48, 60, 72) for l in top3), (
        f"harmonics of 12 survived as periods: {top3}"
    )


def test_acf_verification_rejects_isolated_noise_spike():
    """A lone tall spike in the ACF is not a period: a true period repeats at
    its multiples. The verification step must reject candidates whose 2L
    neighbourhood shows nothing."""
    det = ACFWindowDetector(verbose=False)
    acf = np.zeros(200)
    acf[0] = 1.0
    acf[40] = 0.6          # isolated spike, no echo at 80 or 120
    lags, _ = det._find_acf_peaks(acf, min_lag=3)
    assert 40 not in list(lags), "isolated spike accepted as a period"


# ---------------------------------------------------------------------------
# DFT must propose ALL strong spectral peaks (PIPELINE_FIXES_SPEC.md Fix 4)
# ---------------------------------------------------------------------------

def test_dft_recovers_non_multiple_period_pair():
    """Planted [11, 31], amplitudes 10 and 7.

    Chosen so neither period's harmonic ladder ({P/2, 2P, 4P}) lands inside
    the other's tolerance window -- 11 -> {5.5, 22, 44}, 31 -> {15.5, 62,
    124} -- so the ladder cannot fake the recovery (an earlier draft used
    [11, 45] and passed on unfixed code because 4*11 = 44 sat inside the
    45-day window). A single argmax returns only the stronger 11d component;
    top-k spectral peaks must recover the 31d one as well."""
    rs = np.random.RandomState(0)
    n = 1000
    t = np.arange(n)
    sig = (10 * np.sin(2 * np.pi * t / 11)
           + 7 * np.sin(2 * np.pi * t / 31) + rs.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "v": sig,
    })
    det = DFTWindowDetector(verbose=False)
    windows, conf = det.detect_optimal_windows(df, "date", ["v"], sampling_rate="D")
    days = sorted(w.days for w in windows)
    assert any(9 <= d <= 13 for d in days), f"11d missing: {days}"
    assert any(27 <= d <= 36 for d in days), f"31d missing: {days}"


def test_dft_multi_peak_does_not_regress_noise_rejection():
    """Top-k must not turn noise bins into detections: white noise still
    yields no periodicity verdict."""
    det = DFTWindowDetector(verbose=False)
    is_p, confv, _ = det.assess_periodicity(_frame("noise"), "date", FEATURE_COLS)
    assert not is_p, f"noise detected as periodic at conf {confv:.3f}"


def test_ladder_never_drops_detected_fundamentals():
    """Fix 5: truncation to n_windows removes derived harmonics, never the
    detected periods themselves. Ascending truncation previously cut the
    ladder [4,5,8,11,15,16,22,30,...] at slot six, losing detected 30 to
    derived 4 and 5."""
    det = DFTWindowDetector(verbose=False, n_windows=4)
    windows = det._generate_multiscale_windows([7.0, 30.0, 91.0])
    days = {w.days for w in windows}
    assert {7, 30, 91} <= days, f"fundamental dropped: kept {sorted(days)}"
    assert len(windows) <= 4


# ---------------------------------------------------------------------------
# End-to-end: ensemble windows and lags on the shared two-period fixture
# (PIPELINE_FIXES_SPEC.md Fixes 2/3 acceptance)
# ---------------------------------------------------------------------------

def _fixture_frame():
    rs = np.random.RandomState(0)
    n = 730
    t = np.arange(n)
    sig = (10 * np.sin(2 * np.pi * t / 7)
           + 8 * np.sin(2 * np.pi * t / 30) + rs.randn(n) * 0.5)
    X = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "v": sig, "u": rs.rand(n),
    })
    y = pd.Series(np.roll(sig, -1))
    return X, y


def _fit_fixture():
    bf = bb.BigFeat(task_type="regression", enable_time_series="auto",
                    datetime_col="date", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bf.fit(_fixture_frame()[0], _fixture_frame()[1],
               gen_size=3, iterations=1, random_state=0)
    return bf


def test_ensemble_windows_bracket_both_planted_periods():
    """Before Fixes 1/4/5 the pooled set was [1,4,7,14,105,210]: the 30-day
    period lost entirely, two slots on ACF harmonics. Pins the repair."""
    bf = _fit_fixture()
    days = sorted(w.days for w in bf.window_sizes)
    assert any(6 <= d <= 8 for d in days), f"7d not bracketed: {days}"
    assert any(26 <= d <= 34 for d in days), f"30d not bracketed: {days}"
    assert all(d <= 60 for d in days), f"harmonic junk: {[d for d in days if d > 60]}"


def test_lag_periods_contain_the_detected_fundamentals():
    """Fix 3: lags were positional picks from the window ladder
    (windows[0], windows[1], windows[mid]) -- on this fixture [1, 3, 7],
    with the 30-day fundamental absent and 7 present only by accident of
    position. A lag should EQUAL a detected cycle; a window should span one."""
    bf = _fit_fixture()
    lag_days = sorted(getattr(l, "days", l) for l in bf.lag_periods)
    assert 1 in lag_days, f"1-step lag missing: {lag_days}"
    assert any(6 <= d <= 8 for d in lag_days), f"7d lag missing: {lag_days}"
    assert any(26 <= d <= 34 for d in lag_days), f"30d lag missing: {lag_days}"


# ---------------------------------------------------------------------------
# Stationarity gate must measure stationarity, not smoothness
# (PIPELINE_STAGE_REVIEW.md section 2)
# ---------------------------------------------------------------------------

def _fit_auto(X, y):
    bf = bb.BigFeat(task_type="regression", enable_time_series="auto",
                    datetime_col="date", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bf.fit(X, y, gen_size=2, iterations=1, random_state=0)
    return bf


def _restricted(bf):
    ops = getattr(bf, "time_series_operators", None)
    return ops is not None and len(ops) <= 3


def test_smooth_stationary_seasonal_keeps_the_full_operator_pool():
    """A clean 30-day seasonal is STATIONARY, but its smoothness gives it
    lag-1 autocorrelation ~0.97 -- above the old 0.85 gate, which therefore
    stripped rolling/seasonal operators from exactly the data this subsystem
    exists for. ADF classifies it correctly (p ~ 0.000)."""
    rs = np.random.RandomState(0)
    n = 400
    t = np.arange(n)
    X = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="D"),
        "a": 10 * np.sin(2 * np.pi * t / 30) + rs.randn(n) * 0.5,
        "b": 8 * np.sin(2 * np.pi * t / 30 + 1.0) + rs.randn(n) * 0.5,
    })
    # guard: the fixture really is the old gate's false-positive case
    lag1 = float(np.mean([abs(X[c].autocorr(1)) for c in ("a", "b")]))
    assert lag1 > 0.85, f"fixture drift: lag1={lag1:.3f} no longer trips the old gate"

    y = pd.Series(X["a"].values * 1.5 + rs.randn(n) * 0.3)
    bf = _fit_auto(X, y)
    assert bf.enable_time_series, "periodic fixture must enable time series"
    assert not _restricted(bf), (
        "stationary seasonal data was routed to restricted mode: the gate is "
        "measuring smoothness, not stationarity"
    )


def test_unit_root_below_the_lag1_radar_is_still_restricted():
    """A random walk observed with measurement noise has lag-1 autocorrelation
    BELOW 0.85 (the old gate waves it through to full mode) while ADF still
    shows a clear unit root -- the covid_deaths failure mode, where a
    smoothing operator was selected on trending data and cost 5x in MASE."""
    rs = np.random.RandomState(2)
    n = 400
    t = np.arange(n)
    cols = {}
    for i, k in enumerate((3.0, 3.2, 3.4)):
        walk = np.cumsum(rs.randn(n))
        cols[f"c{i}"] = walk + rs.randn(n) * k + 2 * np.sin(2 * np.pi * t / 7)
    X = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=n, freq="D"),
                      **cols})
    lag1 = float(np.mean([abs(X[c].autocorr(1)) for c in cols]))
    assert lag1 < 0.85, f"fixture drift: lag1={lag1:.3f} would trip the old gate anyway"

    y = pd.Series(X["c0"].values + rs.randn(n) * 0.3)
    bf = _fit_auto(X, y)
    if not bf.enable_time_series:
        pytest.skip("detectors declined this fixture entirely; gate untested")
    assert _restricted(bf), (
        f"unit-root data (avg lag1={lag1:.3f}, under the old 0.85 radar) "
        f"reached the FULL operator pool; rolling/smoothing operators will "
        f"describe the trend, not the signal"
    )
