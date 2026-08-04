"""Behavioural invariants that must hold regardless of internal refactoring.

These are the core of the safety net. Unlike golden-output tests they do not
pin specific numbers, so they survive intentional behaviour changes while
still catching the state-leak and replay classes of bug.
"""
import numpy as np
import pytest

import bigfeat.bigfeat_base as bb
from conftest import FIT_KWARGS

# --------------------------------------------------------------------------
# Each entry: (id, ctor kwargs, fixture name)
# --------------------------------------------------------------------------
#
# Two of these cases exist specifically to trigger bugs that the plain cases
# leave latent:
#
#   *_heavy_ts  -- ts_operation_weight_multiplier is raised so that time-series
#                  operators actually survive selection. With default weights a
#                  run may pick only one weak TS op, which hides the _is_sorted
#                  state leak almost entirely.
#   *_fanova    -- selection='fAnova' is required to reach the SelectKBest
#                  branch in transform(), which the time-series early-return
#                  currently skips.
#
CASES = [
    ("clf_nots", dict(task_type="classification", enable_time_series="no"),
     "clf_data", {}),
    ("reg_nots", dict(task_type="regression", enable_time_series="no"),
     "reg_data", {}),
    ("reg_ts", dict(task_type="regression", enable_time_series="yes",
                    datetime_col="date"), "periodic_data", {}),
    ("reg_ts_auto", dict(task_type="regression", enable_time_series="auto",
                         datetime_col="date"), "periodic_data", {}),
    ("reg_heavy_ts", dict(task_type="regression", enable_time_series="yes",
                          datetime_col="date",
                          ts_operation_weight_multiplier=50.0),
     "periodic_data", {}),
    ("reg_ts_fanova", dict(task_type="regression", enable_time_series="yes",
                           datetime_col="date",
                           ts_operation_weight_multiplier=50.0),
     "periodic_data", dict(selection="fAnova")),
    ("clf_nots_fanova", dict(task_type="classification",
                             enable_time_series="no"),
     "clf_data", dict(selection="fAnova")),
]
CASE_IDS = [c[0] for c in CASES]


# Cases whose ts_operation_weight_multiplier makes them materially slower.
SLOW_CASES = {"reg_heavy_ts", "reg_ts_fanova"}

CASE_PARAMS = [
    pytest.param(cid, marks=pytest.mark.slow) if cid in SLOW_CASES
    else pytest.param(cid)
    for cid in CASE_IDS
]


@pytest.fixture
def case(request):
    _id, kwargs, fixture_name, fit_extra = next(
        c for c in CASES if c[0] == request.param
    )
    X, y = request.getfixturevalue(fixture_name)
    return kwargs, X, y, {**FIT_KWARGS, **fit_extra}


@pytest.mark.parametrize("case", CASE_PARAMS, indirect=True)
def test_transform_is_stable_across_repeated_calls(case):
    """Two identical consecutive transform() calls must agree.

    Regression guard for the _is_sorted leak: the flag was set during fit and
    never reset, so transform() mutated shared state and successive calls on
    identical input returned different shapes and values.
    """
    kwargs, X, y, fit_kwargs = case
    bf = bb.BigFeat(verbose=False, **kwargs)
    bf.fit(X, y, **fit_kwargs)

    first = bf.transform(X)
    second = bf.transform(X)
    third = bf.transform(X)

    assert first.shape == second.shape == third.shape, (
        f"transform() shape is not stable across calls: "
        f"{first.shape} then {second.shape} then {third.shape}"
    )
    np.testing.assert_allclose(
        np.asarray(first, dtype=float), np.asarray(second, dtype=float),
        rtol=1e-9, atol=1e-9,
        err_msg="transform() values changed between identical consecutive calls",
    )


@pytest.mark.parametrize("case", CASE_PARAMS, indirect=True)
def test_fit_output_matches_transform_on_training_data(case):
    """fit(X) and transform(X) must produce the same matrix for the same X.

    fit() returns the feature matrix it built; transform() replays the stored
    recipes. If replay is faithful the two agree on the training data. This is
    the single strongest check on the recipe machinery -- it catches the
    _is_sorted leak, the fAnova early-return, and expression-tree replay bugs.
    """
    kwargs, X, y, fit_kwargs = case
    bf = bb.BigFeat(verbose=False, **kwargs)
    fitted = bf.fit(X, y, **fit_kwargs)
    replayed = bf.transform(X)

    assert fitted.shape == replayed.shape, (
        f"fit() returned {fitted.shape} but transform() on the same data "
        f"returned {replayed.shape}"
    )
    np.testing.assert_allclose(
        np.asarray(fitted, dtype=float), np.asarray(replayed, dtype=float),
        rtol=1e-6, atol=1e-6,
        err_msg="transform() did not reproduce fit()'s features on training data",
    )


@pytest.mark.parametrize("case", CASE_PARAMS, indirect=True)
def test_fit_is_deterministic_given_random_state(case):
    """random_state alone must determine the output (Phase 1 guarantee)."""
    kwargs, X, y, fit_kwargs = case

    def once():
        # Perturb the global RNG; a correctly seeded fit() ignores it.
        np.random.seed(np.random.randint(0, 100000))
        np.random.rand(313)
        bf = bb.BigFeat(verbose=False, **kwargs)
        return bf.fit(X, y, **fit_kwargs)

    a, b = once(), once()
    assert a.shape == b.shape
    np.testing.assert_allclose(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float),
        rtol=1e-9, atol=1e-9,
        err_msg="fit() output depends on the global RNG, not just random_state",
    )


@pytest.mark.parametrize("case", CASE_PARAMS, indirect=True)
def test_transform_is_invariant_to_input_row_order(case):
    """Shuffling the input rows must permute the output rows identically.

    This is the test that actually catches the _is_sorted leak. Checking that
    two consecutive transform() calls agree is NOT sufficient: the flag stays
    True for every call, so all of them take the early-return and are
    *consistently* wrong. Row order is the observable that breaks -- the
    early-return skips the datetime sort, so a row's features depend on where
    it happened to sit in the input rather than on its timestamp.

    For a correct time-aware transform, feature values are a function of the
    row's position in *time*, not its position in the array.
    """
    kwargs, X, y, fit_kwargs = case
    bf = bb.BigFeat(verbose=False, **kwargs)
    bf.fit(X, y, **fit_kwargs)

    straight = np.asarray(bf.transform(X), dtype=float)

    perm = np.random.RandomState(0).permutation(len(X))
    X_shuffled = X.iloc[perm].reset_index(drop=True)
    shuffled_out = np.asarray(bf.transform(X_shuffled), dtype=float)

    # Undo the permutation: row i of the shuffled output corresponds to
    # row perm[i] of the original.
    restored = np.empty_like(shuffled_out)
    restored[perm] = shuffled_out

    assert straight.shape == restored.shape
    np.testing.assert_allclose(
        straight, restored, rtol=1e-6, atol=1e-6,
        err_msg=(
            "transform() output depends on input row order; time-series "
            "features must be determined by timestamp, not array position"
        ),
    )


@pytest.mark.parametrize("case", CASE_PARAMS, indirect=True)
def test_transform_row_count_is_preserved(case):
    """transform() must return exactly one row per input row, in input order."""
    kwargs, X, y, fit_kwargs = case
    bf = bb.BigFeat(verbose=False, **kwargs)
    bf.fit(X, y, **fit_kwargs)
    out = bf.transform(X)
    assert out.shape[0] == len(X)


def test_fanova_selection_is_applied_on_the_time_series_path(periodic_data):
    """selection='fAnova' must narrow transform() output, TS enabled or not.

    transform() returns early on the time-series path, before the SelectKBest
    branch. The mismatch is masked while _is_sorted causes that early-return to
    fire first, so this asserts the contract directly rather than relying on
    the masking bug staying in place.
    """
    X, y = periodic_data
    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", ts_operation_weight_multiplier=50.0,
                    verbose=False)
    fitted = bf.fit(X, y, **{**FIT_KWARGS, "selection": "fAnova"})
    assert bf.enable_time_series is True, "fixture must exercise the TS path"

    out = bf.transform(X)
    assert out.shape[1] == fitted.shape[1], (
        f"fit() applied fAnova and returned {fitted.shape[1]} columns but "
        f"transform() returned {out.shape[1]}; the SelectKBest step was "
        f"skipped by the time-series early-return"
    )


@pytest.mark.parametrize("case", CASE_PARAMS, indirect=True)
def test_transform_output_is_finite(case):
    """Generated features must not contain NaN/inf."""
    kwargs, X, y, fit_kwargs = case
    bf = bb.BigFeat(verbose=False, **kwargs)
    bf.fit(X, y, **fit_kwargs)
    out = np.asarray(bf.transform(X), dtype=float)
    assert np.isfinite(out).all(), (
        f"{(~np.isfinite(out)).sum()} non-finite values in transform() output"
    )
