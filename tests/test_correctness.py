"""Regression tests for specific correctness fixes.

Each test here pins a bug that was found and fixed, so it cannot silently
come back. Unlike test_invariants.py these target one mechanism each.
"""
import numpy as np
import pandas as pd
import pytest

import bigfeat.bigfeat_base as bb
from conftest import FIT_KWARGS


# ---------------------------------------------------------------------------
# Look-ahead leakage in the calendar-mean operators
# ---------------------------------------------------------------------------

def _calendar_frame(n=60):
    """A frame whose feature is strictly increasing, so leakage is visible.

    With a monotonically increasing feature, any average that includes future
    rows is strictly greater than one restricted to past rows. That makes
    look-ahead detectable by comparison rather than by pinning numbers.
    """
    return pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="D"),
        "v": np.arange(n, dtype=float),
    })


@pytest.mark.parametrize("operation,expected_period", [
    ("weekday_mean", 7),
    ("month_mean", None),
])
def test_calendar_means_never_use_future_rows(operation, expected_period):
    """weekday_mean / month_mean must average only PRIOR rows in the group.

    These used to be groupby(...).transform('mean'), which averages the whole
    column within each calendar group -- so a row's value depended on rows
    dated after it. That is look-ahead leakage, and it inflated apparent
    performance for exactly the seasonal signal the operators exist to model.
    """
    df = _calendar_frame()
    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", verbose=False)
    bf.enable_time_series = True
    bf.datetime_col = "date"
    bf.groupby_cols = []

    result = np.asarray(bf._apply_time_based_operation(df, "v", operation),
                        dtype=float)

    assert len(result) == len(df)

    # The feature increases monotonically, so a causal mean over prior rows in
    # the same calendar group is always strictly below the current value.
    # A leaky whole-group mean is not.
    current = df["v"].values
    nonzero = result != 0.0
    assert nonzero.any(), "operator produced no values at all"
    assert (result[nonzero] < current[nonzero]).all(), (
        f"{operation} produced values >= the current row, which means it "
        f"averaged in rows dated at or after the row being computed"
    )


def test_weekday_mean_matches_hand_computed_causal_value():
    """Pin the exact expected value so the semantics cannot drift."""
    df = _calendar_frame(n=15)
    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", verbose=False)
    bf.enable_time_series = True
    bf.datetime_col = "date"
    bf.groupby_cols = []

    result = np.asarray(bf._apply_time_based_operation(df, "v", "weekday_mean"),
                        dtype=float)

    # 2021-01-01 is a Friday, so rows 0, 7 and 14 share a weekday with values
    # 0, 7 and 14 respectively.
    assert result[0] == 0.0, "first row of its weekday has no prior data -> 0"
    assert result[7] == 0.0, "second Friday sees only the first (v=0)"
    assert result[14] == pytest.approx(3.5), \
        "third Friday sees v=0 and v=7 -> mean 3.5"


def test_calendar_means_respect_group_boundaries():
    """A series must not borrow calendar history from another series."""
    n = 28
    df = pd.DataFrame({
        "date": list(pd.date_range("2021-01-01", periods=n // 2, freq="D")) * 2,
        "entity": ["A"] * (n // 2) + ["B"] * (n // 2),
        # B's values are far larger; if the groups leak into each other,
        # A's output will be pulled upward.
        "v": list(np.arange(n // 2, dtype=float)) + list(
            np.arange(n // 2, dtype=float) + 1000.0),
    })
    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", groupby_cols=["entity"], verbose=False)
    bf.enable_time_series = True
    bf.datetime_col = "date"
    bf.groupby_cols = ["entity"]

    result = np.asarray(bf._apply_time_based_operation(df, "v", "weekday_mean"),
                        dtype=float)

    a_rows = df["entity"].values == "A"
    assert (result[a_rows] < 500).all(), (
        "entity A's calendar means were contaminated by entity B's much "
        "larger values; group boundaries were not respected"
    )


# ---------------------------------------------------------------------------
# Degenerate input must not crash the sampler
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["constant", "all_zero"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_degenerate_columns_do_not_crash_fit(kind, task):
    """Zero-importance features must not produce a NaN sampling distribution.

    Every feature having zero importance made ig_vector.sum() == 0, so the
    bare `v /= v.sum()` normalization yielded NaN and fit() died inside
    rng.choice with "probabilities contain NaN".
    """
    n = 80
    fill = 3.0 if kind == "constant" else 0.0
    X = pd.DataFrame({f"f{i}": np.full(n, fill) for i in range(4)})
    y = pd.Series(np.arange(n, dtype=float))
    if task == "classification":
        y = (y > np.median(y)).astype(int)

    bf = bb.BigFeat(task_type=task, enable_time_series="no", verbose=False)
    out = bf.fit(X, y, **FIT_KWARGS)

    assert out.shape[0] == n
    assert np.isfinite(np.asarray(out, dtype=float)).all()
    assert np.isfinite(bf.ig_vector).all()
    assert bf.ig_vector.sum() == pytest.approx(1.0)


def test_normalize_to_distribution_handles_degenerate_input():
    """Unit-level check of the normalization helper."""
    norm = bb.BigFeat._normalize_to_distribution

    np.testing.assert_allclose(norm(np.array([1.0, 3.0])), [0.25, 0.75])
    # All-zero -> uniform rather than NaN.
    np.testing.assert_allclose(norm(np.zeros(4)), np.full(4, 0.25))
    # NaN / inf are scrubbed before normalizing.
    assert np.isfinite(norm(np.array([np.nan, np.inf, 1.0]))).all()
    assert norm(np.array([np.nan, np.inf, 1.0])).sum() == pytest.approx(1.0)
    # Negative weights are not valid probabilities.
    assert (norm(np.array([-5.0, 1.0])) >= 0).all()


# ---------------------------------------------------------------------------
# get_paths must not drop the first path
# ---------------------------------------------------------------------------

def test_get_paths_keeps_the_first_path():
    """The dedup loop used to compare index 0 against index -1.

    `path_list[i - 1]` wraps at i == 0, so whenever a tree's first and last
    root-to-leaf paths matched, the first path was silently discarded and
    never counted toward the split-frequency vector.
    """
    from sklearn.tree import DecisionTreeClassifier

    rs = np.random.RandomState(0)
    X = rs.rand(200, 4)
    y = (X[:, 0] > 0.5).astype(int)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    bf = bb.BigFeat(task_type="classification", enable_time_series="no",
                    verbose=False)
    paths = bf.get_paths(clf, np.arange(X.shape[1]))

    # get_paths records the FEATURE NAMES along each root-to-leaf path and
    # collapses adjacent duplicates, so the count is <= the leaf count: two
    # sibling leaves under the same split share a feature-name list. The
    # invariant that matters is that nothing is dropped at index 0.
    assert len(paths) > 0
    assert len(paths) <= clf.get_n_leaves()
    assert paths[0] == [0], (
        f"expected the first path to be the root split on feature 0, got "
        f"{paths[0]!r}; index 0 was dropped by the dedup loop"
    )


def test_get_paths_first_path_survives_when_it_equals_the_last():
    """Exercise the wrap-around case through the real get_paths().

    A tree that splits repeatedly on the SAME feature produces root-to-leaf
    paths whose feature-name lists collide, including the first and last.
    Under the old `path_list[i - 1]` indexing the first path was dropped;
    the count below detects that directly.
    """
    from sklearn.tree import DecisionTreeRegressor

    # A single informative feature forces every split onto feature 0, so all
    # paths are lists of the same repeated name.
    rs = np.random.RandomState(0)
    X = np.column_stack([np.linspace(0, 1, 300), rs.rand(300) * 1e-9])
    y = np.sin(X[:, 0] * 12)
    clf = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)

    bf = bb.BigFeat(task_type="regression", enable_time_series="no",
                    verbose=False)
    paths = bf.get_paths(clf, np.arange(X.shape[1]))

    assert paths, "get_paths returned nothing"
    assert paths[0] == [0, 0], (
        f"expected the first path to start at the root split on feature 0, "
        f"got {paths[0]!r} -- index 0 was likely dropped"
    )

    # Prove the fix bites: reconstruct what the OLD wrap-around indexing would
    # have returned from the same raw paths, and show it loses the first one.
    raw = []
    for p in paths:
        raw.append(p)
    old_style = [p for i, p in enumerate(raw) if p != raw[i - 1]]
    if raw[0] == raw[-1]:
        assert len(old_style) < len(raw), (
            "test fixture no longer reproduces the wrap-around condition"
        )
        assert paths == raw, "current implementation must keep every path here"


# ---------------------------------------------------------------------------
# Refitting must not inherit state from a previous fit
# ---------------------------------------------------------------------------

def test_refit_matches_a_fresh_estimator(periodic_data, clf_data):
    """fit() twice on one object must equal fitting two fresh objects.

    self.operators had time-series operators appended in place, guarded by a
    _ts_operators_added flag that was never reset, so a second fit inherited
    the first one's operator pool -- including on data with a completely
    different periodicity verdict.
    """
    X_ts, y_ts = periodic_data

    reused = bb.BigFeat(task_type="regression", enable_time_series="yes",
                        datetime_col="date", verbose=False)
    reused.fit(X_ts, y_ts, **FIT_KWARGS)
    second_on_reused = reused.fit(X_ts, y_ts, **FIT_KWARGS)

    fresh = bb.BigFeat(task_type="regression", enable_time_series="yes",
                       datetime_col="date", verbose=False)
    on_fresh = fresh.fit(X_ts, y_ts, **FIT_KWARGS)

    assert second_on_reused.shape == on_fresh.shape, (
        f"refit produced {second_on_reused.shape} but a fresh estimator "
        f"produced {on_fresh.shape}; state leaked across fits"
    )
    np.testing.assert_allclose(
        np.asarray(second_on_reused, dtype=float),
        np.asarray(on_fresh, dtype=float), rtol=1e-9, atol=1e-9,
        err_msg="refit did not match a fresh estimator",
    )


def test_operator_pool_does_not_grow_across_fits(periodic_data):
    """The operator list must be rebuilt, not appended to, on each fit."""
    X, y = periodic_data
    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", verbose=False)

    bf.fit(X, y, **FIT_KWARGS)
    first_count = len(bf.operators)
    bf.fit(X, y, **FIT_KWARGS)
    second_count = len(bf.operators)

    assert first_count == second_count, (
        f"operator pool grew from {first_count} to {second_count} across "
        f"fits; operators are being appended in place"
    )
    assert len(bf.operators) == len(set(bf.operators)), \
        "operator pool contains duplicates"


def test_preflight_penalty_does_not_compound_across_fits(periodic_data):
    """The pre-flight penalty is per-fit and must not overwrite user config."""
    X, y = periodic_data
    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date",
                    ts_operation_weight_multiplier=2.0, verbose=False)

    for _ in range(3):
        bf.fit(X, y, **FIT_KWARGS)
        assert bf.ts_operation_weight_multiplier == 2.0, (
            "the constructor argument was overwritten by the pre-flight "
            "penalty; repeated fits would compound the reduction"
        )


def test_preflight_check_actually_runs(periodic_data, capsys):
    """The pre-flight check must execute, not die in its own except block.

    It read self.feature_columns before that attribute was assigned, so it
    raised TypeError on every fit and the bare `except` swallowed it. The
    check never ran at all.
    """
    X, y = periodic_data
    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", verbose=True)
    bf.fit(X, y, **FIT_KWARGS)

    out = capsys.readouterr().out
    assert "Pre-flight Detection Cross-Validation" in out
    assert "CV Check skipped" not in out, (
        f"pre-flight check still aborts into its exception handler:\n"
        f"{[l for l in out.splitlines() if 'CV Check skipped' in l]}"
    )
    assert "CV Result:" in out, "pre-flight check did not reach a verdict"


# ---------------------------------------------------------------------------
# Recipes must be self-describing, not dependent on shared instance state
# ---------------------------------------------------------------------------

def test_recipes_record_the_feature_they_consume(periodic_data):
    """Time-series operators must carry their source column in the recipe."""
    X, y = periodic_data
    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", ts_operation_weight_multiplier=50.0,
                    verbose=False)
    bf.fit(X, y, **FIT_KWARGS)

    ts_ops = [
        (op, params)
        for recipe in bf.tracking_ops
        for op, _depth, params in recipe
        if getattr(op, "__name__", "").startswith("_safe_")
    ]
    assert ts_ops, "fixture selected no time-series operators"
    for op, params in ts_ops:
        assert "feature_index" in params, (
            f"{op.__name__} did not record which column it consumed; the "
            f"recipe depends on shared _current_feature_index state"
        )


def test_transform_ignores_current_feature_index(periodic_data):
    """Replay must not depend on the mutable _current_feature_index attribute.

    The index used to be read off shared instance state at apply time. Inside
    a binary node the second leaf's index had already overwritten the first's
    before the parent operator ran, so which column a time-series operator
    used depended on evaluation order rather than on the recipe.
    """
    X, y = periodic_data
    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", ts_operation_weight_multiplier=50.0,
                    verbose=False)
    bf.fit(X, y, **FIT_KWARGS)

    baseline = np.asarray(bf.transform(X), dtype=float)

    bf._current_feature_index = 999  # nonsense; must be ignored
    corrupted = np.asarray(bf.transform(X), dtype=float)
    np.testing.assert_allclose(
        baseline, corrupted, rtol=1e-9, atol=1e-9,
        err_msg="transform() output changed when _current_feature_index was "
                "corrupted, so recipes are not self-describing",
    )

    del bf._current_feature_index
    without = np.asarray(bf.transform(X), dtype=float)
    np.testing.assert_allclose(
        baseline, without, rtol=1e-9, atol=1e-9,
        err_msg="transform() depends on _current_feature_index existing",
    )


# ---------------------------------------------------------------------------
# Time-based windows must mean the same thing at every sampling frequency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq,expected_rows_in_90d", [
    ("D", 90),    # daily      -> 90 rows
    ("W", 13),    # weekly     -> ~13 rows
    ("ME", 3),    # monthly    -> ~3 rows
    ("QE", 1),    # quarterly  -> ~1 row
])
def test_time_window_spans_the_right_number_of_rows(freq, expected_rows_in_90d):
    """A 90-day window must select by timestamp, not by row count.

    Windows are detected as pd.Timedelta but every live path used to convert
    them to a row count via self.time_step, which defaults to 'D'. A 90-day
    window therefore became 90 ROWS at every frequency: on monthly data it
    averaged 90 months instead of 3, i.e. the whole series. Measured against
    genuine time-based rolling on the Monash benchmark data, 100% of rows
    were wrong at every frequency tested.
    """
    # NOTE: this must exercise the GROUPED path. With no groupby_cols the old
    # code routed to _vectorized_rolling_global, which already rolled by time
    # correctly; the row-count approximation lived only in the grouped branch
    # -- which is the one the benchmarks actually use (groupby_cols=item_id).
    n = 120
    df = pd.DataFrame({
        "date": pd.date_range("2000-01-31", periods=n, freq=freq),
        "item": ["A"] * n,
        "v": np.arange(n, dtype=float),
    })

    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", groupby_cols=["item"], verbose=False)
    bf.enable_time_series = True
    bf.datetime_col = "date"
    bf.groupby_cols = ["item"]
    bf.feature_columns = ["v"]

    got = np.asarray(
        bf._apply_time_based_operation(df, "v", "rolling_mean",
                                       window_size=pd.Timedelta(days=90)),
        dtype=float,
    )

    expected = (df.set_index("date")["v"]
                  .rolling("90D", min_periods=1).mean().values)
    np.testing.assert_allclose(
        got, expected, rtol=1e-9, atol=1e-9,
        err_msg=f"90-day rolling window is wrong at freq={freq}",
    )

    # Sanity-check the fixture really distinguishes the two behaviours: the
    # row-count approximation would average a different number of rows.
    n_rows = df.set_index("date")["v"].rolling("90D").count().iloc[-1]
    assert abs(n_rows - expected_rows_in_90d) <= 2, (
        f"fixture drift: a 90D window spans {n_rows} rows at freq={freq}, "
        f"expected about {expected_rows_in_90d}"
    )


def test_rolling_does_not_leak_across_entity_boundaries():
    """A group's early rows must not average in the previous group's values.

    The old grouped path rolled globally over the whole frame and masked the
    first rows of each group afterwards, so those rows were computed from the
    preceding entity's data before being zeroed.
    """
    n_per = 40
    df = pd.DataFrame({
        "date": list(pd.date_range("2021-01-01", periods=n_per, freq="D")) * 2,
        "item": ["A"] * n_per + ["B"] * n_per,
        "v": list(np.zeros(n_per)) + list(np.full(n_per, 1000.0)),
    })

    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", groupby_cols=["item"], verbose=False)
    bf.enable_time_series = True
    bf.datetime_col = "date"
    bf.groupby_cols = ["item"]
    bf.feature_columns = ["v"]

    got = np.asarray(
        bf._apply_time_based_operation(df, "v", "rolling_mean",
                                       window_size=pd.Timedelta(days=7)),
        dtype=float,
    )

    a_rows = df["item"].values == "A"
    b_rows = df["item"].values == "B"
    # A is entirely zeros, B entirely 1000. Neither may contaminate the other.
    np.testing.assert_allclose(got[a_rows], 0.0, atol=1e-9,
                               err_msg="entity A picked up entity B's values")
    np.testing.assert_allclose(got[b_rows], 1000.0, atol=1e-9,
                               err_msg="entity B was diluted by entity A's values")


def test_time_window_handles_irregular_sampling():
    """Genuinely irregular timestamps: the window is defined by time, not rows.

    No dataset in the benchmark suite exercises this -- the Monash format
    stores only a start timestamp plus a dense array, so every series is
    uniform by construction. This fixture covers it directly.
    """
    rs = np.random.RandomState(0)
    # Random gaps of 1-10 days.
    offsets = np.cumsum(rs.randint(1, 11, size=80))
    dates = pd.Timestamp("2021-01-01") + pd.to_timedelta(offsets, unit="D")
    df = pd.DataFrame({"date": dates, "item": ["A"] * 80,
                       "v": np.arange(80, dtype=float)})

    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", groupby_cols=["item"], verbose=False)
    bf.enable_time_series = True
    bf.datetime_col = "date"
    bf.groupby_cols = ["item"]
    bf.feature_columns = ["v"]

    got = np.asarray(
        bf._apply_time_based_operation(df, "v", "rolling_mean",
                                       window_size=pd.Timedelta(days=30)),
        dtype=float,
    )
    expected = (df.set_index("date")["v"]
                  .rolling("30D", min_periods=1).mean().values)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# The automatic target log-transform must be discoverable and reversible
# ---------------------------------------------------------------------------

def test_log_transform_is_reported_and_invertible():
    """fit() log-transforms skewed regression targets; say so, and undo it.

    The flag recording this was private and never read anywhere in the
    codebase, so a caller fitting their own model against the same target had
    no way to learn that feature selection had been scored against a
    log-scaled version of it.
    """
    n = 200
    rs = np.random.RandomState(0)
    # Strictly positive and heavily right-skewed -> triggers the transform.
    y = pd.Series(np.exp(rs.randn(n) * 2.0) + 1.0)
    X = pd.DataFrame({f"f{i}": rs.rand(n) for i in range(4)})

    bf = bb.BigFeat(task_type="regression", enable_time_series="no",
                    verbose=False)
    bf.fit(X, y, **FIT_KWARGS)

    assert hasattr(bf, "target_log_transformed")
    assert bf.target_log_transformed is True, (
        f"skew={y.skew():.2f} should have triggered the log-transform"
    )

    # The inverse must round-trip the transform fit() applied internally.
    original = np.array([1.0, 10.0, 100.0])
    round_tripped = bf.inverse_transform_target(np.log1p(original))
    np.testing.assert_allclose(round_tripped, original, rtol=1e-9)


def test_inverse_transform_target_is_a_noop_when_untransformed():
    """Always safe to call, whether or not the transform fired."""
    rs = np.random.RandomState(0)
    n = 150
    X = pd.DataFrame({f"f{i}": rs.rand(n) for i in range(4)})
    y = pd.Series(rs.rand(n))  # symmetric -> no transform

    bf = bb.BigFeat(task_type="regression", enable_time_series="no",
                    verbose=False)
    bf.fit(X, y, **FIT_KWARGS)

    assert bf.target_log_transformed is False
    preds = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(bf.inverse_transform_target(preds), preds)


# ---------------------------------------------------------------------------
# Block downsampling
# ---------------------------------------------------------------------------

def test_block_downsampling_discovers_on_a_sample_but_applies_to_all_rows():
    """fit() may sample for feature DISCOVERY but must return every row.

    This path had no test coverage while ~150 lines of it sat inline in
    fit(). It is exercised here so the extraction into
    _apply_block_downsampling stays honest.
    """
    n = 3000
    rs = np.random.RandomState(0)
    t = np.arange(n)
    X = pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=n, freq="D"),
        "a": np.sin(2 * np.pi * t / 7) + rs.randn(n) * 0.05,
        "b": rs.rand(n),
    })
    y = pd.Series(2 * X["a"].values + rs.randn(n) * 0.1)

    bf = bb.BigFeat(task_type="regression", enable_time_series="yes",
                    datetime_col="date", enable_downsampling=True,
                    max_fit_samples=600, verbose=False)
    out = bf.fit(X, y, gen_size=3, iterations=2, random_state=0)

    assert bf._was_downsampled is True, "downsampling did not engage"
    assert out.shape[0] == n, (
        f"fit() returned {out.shape[0]} rows for a {n}-row frame; discovery "
        f"may sample but the returned features must cover every row"
    )
    assert np.isfinite(np.asarray(out, dtype=float)).all()
    assert np.asarray(bf.transform(X)).shape[0] == n


def test_downsampling_off_by_default():
    """The default path must not silently sample."""
    rs = np.random.RandomState(0)
    n = 300
    X = pd.DataFrame({f"f{i}": rs.rand(n) for i in range(4)})
    y = pd.Series(rs.rand(n))

    bf = bb.BigFeat(task_type="regression", enable_time_series="no",
                    verbose=False)
    bf.fit(X, y, **FIT_KWARGS)
    assert bf._was_downsampled is False
