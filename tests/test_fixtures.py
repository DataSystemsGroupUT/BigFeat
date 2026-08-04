"""Guard tests: confirm each fixture still reaches the branch it was built for.

These do not test correctness. They exist so that if detector tuning changes
which branch a fixture lands in, this fails loudly -- rather than the
behavioural tests silently drifting to cover something else.
"""
import bigfeat.bigfeat_base as bb
from conftest import FIT_KWARGS


def _fit(X, y, **kw):
    bf = bb.BigFeat(verbose=False, **kw)
    bf.fit(X, y, **FIT_KWARGS)
    return bf


def test_clf_fixture_disables_time_series(clf_data):
    X, y = clf_data
    bf = _fit(X, y, task_type="classification", enable_time_series="auto")
    assert bf.enable_time_series is False, "no datetime column -> TS must stay off"


def test_reg_fixture_disables_time_series(reg_data):
    X, y = reg_data
    bf = _fit(X, y, task_type="regression", enable_time_series="auto")
    assert bf.enable_time_series is False


def test_periodic_fixture_reaches_pooled_ensemble(periodic_data):
    X, y = periodic_data
    bf = _fit(X, y, task_type="regression", enable_time_series="auto",
              datetime_col="date")
    assert bf.enable_time_series is True
    assert bf.detection_strategy == "pooled_ensemble", (
        f"periodic fixture no longer reaches pooled_ensemble "
        f"(got {bf.detection_strategy!r}); retune the fixture"
    )


def test_nonstationary_fixture_triggers_stationarity_gate(nonstationary_data):
    X, y = nonstationary_data
    bf = _fit(X, y, task_type="regression", enable_time_series="auto",
              datetime_col="date")
    assert bf.enable_time_series is True
    assert bf.avg_lag1 > 0.85, (
        f"non-stationary fixture has avg_lag1={bf.avg_lag1:.3f}, "
        f"no longer trips the > 0.85 gate"
    )
    # Restricted mode narrows the operator pool to lag/diff-style ops only.
    names = {op.__name__ for op in bf.time_series_operators}
    assert names <= {"_safe_lag_feature", "_safe_diff_feature",
                     "_safe_pct_change"}, f"unexpected restricted pool: {names}"
