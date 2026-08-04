"""Golden-output characterization tests.

These pin the *current* feature matrices so that unintended behaviour drift
during refactoring is caught. They are deliberately not a correctness claim:
some of these numbers are produced by code paths with known bugs. When a fix
intentionally changes output, regenerate with:

    python -m pytest tests/test_golden.py --regen-golden

and review the diff in the commit.

Baselines are only meaningful because random_state now fully determines
fit() output (see the Phase 1 RNG fix); they were captured after it.
"""
import hashlib
import json
import pathlib

import numpy as np
import pytest

import bigfeat.bigfeat_base as bb
from conftest import FIT_KWARGS

GOLDEN_PATH = pathlib.Path(__file__).parent / "golden.json"

CASES = {
    "clf_nots": (dict(task_type="classification", enable_time_series="no"),
                 "clf_data", {}),
    "reg_nots": (dict(task_type="regression", enable_time_series="no"),
                 "reg_data", {}),
    "reg_ts_periodic": (dict(task_type="regression", enable_time_series="yes",
                             datetime_col="date"), "periodic_data", {}),
    "reg_ts_nonstationary": (dict(task_type="regression",
                                  enable_time_series="auto",
                                  datetime_col="date"),
                             "nonstationary_data", {}),
}


def _digest(arr):
    """Shape + a stable hash, so goldens stay small and diffs stay readable."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    # Round before hashing: tiny platform-level FP noise should not flip the
    # digest, but any real behavioural change moves values far more than this.
    rounded = np.round(a, 6) + 0.0  # +0.0 normalises -0.0 to 0.0
    return {
        "shape": list(a.shape),
        "sha256": hashlib.sha256(rounded.tobytes()).hexdigest(),
        "sum": round(float(np.nansum(rounded)), 4),
    }


def _run(kwargs, X, y, fit_extra):
    bf = bb.BigFeat(verbose=False, **kwargs)
    fitted = bf.fit(X, y, **{**FIT_KWARGS, **fit_extra})
    return {
        "fit": _digest(fitted),
        "transform": _digest(bf.transform(X)),
        "n_features": int(fitted.shape[1]),
        "time_series_enabled": bool(bf.enable_time_series),
        "detection_strategy": getattr(bf, "detection_strategy", None),
    }


def _load():
    if not GOLDEN_PATH.exists():
        return {}
    return json.loads(GOLDEN_PATH.read_text())


@pytest.mark.parametrize("name", list(CASES))
def test_matches_golden(name, request, regen_golden):
    kwargs, fixture_name, fit_extra = CASES[name]
    X, y = request.getfixturevalue(fixture_name)
    actual = _run(kwargs, X, y, fit_extra)

    if regen_golden:
        golden = _load()
        golden[name] = actual
        GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"regenerated golden for {name}")

    golden = _load()
    if name not in golden:
        pytest.fail(
            f"no golden baseline for {name!r}; run "
            f"`pytest tests/test_golden.py --regen-golden` to create it"
        )

    expected = golden[name]
    # Compare shape first: it gives a far clearer failure than a hash mismatch.
    assert actual["fit"]["shape"] == expected["fit"]["shape"], (
        f"fit() output shape changed for {name!r}: "
        f"{expected['fit']['shape']} -> {actual['fit']['shape']}"
    )
    assert actual == expected, (
        f"behaviour changed for {name!r}.\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        f"If this change is intentional, regenerate with --regen-golden "
        f"and explain the diff in the commit message."
    )
