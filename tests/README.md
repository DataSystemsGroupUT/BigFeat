# BigFeat Test Suite

91 tests, ~85 s for the full run. Built as a safety net for the correctness
review documented in [`../docs/CORRECTNESS_FIXES.md`](../docs/CORRECTNESS_FIXES.md).

```bash
pip install -r ../requirements.txt
pip install -e ..
pip install pytest

pytest                  # full suite, ~85 s
pytest -m "not slow"    # fast loop while iterating, ~58 s
pytest tests/test_detectors.py -v
```

---

## Design: invariants first, goldens second

The suite is deliberately **invariant-led**. Golden outputs pin exact numbers
and break on every intentional change, which trains people to regenerate them
without reading the diff. Invariants state properties that must hold *whatever*
the numbers are, so they survive intentional changes while still catching the
state-leak and replay bugs that dominated this codebase.

This weighting was validated in practice: the `_is_sorted` fix
(`docs/CORRECTNESS_FIXES.md` §2.2) changed no golden at all — it only affects
reordered input — but the invariant tests caught it immediately.

### `test_invariants.py` — 43 tests

Properties that must hold across all 7 configurations (classification and
regression, TS on/off/auto, heavy-TS, fAnova):

| Invariant | Catches |
|---|---|
| `fit(X)` output == `transform(X)` on training data | Replay bugs, fAnova width mismatch, state leaks |
| Two consecutive `transform()` calls agree | Mutable state surviving a call |
| **`transform()` is invariant to input row order** | The `_is_sorted` leak |
| Fixed `random_state` → identical output | Global-RNG dependence |
| Row count preserved; output finite | Shape and NaN regressions |

**Read this if you add an invariant.** The row-order test is the one that
matters most, and the reason is instructive: checking that two consecutive
`transform()` calls agree does **not** catch the `_is_sorted` bug, because the
flag stayed set for every call, so all of them were *consistently* wrong.
Consistency-checking cannot detect stable corruption. Row order was the
observable that broke.

Two cases exist purely to reach code the defaults leave dormant:

- `reg_heavy_ts` — `ts_operation_weight_multiplier=50.0`, because with default
  weights a run may select only one weak TS operator, hiding the bug
- `reg_ts_fanova` — `selection='fAnova'`, required to reach the `SelectKBest`
  branch in `transform()`

Both are marked `slow`.

### `test_correctness.py` — 27 tests

One test per fixed defect, each naming the mechanism it pins. Grouped by
subject: calendar-mean leakage, degenerate-input crashes, `get_paths`,
cross-fit state, self-describing recipes, time-window semantics, the target
log-transform, and block downsampling.

### `test_detectors.py` — 13 tests

The central property: **a detector must say "periodic" for periodic data and
"not periodic" for noise.** All three failed that on one side or the other.
Also covers sampling-rate inference, ACF's short-overlap artifact, DFT's
confidence separation, and the end-to-end ensemble verdict.

### `test_fixtures.py` — 4 tests

Guards that each fixture still reaches the `_setup_time_series` branch it was
built for. If detector tuning changes which branch a fixture lands in, this
fails loudly rather than letting the behavioural tests silently drift to
covering something else.

### `test_golden.py` — 4 cases

Digest-based drift detection over four scenarios. Regenerate deliberately:

```bash
pytest tests/test_golden.py --regen-golden
```

Then **read the diff and explain it in the commit message.** Goldens changed
twice during the review, each time for a documented reason (see
`CORRECTNESS_FIXES.md` §2.6 and §2.10).

---

## Fixtures

Defined in `conftest.py`. Small by design — the suite should run in seconds.

| Fixture | Data | Branch reached |
|---|---|---|
| `clf_data` | Tabular, no datetime | TS disabled |
| `reg_data` | Tabular, no datetime | TS disabled |
| `periodic_data` | 7- and 30-day seasonality | `pooled_ensemble` |
| `nonstationary_data` | Random walk + weekly signal | Stationarity gate, `avg_lag1 > 0.85` |

A fifth branch, `trend_stationarity`, is reachable only with irregular
timestamps and is **seed-dependent** (2 of 7 seeds). It is not currently
fixtured; if you add it, pin the seed and assert on `detection_strategy`.

---

## Writing tests here: what this codebase taught us

**A passing test proves nothing until you have seen it fail for the right
reason.** Two concrete near-misses during the review:

1. The first version of the frequency tests **passed against the buggy code**.
   They were written without `groupby_cols`, which routes to a *different*
   implementation that was already correct. The bug lived only in the grouped
   path — the one the benchmarks use. Caught only by explicitly running the new
   tests against the old implementation.

2. A `git stash` A/B restored the new code before the tests ran, so it was
   testing new-against-new and reading it as a real result. Verify which code
   is actually on disk.

**Before committing a regression test, check it out against the pre-fix
commit and confirm it fails.** For example:

```bash
git stash                      # or: git worktree add /tmp/old <pre-fix-sha>
git checkout <pre-fix-sha> -- bigfeat/
pytest tests/test_correctness.py -k your_new_test   # must FAIL
git checkout HEAD -- bigfeat/
```

**Prefer differential assertions to pinned constants** where the property
allows it. `test_calendar_means_never_use_future_rows` uses a monotonically
increasing feature so that *any* average including future rows is strictly
greater than a causal one — detecting leakage by comparison rather than by
hard-coded numbers, which survives changes to the fill value or window.

---

## Known gaps

- **The goldens do not cover the calendar-mean operators.** None of the four
  pinned scenarios happens to select `weekday_mean`/`month_mean`, so the
  leakage fix moved no golden. Dedicated tests cover it, but be aware the
  goldens alone would not have caught it.
- **Downsampling has 2 tests and had none before.** The path is large
  (`_apply_block_downsampling`, 166 lines) relative to its coverage.
- **No irregular-sampling benchmark data exists.** The Monash format stores
  only a start timestamp plus a dense array, so every series is uniform by
  construction. `test_time_window_handles_irregular_sampling` covers it
  synthetically; the Lomb-Scargle detector, which exists specifically for
  irregular data, has no real dataset exercising its advantage.
- **Single-seed fixtures.** Nothing sweeps seeds, so seed-sensitive behaviour
  in detection could hide.
