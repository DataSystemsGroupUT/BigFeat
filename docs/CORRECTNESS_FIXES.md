# Correctness Fixes — Time-Series Subsystem

This document records a correctness review of BigFeat's time-series feature
engineering, the defects it found, how each was verified, and what the changes
mean for previously collected benchmark results.

For how the library works — the generation loop, feature replay, the
time-series subsystem — see [ARCHITECTURE.md](ARCHITECTURE.md).

**Scope.** Twelve commits on `feature/time-series-ops`, starting from
`a9021c8` ("Time series major release"). The classic (non-time-series) BigFeat
core is largely untouched except where a defect was shared between both paths.

**Bottom line.** Six defects made output *wrong* rather than merely
suboptimal — including rolling windows that spanned the wrong time range on
100% of rows across all 25 benchmark datasets, and a periodicity vote that
fired on pure white noise. Any result collected before these fixes should be
treated as measuring different behaviour than its configuration describes.

---

## 1. Why results collected before this review need re-checking

Three defects invalidate assumptions a reader would reasonably make about
earlier numbers:

| Assumption | Reality before the fix |
|---|---|
| "`random_state=0` makes runs reproducible" | The sampler drew from the global NumPy RNG. Five runs at one seed gave four distinct outputs; five *different* seeds gave byte-identical output. `random_state` was very nearly inert. |
| "A 90-day rolling window covers 90 days" | It covered 90 **rows**. On monthly data that is 90 months — the entire series. Wrong on 100% of rows on every dataset tested. |
| "The pre-flight check guards against hallucinated seasonality" | It read `self.feature_columns` before that attribute was assigned, raised `TypeError` on every fit, and a bare `except` swallowed it. It had never executed. |

None of these produce visibly broken output. They produce plausible numbers
that do not mean what the configuration says they mean.

---

## 2. Defects found and fixed

Each entry gives the observable symptom, the mechanism, and how the fix was
verified. Commit hashes are on `feature/time-series-ops`.

### 2.1 `random_state` did not control the output (`9bf7fd5`)

`get_feature_importances` sampled rows with `np.random.choice` — the global
RNG — rather than `self.rng`.

*Measured before:* five runs at `random_state=0` produced four distinct
outputs. Seeds `0, 1, 2, 5, 1234` produced byte-identical output.

*Fix:* draw from `self.rng`, which is seeded from `random_state`.

*Verified:* with the global RNG deliberately perturbed between runs, a fixed
seed now yields identical output, and each of the five seeds yields different
output. Confirmed on both the time-series and non-time-series paths.

### 2.2 `transform()` corrupted features for reordered input (`8253024`)

Three coupled defects, each masking the next:

1. `_prepare_time_series_data` overwrote the datetime and groupby columns from
   `self.original_data` **positionally**. Positional alignment against the
   training frame is only meaningful when the incoming rows *are* the training
   rows in training order. For any other order this paired each row with
   another row's timestamp.
2. A `self._is_sorted` flag short-circuited the whole function. Set on first
   use during `fit` and reset only at the top of `fit`, so every subsequent
   `transform` returned `X` untouched — skipping the datetime sort, the dtype
   coercion, and the `_original_index` bookkeeping the rest of `transform`
   depends on.
3. The `fAnova` branch sat *after* `transform`'s time-series early-return,
   making it unreachable whenever time series was enabled: `fit` applied
   `SelectKBest` and returned *k* columns while `transform` returned the full
   width.

*Measured before:* on a 300-row periodic series, shuffling the input rows
changed 25% of output elements (max absolute difference 2.06), and features
were inconsistent with their own timestamps — correlation between the correct
and actual values was 0.04. With `selection='fAnova'` and time series enabled,
`fit` returned 3 columns and `transform` returned 6.

*Note on detection.* Comparing two consecutive `transform` calls does **not**
catch this: the flag stays set for every call, so all of them are
*consistently* wrong. Row-order invariance is the observable that breaks.

### 2.3 Look-ahead leakage in the calendar-mean operators (`1d90577`)

`weekday_mean` and `month_mean` used `groupby(calendar_key).transform('mean')`,
which averages the **entire** column within each weekday or month. A row's
value therefore depended on rows dated after it — at `transform` time, on rows
from the future relative to the point being predicted.

*Fix:* `shift(1).expanding().mean()` within each calendar group, so a row sees
only earlier rows and never itself.

*Note.* The leaky implementation existed in **two** places — the grouped path
in `_apply_time_based_operation` and the datetime-indexed path in
`_apply_single_group_operation`, reached when `groupby_cols` is empty. Fixing
only the first left the default configuration untouched.

*Verified:* zero leaking rows across all four operator code paths plus the
public `_safe_*` wrapper, where the previous implementation leaked on every
row.

### 2.4 Degenerate input crashed `fit()` (`1d90577`)

`ig_vector` and `split_vec` were normalized with a bare `v /= v.sum()`. When
every feature has zero importance — as for constant or all-zero columns — the
sum is 0, the division yields NaN, and `fit` died inside `rng.choice` with
`ValueError: probabilities contain NaN`. Reproduced on 12 of 30 degenerate
input combinations.

*Fix:* `_normalize_to_distribution` scrubs NaN/inf, clips negatives, and falls
back to a uniform distribution when there is no signal.

### 2.5 `get_paths` dropped its first path (`1d90577`)

The dedup loop compared `path_list[i]` against `path_list[i - 1]`, which at
`i == 0` wraps to the **last** element. Whenever a tree's first and last
root-to-leaf paths matched, the first was discarded and never counted toward
the split-frequency vector.

### 2.6 The pre-flight seasonality check had never run (`254e04c`)

Three coupled defects in the block that exists to catch hallucinated
seasonality:

1. **It never executed.** It read `self.feature_columns`, assigned *after* the
   block, so it raised `TypeError` on every fit and the bare `except`
   swallowed it.
2. **Its test features were leaky.** Each side rolled independently, so the
   window restarted at the start of the test slice. The code carried a comment
   acknowledging this (*"Rolling on test is leaky"*) and did it anyway.
3. **Its penalty compounded.** On weak improvement it overwrote
   `self.ts_operation_weight_multiplier` — the constructor argument — so
   repeated fits kept halving it, with no way to recover short of rebuilding
   the object.

*Effect of the fix:* the check now runs. On the periodic test fixture it
measures −0.3% improvement from TS features and halves the operator weight —
its documented job, executing for the first time.

### 2.7 State leaked between `fit()` calls (`254e04c`)

`self.operators` and `self.unary_operators` were extended in place with the
time-series operators (and filtered in restricted mode), guarded by a
`_ts_operators_added` flag that was never cleared. A second `fit` inherited the
first's operator pool even when the periodicity verdict differed. Now rebuilt
from base definitions on every fit via `_reset_fit_state()`.

### 2.8 Recipes depended on shared mutable state (`0524134`)

Time-series operators resolved their source column by reading
`self._current_feature_index`, assigned as each leaf of the expression tree was
resolved. But the operator reads it *later*, when its parent node fires. Inside
a binary node with two different leaves, the second leaf's index had already
overwritten the first's — so which column an operator used depended on
evaluation order rather than on the recipe. Reachable in practice: a depth-3
recipe applied TS operators against two different source columns.

*Fix:* the consumed column is recorded in the operator's `params` dict at
generation time, which already persists into `transform`.

*Verified:* `transform` output is byte-identical after corrupting
`_current_feature_index` to a nonsense value and after deleting it entirely.

### 2.9 Time windows were row counts, not time (`7c6c705`)

The most consequential defect. Window sizes are detected as `pd.Timedelta`, but
every live rolling path converted them to a row count via
`_estimate_window_rows`, which used `self.time_step` (default `'D'`). A 90-day
window became 90 **rows** regardless of sampling.

Measured against genuine time-based rolling on the Monash data, using the
grouped path the benchmarks use (`groupby_cols=['item_id']`):

| Dataset | Freq | mean abs error | Rows a 90-day window *should* span |
|---|---|---|---|
| m1_monthly | M | 72,057 | 3 |
| tourism_quarterly | Q | 110,185 | 1 |
| m1_yearly | Y | 1,463,465 | 1 |
| electricity_weekly | W | 48,889 | 12 |
| nn5_daily | D | 16.7 | 40.5 |

**100% of rows were wrong on every dataset.** In every case the approximation
used 90 rows. Thirteen of the 25 benchmark datasets are monthly, quarterly or
yearly, where calendar units are not fixed durations (28–31 day months, 90–92
day quarters, 365–366 day years) — exactly where a row count cannot be correct.

*Fix:* pass the `Timedelta` to pandas, which selects rows by timestamp. All
five frequencies now agree exactly with true time-based rolling (error 0.0000,
0% of rows wrong).

*Second defect in the same path:* rolling was computed **globally** across the
whole frame and each group's first rows masked afterwards, so those rows
averaged in the preceding entity's values before being zeroed. Grouping now
happens before rolling.

*Performance:* ~5× slower on small frames (1.3 ms → 6.8 ms at ~900 rows), but
per-row cost falls with size because the overhead is per-group: 1.62 µs/row at
8k rows, 0.20 µs/row at 200k rows (200,000 rows in 39.6 ms).

### 2.10 Detector confidence metrics were broken (`0abee86`)

Two of the three detectors were wrong in opposite directions, on data where the
answer is unambiguous.

**ACF reported white noise as periodic** (confidence 0.973). `_compute_acf` ran
up to lag `len(series)-1`. At lag 363 of a 365-sample series only 2 points
overlap, and `np.corrcoef` of two points is always exactly ±1. Measured on
white noise: max |ACF| was 0.158 for lags ≤ 180 (correct) but **1.000** for
lags > 300. Those spurious unit correlations sorted to the top by height and
became both the reported period and the confidence.

**DFT reported a clean 7-day sine as NOT periodic** (0.244). Its confidence was
`1 - sorted[1]/sorted[0]` — close to the opposite of what it claimed. `np.sort`
places the two largest bins adjacent, and for a real peak those are
neighbouring bins of the *same* peak split by spectral leakage, so strong
periodicity drove the score toward zero. On noise it scored 0.075 — only 0.17
apart, both below the 0.3 threshold.

| | Clean 7-day sine | White noise |
|---|---|---|
| DFT before | `False` (0.244) ✗ | `False` (0.075) |
| DFT after | `True` (0.628) ✓ | `False` (0.263) |
| ACF before | `True` (0.973) | `True` (0.973) ✗ |
| ACF after | `True` (0.996) ✓ | `False` (0.000) ✓ |

**End-to-end effect.** On pure white noise the default `'auto'` ensemble
previously **enabled** time-series features with windows of 163–363 days —
precisely the hallucinated seasonality the consensus vote exists to prevent. It
now correctly disables them, while still enabling on genuinely periodic data.

**ACF unit confusion.** `max_window_days`, a *duration*, was used directly as a
lag *count*, so at hourly sampling a 365-day ceiling became 365 hours (~15
days).

### 2.11 Sampling rate was assumed to be daily (`44a1eab`)

Detectors work in samples and convert to days via `sampling_rate`. The `'yes'`
path passed `self.time_step`; the **ensemble path used by the default `'auto'`
mode passed nothing at all**, so it defaulted to `'D'`. Monthly and quarterly
observations were treated as one-day samples and every detected period came out
30–90× too small.

*Fix:* `infer_sampling_rate` measures median timestamp spacing and snaps it to
a frequency alias. Verified to recover M/Q/Y/D correctly on the Monash data.

### 2.12 Pooled windows were truncated to the *shortest* (`44a1eab`)

The ensemble sorted pooled candidates ascending and took the first
`n_windows` — discarding every long window. Pooling three detectors reliably
produces more than `n_windows` candidates, so the long scales were dropped
every time. Now samples at even quantiles.

Combined with 2.11, this was still producing 1–6 day windows for monthly series
*after* the detectors were fixed. On `m1_monthly` the selected windows go from
`[1,2,3,4,5,6]` to `[1,3,5,90,182,365]` days.

### 2.13 A hidden, un-invertible target transform (`04834ae`)

`fit` silently log-transforms strictly positive regression targets with skew
> 2 before scoring feature importances. The flag recording this was private and
**never read anywhere in the codebase**, including the benchmark harness, so a
caller training their own model had no way to learn that feature selection had
been scored against a log-scaled target.

Now public (`target_log_transformed`) and paired with
`inverse_transform_target()`, a no-op when the transform did not fire. Note the
transform rebinds a local only — the caller's `y` array was never modified.

---

## 3. Two claims that did NOT survive verification

Recorded because they shaped decisions, and because both would have caused harm
if acted on.

### 3.1 The `np.subtract` replay "bug" is not a bug

Initial analysis flagged the operand swap at `feat_with_depth_gen`
(`op(feat_2, feat_1)`) as an asymmetry that would replay a different expression
than was fitted.

**Refuted.** 1500 randomly generated trees were compared symbolically and
numerically: zero mismatches. Two reversals cancel exactly — `op_ls`/`feat_ls`
are pushed post-order but popped LIFO, which yields a mirrored traversal, and
the operand swap un-mirrors it.

The swap is **load-bearing**. Changing it to the "natural-looking"
`op(feat_1, feat_2)` breaks **1070 of 1500** trees:

```
fit   : subtract(absolute(x2), subtract(x1, x2))
replay: subtract(subtract(x2, x1), absolute(x2))
```

Do not "clean this up." A property test guards it.

### 3.2 `_apply_single_group_operation` is not dead code

Static analysis called it unreachable. Instrumenting every operator across both
the grouped and ungrouped paths showed it is called 4 times, serving the
calendar and seasonal operators on the no-groups branch. Deleting it would have
broken `weekday_mean`, `month_mean`, `seasonal_decompose` and `trend` for the
default configuration.

---

## 4. Benchmark impact

A controlled A/B was run on 12 Monash datasets: the original code
(`a9021c8`) against the fixed code, on byte-identical train/test splits with
the same seed and the same downstream estimator. Baseline drift between the two
runs was 3.6e-15, confirming only the library differed.

| Comparison | n | geo-mean new/orig | Better |
|---|---|---|---|
| All 12 | 12 | 1.149 | 2/12 |
| Excluding `covid_deaths` | 11 | **1.001** | 2/11 |
| Calendar freq (M/Q/Y) | 8 | **0.999** | 1/8 |

**Accuracy is essentially unchanged.** This was not the expected result — the
window fixes were predicted to show clear gains on calendar-frequency data.
Runtime improved 2.8× (358 s → 129 s).

### The `covid_deaths` regression is real and explained

MASE went 0.479 → 2.532, the only dataset worse than baseline. Traced rather
than dismissed:

- Both versions take the same branch with identical `avg_lag1` (0.725)
- The original selected `absolute` — a passthrough that ignores time entirely
- The fixed version selects `_safe_ewm`
- On cumulative counts, EWM is measurably **3× worse** than the raw previous
  value (MAE 11.7 vs 4.0)

So the corrected machinery genuinely picked a time-series operator where the
old code, by accident of its broken windows, picked a no-op. The fix works as
designed; the *operator choice* is wrong for trending data.

**This exposes an open issue** (see §6): `avg_lag1 = 0.725` sits below the 0.85
stationarity gate, so strongly-trending data is not routed to restricted mode.
The threshold was deliberately **not** changed — tuning it on one dataset's
evidence would be unprincipled.

### Caveats

12 datasets, ≤25 series each, ≤120 rows, single seed. **Directional, not
publication-grade.** A full run of `testing/Benchmarking/benchmark.py` across
all 25 datasets and all method combinations takes roughly 172 hours based on
the wall times in the committed results.

---

## 5. Reproducing the analysis

```bash
pip install -r requirements.txt          # library deps
pip install -r requirements-benchmark.txt  # + benchmark deps
pip install -e .

pytest                     # full suite, ~85 s
pytest -m "not slow"       # fast loop, ~58 s
```

The full benchmark, scoped:

```bash
python testing/Benchmarking/benchmark.py \
    --datasets m1_monthly tourism_quarterly \
    --methods baseline auto_ensemble \
    --output-dir ./benchmark_results
```

---

## 6. Open issues

**The stationarity gate may be too permissive.** `covid_deaths` has
`avg_lag1 = 0.725`, below the 0.85 threshold, so strongly-trending cumulative
data is not routed to restricted mode and smoothing operators get selected
where they hurt. Needs evidence across more trending datasets before the
threshold is moved.

**`local_utils.py` is ~690 unreferenced lines.** Roughly 40 technical-analysis
indicators (RSI, MACD, Bollinger, Aroon, Parabolic SAR, Fibonacci, …) were
written but never wired into the operator pool, which is fully specified at
`bigfeat_base.py:154`, `:156` and `_add_time_series_operators`. Only
`original_feat` is referenced — and it sits in `unary_operators` but not
`operators`, so it can never be sampled. Either connect them or remove them.

**`fit()` is 664 lines; `_setup_time_series` is 360.** Splitting them further
is mechanical work with real regression risk and no behavioural benefit; better
done against a specific need than speculatively.

**Benchmark datasets are ~2.3 GB in git history.** They are regenerable via
`testing/Benchmarking/datasets/download_datasets.py`. Several individual files
exceed GitHub's 100 MB limit.
