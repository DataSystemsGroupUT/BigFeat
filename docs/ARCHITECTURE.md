# BigFeat Architecture

How the library actually works: what `fit()` does, how features are represented
and replayed, how time-series windows are chosen, and where the seams are.

Written for someone who needs to modify the code. For *using* BigFeat see the
[README](../README.md); for the correctness review and its findings see
[CORRECTNESS_FIXES.md](CORRECTNESS_FIXES.md).

---

## 1. What BigFeat does

BigFeat is an automated feature-engineering library. Given tabular `X, y`, it
synthesises new features by randomly composing arithmetic (and optionally
time-aware) operators into expression trees, keeping the ones a tree ensemble
ranks as important.

It follows the scikit-learn `fit`/`transform` contract but is **not** an sklearn
`BaseEstimator` — no `get_params`, no `TransformerMixin`, so it will not drop
into a `Pipeline` or `GridSearchCV` unwrapped.

```
X, y ──▶ fit()  ──▶ feature matrix + stored "recipes"
X_new ──▶ transform() ──▶ replays those recipes ──▶ feature matrix
```

The central idea: `fit` does not store feature *values*, it stores **recipes** —
the operators, source-column indices, and parameters needed to rebuild each
feature. `transform` replays them on new data.

---

## 2. Module map

| File | Lines | Responsibility |
|---|---|---|
| `bigfeat/bigfeat_base.py` | 3576 | The `BigFeat` class: `fit`, `transform`, the generation loop, the 17 time-series operators, the time-series machinery, block downsampling |
| `bigfeat/window_detector_base.py` | 269 | `BaseWindowDetector`: shared detector behaviour — datetime-column detection, sampling-rate inference, sample→day conversion, window ladders, the periodicity verdict |
| `bigfeat/dft_window_detector.py` | 379 | Fourier-based periodicity detection |
| `bigfeat/acf_window_detector.py` | 349 | Autocorrelation-based detection |
| `bigfeat/lomb_scargle_window_detector.py` | 302 | Lomb-Scargle detection (for irregular sampling) |
| `bigfeat/local_utils.py` | 712 | **Almost entirely unused** — see §8 |

---

## 3. The feature representation

This is the concept everything else hangs off.

A generated feature is an **expression tree**. `mul(abs(x2), sub(x1, x2))` is a
depth-2 tree with three operator nodes and three leaves. Each tree is stored as
three parallel arrays, one entry per generated feature:

| Attribute | Contents |
|---|---|
| `self.tracking_ops[i]` | List of `(operator, depth, params)` tuples, in post-order |
| `self.tracking_ids[i]` | List of source-column indices (the leaves) |
| `self.feat_depths[i]` | The tree's depth |

Plus `self.scaler` (a fitted `RobustScaler`) and, for `selection='fAnova'`,
`self.fAnova_best`.

### Building and replaying

`feat_with_depth` (build) and `feat_with_depth_gen` (replay) are mirror images:

- **Build** recurses down, *appending* ops and leaves as it goes (post-order).
- **Replay** *pops* from the end of those lists — LIFO, which walks the tree in
  the mirrored order, visiting right subtrees first.

Because replay is mirrored, `feat_with_depth_gen` applies binary operators as
`op(feat_2, feat_1)` — the operands swapped relative to the build side.

> **Do not "fix" that swap.** It looks like a bug and is not. The two reversals
> cancel exactly. Changing it to the natural-looking `op(feat_1, feat_2)` breaks
> 1070 of 1500 randomly generated trees. This was verified symbolically and
> numerically; see [CORRECTNESS_FIXES.md §3.1](CORRECTNESS_FIXES.md). A property
> test guards it.

Time-series operators additionally record `feature_index` in their `params`, so
a recipe fully describes which column it consumed rather than depending on
traversal-order state.

---

## 4. `fit()` — the generation loop

`fit()` is ~660 lines in five phases.

### Phase 1 — Setup

1. **Reset per-fit state** (`_reset_fit_state`). The operator pool is rebuilt
   from base definitions; without this a second `fit` on the same object
   inherits the first's operators.
2. **Optional target log-transform.** Strictly positive regression targets with
   skew > 2 are log-transformed *for importance scoring only*. Exposed as
   `target_log_transformed` / `inverse_transform_target()`.
3. **Time-series setup** (`_setup_time_series`) — see §5.
4. **Pre-flight check.** Fits two `LinearRegression` models on the tail of the
   series, with and without simple rolling features. If the improvement is
   < 5%, the TS operator weight is halved *for this fit only*.
5. **Optional block downsampling** (`_apply_block_downsampling`) — see §6.
6. **Scaling and seed importances.** A `RobustScaler` is fitted, then
   RandomForest/LightGBM importances seed `self.ig_vector`, the distribution
   that leaf sampling draws from. With `split_feats='comb'` this is multiplied
   by per-feature decision-tree split frequencies.

### Phase 2 — The iteration loop

This is a **weighted hill-climb with elitism**, not a genetic algorithm: there
is no crossover, and surviving recipes are never mutated — each round generates
fresh candidates from scratch.

Per iteration:

```
1. Elitism      carry over the top n_feats//5 features from the previous
                round verbatim, with their recipes
2. Generate     for each remaining slot, up to 3 attempts:
                  - sample depth d from weights ∝ 1/2^d
                  - build a random tree of that depth
                  - accept if |corr(new, y)| >= max |corr(parent, y)|
                  - keep the best attempt if none clears the bar
3. Score        RandomForest/LightGBM importance over all candidates
4. Select       argsort, keep the top n_feats; slice recipes in parallel
5. Reweight     update operator sampling weights (below)
```

**Operator weights** evolve so the search concentrates on what works without
collapsing onto one operator:

- Decay all weights by **0.8** (history fades but still counts)
- Increment by usage count in this round
- **Diversity penalty**: any operator exceeding 50% of usage has its increment
  multiplied by **0.1**, forcing exploration
- Time-series operators are scaled by `effective_ts_weight_multiplier`
- Normalize with a floor so no operator is starved to zero

**Leaf sampling** draws from `ig_vector` (importance-weighted), so
high-importance columns appear more often in expression trees.

### Phase 3 — Final selection

- `selection='stability'` re-scores the full history across all iterations
- `check_corr=True` drops highly correlated generated features
- The scaled original features are appended (`hstack`)
- `selection='fAnova'` fits a `SelectKBest` and applies it

### Phase 4 — Output

If downsampling ran, `transform` is re-run over the full data so the returned
matrix covers every row. Otherwise the row order is restored to match the input.

---

## 5. Time-series subsystem

### The `enable_time_series` modes

| Mode | Behaviour |
|---|---|
| `'no'` | Disabled. Arithmetic operators only. |
| `'yes'` | Forced on. Requires `datetime_col`. Uses the single configured detector. |
| `'auto'` (default) | Detects a datetime column, then runs the ensemble vote below. |

### The ensemble vote

In `'auto'` mode all three detectors run and each returns a periodicity verdict.
Time-series features are enabled if **at least two agree** the data is periodic,
or a single detector exceeds 0.7 confidence. Windows are then pooled from the
agreeing detectors and sampled at even quantiles to span the full detected
scale range.

### Detector interface

All three subclass `BaseWindowDetector` and implement only:

```python
_preprocess_signal(series)      # conditioning: NaN fill, detrend, windowing
detect_optimal_windows(...)     # -> (windows, per-feature confidences)
```

Everything else — datetime-column detection, sampling-rate inference,
sample→day conversion, the window ladder, `assess_periodicity`,
`smart_window_selection` — lives in the base class, so the three cannot silently
diverge. `assess_periodicity` requires both average confidence above threshold
*and* at least half the features individually periodic.

| Detector | Method | Best for |
|---|---|---|
| DFT | FFT magnitude spectrum; confidence = peak-to-median ratio | Regular sampling, strong single period |
| ACF | Autocorrelation peaks; confidence = peak height | Short series, robust to noise |
| Lomb-Scargle | Periodogram over true timestamps | Irregular sampling, missing data |

One asymmetry remains deliberate: DFT alone applies `_apply_seasonal_bias`,
which snaps a detected period onto a common seasonality (7/14/30/90/180/365
days) when it lands within ±10%. The other two report their raw estimate. This
means identical data can yield slightly different windows depending on which
detector found it — worth knowing when comparing detectors head to head.

### Windows are real time spans

Detected windows are `pd.Timedelta`. A 90-day window covers 90 days of
timestamps regardless of whether the data is hourly, daily or monthly, and
regardless of calendar units having unequal lengths (28–31 day months, 90–92 day
quarters). The sampling rate is **measured from the data**
(`infer_sampling_rate`), not assumed.

This matters more than it sounds — the previous row-count approximation was
wrong on 100% of rows across all 25 benchmark datasets
([CORRECTNESS_FIXES.md §2.9](CORRECTNESS_FIXES.md)).

### Operator dispatch

The 17 `_safe_*` operators are thin wrappers. Each resolves its source column
via `_resolve_feature_col(params['feature_index'])` and delegates to
`_apply_time_based_operation`, which dispatches on an operation string:

```
_safe_rolling_mean ──▶ _apply_time_based_operation(data, col, 'rolling_mean', ...)
                          │
                          ├── groups present ──▶ _time_based_rolling (grouped)
                          ├── no groups      ──▶ _time_based_rolling (global)
                          └── calendar/seasonal ops, no groups
                                              ──▶ _apply_time_based_operation_loop
                                                    └─▶ _apply_single_group_operation
```

**All operators are causal.** A row's features depend only on rows dated at or
before it. Rolling windows never cross an entity boundary when `groupby_cols`
is set.

> The `_apply_time_based_operation_loop` → `_apply_single_group_operation` path
> looks dead to static analysis. It is not: it serves `weekday_mean`,
> `month_mean`, `seasonal_decompose` and `trend` when `groupby_cols` is empty.
> Instrumentation confirms 4 live calls. See
> [CORRECTNESS_FIXES.md §3.2](CORRECTNESS_FIXES.md).

### Restricted modes

`_setup_time_series` narrows the operator pool when the data does not suit the
full set:

| Trigger | Mode | Operators |
|---|---|---|
| `avg_lag1 > 0.85` (non-stationary) or consensus confidence < 0.5 | Restricted | lag, diff only |
| Not periodic but strongly autocorrelated | Trend | diff, pct_change, lag (weights ×3) |
| Otherwise | Full | all 17 |

---

## 6. Block downsampling

Off by default. When `enable_downsampling=True`, `fit` bounds memory by sampling
**contiguous blocks** of rows rather than random rows — random sampling would
destroy the temporal continuity rolling features depend on.

- Available RAM is measured with `psutil`; the row limit is derived from it and
  capped by `max_fit_samples` if set
- Each block carries a **padding prefix** of prior rows so its first windows
  have real history
- Rows are tagged with `_block_id`, which joins `groupby_cols` so rolling
  operations never span a seam
- **Discovery samples; application does not.** `transform` still returns every
  input row

---

## 7. `transform()`

1. Select the stored `feature_columns`
2. Rebuild time context (`_prepare_time_series_data`) — sorts by datetime,
   records `_original_index` for later restoration
3. Apply the fitted scaler
4. Replay each recipe via `feat_with_depth_gen`
5. `hstack` with the scaled originals
6. Apply `fAnova` selection **before** restoring row order (these are
   independent; doing selection first lets both paths share it)
7. Restore the caller's row order

**Invariant:** `fit(X)` and `transform(X)` must produce the same matrix for the
same `X`, and `transform` must be invariant to input row order. Both are
enforced by the test suite and both caught real bugs.

---

## 8. Known rough edges

**`local_utils.py` is ~690 unreferenced lines.** Roughly 40 technical-analysis
indicators (RSI, MACD, Bollinger, Aroon, Parabolic SAR, Fibonacci, …) were
written but never wired into the operator pool, which is fully specified at
`bigfeat_base.py:154`, `:156` and `_add_time_series_operators`. Only
`original_feat` is referenced — and it sits in `unary_operators` but not
`operators`, so `feat_with_depth` can never sample it. Either connect them or
delete them.

**`fit()` is 664 lines; `_setup_time_series` is 360.** Both have clean internal
seams but splitting them is mechanical work with real regression risk and no
behavioural benefit — better done against a specific need.

**Not sklearn-compatible.** No `get_params`/`set_params`, so no `Pipeline` or
`GridSearchCV` without a wrapper.

**Broad exception handling.** Every `_safe_*` operator and
`_apply_time_based_operation` wrap their body in `except Exception`, degrading
to zeros. This makes a systematically broken operator look identical to an
unhelpful one — the pre-flight check bug hid this way for the entire life of the
feature.

**The stationarity gate may be too permissive.** See
[CORRECTNESS_FIXES.md §6](CORRECTNESS_FIXES.md).

---

## 9. Where to start reading

| Goal | Entry point |
|---|---|
| Understand generation | `fit()` phase 2, then `feat_with_depth` |
| Understand replay | `transform()`, then `feat_with_depth_gen` |
| Add an operator | `_add_time_series_operators`, then any `_safe_*` as a template |
| Change window detection | `window_detector_base.py`, then the specific detector |
| Debug a wrong feature value | `_apply_time_based_operation` — start with the dispatch table |
