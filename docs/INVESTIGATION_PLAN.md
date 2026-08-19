# Investigation Plan — Measuring BigFeat Honestly

A plan to establish what BigFeat's accuracy actually is on the fixed code, why
the correctness fixes did not obviously improve it, and what could.

Written after the correctness review
([CORRECTNESS_FIXES.md](CORRECTNESS_FIXES.md)) closed six defects. A small
post-fix A/B (12 datasets, single seed) showed accuracy roughly unchanged, but
that sample is too small to conclude from — establishing the real number is
Phase A, and it blocks most of what follows.

---

## 0. What the existing data does and does not show

> **The committed `benchmark_results/` is STALE.** Those runs are dated
> 2026-01-24 to 2026-02-12; the twelve correctness fixes landed 2026-08-04.
> Every BigFeat number in that directory was produced by code with rolling
> windows wrong on 100% of rows, a seasonality guard that had never executed,
> and an inert `random_state`. **BigFeat rows below are not evidence about
> current behaviour** and are shown only to motivate the re-run in Phase A.

Analysis of the committed summary (24 datasets, 8 methods):

| Method | geo-mean MASE ratio vs baseline | Wins | Sign-test *p* | Status |
|---|---|---|---|---|
| `openfe` | 1.001 | 12/22 | 0.42 | **valid** |
| `tsfresh` | 1.455 | 8/18 | 0.76 | **valid** |
| `no_standard` (BigFeat, no TS) | 1.112 | 10/24 | 0.85 | stale |
| `yes_dft` | 1.121 | 11/24 | 0.73 | stale |
| `yes_lomb_scargle` | 1.148 | 10/24 | 0.85 | stale |
| `yes_acf` | 1.168 | 10/24 | 0.85 | stale |
| `auto_ensemble` | 1.201 | 8/24 | 0.97 | stale |

### What survives the staleness

**OpenFE and tsfresh do not touch BigFeat's code**, so the fixes cannot have
changed them. Both were measured against the same baseline, on the same
datasets, under the same harness. That gives one durable observation:

> A mature, independent AutoFE tool (OpenFE) ties the no-feature-engineering
> baseline exactly — geo-ratio 1.001, 12/22 wins, *p* = 0.42. tsfresh is 45%
> worse. Neither beats baseline at any conventional significance level.

This is weak evidence that **the ceiling is a property of this benchmark**
rather than of any one tool. It is the single most important input to the plan
below, and it is unaffected by the correctness fixes.

### What does not survive

Every BigFeat row, and both sub-analyses that depended on them:

- ~~No frequency subgroup where BigFeat wins~~ (D 1.103, M 1.186, Q 1.156,
  W 1.179, Y 0.997, h 1.068) — computed from stale BigFeat numbers
- ~~No dataset characteristic predicts success~~ (`corr(log n_series, log
  ratio) = −0.09`) — same

Both must be recomputed after Phase A. They are recorded here so the comparison
can be made: if the post-fix run reproduces these patterns, the fixes were
accuracy-neutral; if it does not, the difference is attributable and worth
reporting.

**The honest current position: BigFeat's accuracy relative to baseline is
UNMEASURED on the fixed code.** The 12-dataset A/B in
[CORRECTNESS_FIXES.md §4](CORRECTNESS_FIXES.md) is the only post-fix evidence,
and it is underpowered (≤25 series, ≤120 rows, single seed).

> **A note on framing.** "Improve the results" and "find out what is true" point
> in different directions here. Searching configurations until something wins,
> on 24 datasets × 8 arms, will find a winner by chance — and the current
> numbers exist partly because an earlier round of tuning was measuring
> artifacts (a `random_state` that did nothing, windows that averaged whole
> series, a guard that never ran). Every phase below fixes its analysis plan
> before looking at outcomes.

---

> **See also** [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md): the Monash suite
> is univariate with no covariates, so it cannot exercise cross-column
> composition at all. That document proposes the replacement evaluation;
> Phase A below should be read as applying to whichever suite is adopted.

## Phase A — Establish the real baseline (blocking)

Everything else is speculation until this exists.

**A1. Re-run the full benchmark on the fixed code.** The committed results
predate all twelve fixes, so they measure behaviour that no longer exists. The
A/B in `CORRECTNESS_FIXES.md` §4 was 12 datasets, ≤25 series, ≤120 rows, single
seed — underpowered, and a geo-mean of 1.001 on that sample cannot be
distinguished from noise.

- Full grid: ~172 h (see `testing/Benchmarking/README.MD`)
- Reduced grid (`baseline`, `no_standard`, `auto_ensemble`, `yes_dft`, `openfe`)
  across all 25 datasets: roughly 90 h
- **Run ≥3 seeds.** Single-seed results cannot separate method effects from
  variance, and this is the flaw that most weakens the current numbers.

**A2. Quantify seed variance first.** Before the full run, take 3 datasets ×
5 seeds. If within-method seed variance is comparable to between-method
differences, the entire comparison needs a different design (paired tests,
more seeds) and that is worth knowing before spending 90 hours.

*Deliverable:* `benchmark_results_v2/` + a variance report.
*Blocks:* every quantitative claim below.

---

## Phase B — Why didn't the fixes help? (parallel, cheap)

The central puzzle. Correct time windows *should* beat windows that averaged
entire series. Three testable hypotheses, in order of how cheaply they can be
falsified.

**B1. The downstream model is insensitive to the features.**
Take one dataset, generate features under both the old and new window
semantics, and fit RandomForest, Ridge and GBM on each. If MASE barely moves
for RF but moves for Ridge, the tree ensemble is absorbing the difference — RF
can approximate a lag internally, so a supplied lag feature adds nothing.
*Cost: hours. Highest prior.*

**B2. Selection is the bottleneck, not generation.**
Instrument which features survive to the final matrix. If generated features are
consistently ranked below the raw passthrough columns, the generator is fine and
the importance-based selector is the constraint. Measure: fraction of final
columns that are generated vs. passthrough, and their importance distribution.
*Cost: hours.*

**B3. The evaluation setup doesn't reward temporal features.**
The harness holds out the last *h* rows per series and predicts them from
contemporaneous features. Check whether the target is effectively predictable
from the raw last value — if a naive-1 forecast is near-optimal, no feature
engineering can help, and MASE ≈ 1 for everything is the expected outcome.
*Cost: hours. If true, this explains the entire table above.*

*Deliverable:* `docs/WHY_NO_GAIN.md` — one section per hypothesis, each with
the measurement that supports or refutes it.

---

## Phase C — The stationarity gate (parallel, cheap)

The one place with concrete evidence of a fixable defect: `covid_deaths`
regressed 5× because `avg_lag1 = 0.725` fell below the 0.85 gate, so trending
data was not routed to restricted mode and EWM was selected where it is 3×
worse than a raw lag.

**C1. Compute `avg_lag1` for all 25 datasets.** The committed summary has it for
only 8. Compute it directly — cheap, no fit required.

**C2. Test whether the gate separates anything.** Plot `avg_lag1` against
`MASE(TS-enabled) / MASE(TS-disabled)`. If trending datasets cluster, the gate
has a principled home. If they do not, the gate is the wrong mechanism and
should be replaced (e.g. by an explicit stationarity test such as ADF/KPSS)
rather than retuned.

**C3. Do not move the threshold on one dataset's evidence.** Pre-register the
rule: change 0.85 only if C2 shows separation across ≥5 datasets, and report
the new value's effect on all 25, not the subset that motivated it.

*Deliverable:* `docs/STATIONARITY_GATE.md` + a figure.

---

## Phase D — Capability work (only after B)

Ordered by expected value, but **each is conditional on Phase B's diagnosis.**
If B3 holds (the task does not reward temporal features), none of these will
help and the effort should go into reframing the paper instead.

**D1. Connect `local_utils.py`.** ~40 written-but-unwired indicators (RSI, MACD,
Bollinger, Aroon, Parabolic SAR, Fibonacci, …). The single largest untapped
capability in the codebase — roughly 690 lines that have never run. Wire them
into the operator pool behind a flag and measure.

**D2. Revisit the depth distribution.** Weights ∝ 1/2^d put ~57% of mass on
depth 1, so deep compositions are rarely explored. Test a flatter distribution.

**D3. Revisit the acceptance test.** Candidates are accepted on
`|corr(feature, y)|` beating their parents, but are *selected* on tree-ensemble
importance. Correlation is a linear proxy for a non-linear criterion; the two
disagree on exactly the interaction features BigFeat exists to find.

**D4. Reconsider the operator pool.** Arithmetic composition of tabular columns
may simply be the wrong prior for forecasting, where lags and calendar structure
dominate.

*Deliverable:* one ablation table, all arms reported.

---

## Phase E — Write-up

**E1. Lead with the reproducibility finding.** This is already documented and
verified, and it is a genuine contribution independent of the accuracy story: a
published AutoFE tool had rolling windows wrong on 100% of rows across an entire
benchmark suite, a seasonality guard that had never once executed, a
`random_state` parameter that did nothing, and look-ahead leakage in two
operators. The corrected behaviour is now covered by 91 tests.

**E2. Report whatever Phase A shows, including a negative result.** If the
post-fix run confirms that no AutoFE method beats baseline on Monash, say so.
OpenFE tying baseline at 1.001 -- a measurement unaffected by the fixes -- is
already evidence that this may be a property of the benchmark rather than of
BigFeat. A well-supported negative is a more useful contribution than a
fractional MASE improvement that will not replicate.

**E3. Pre-commit to the analysis.** Fix the primary metric (MASE), the
comparison (paired across datasets), the test (Wilcoxon signed-rank or a CD
diagram), and the seed count **before** looking at Phase A's output.

---

## Sequencing

```
A2 (seed variance) ──▶ A1 (full re-run, ~90 h) ─────────────┐
                                                             ▼
B1 B2 B3  (parallel, hours)  ──▶ diagnosis ──▶ D1..D4 ──▶ E (write-up)
C1 C2 C3  (parallel, hours)  ──▶ gate decision ────────────▶
```

B and C need no re-run and can start immediately: B runs its own targeted
experiments, and C computes `avg_lag1` directly from the datasets. Neither
depends on the stale BigFeat numbers. A1 is the long pole and should be launched
first so it runs in the background.

Note that C2 correlates `avg_lag1` against a MASE ratio, so its *conclusion*
must wait for Phase A even though its measurement can be taken now.

## What would change the conclusion

Stated in advance, so the plan can fail honestly:

- **B3 refuted + D1 helps** → BigFeat has a real capability gap; fix it and the
  accuracy story becomes viable.
- **B3 confirmed** → the benchmark cannot reward feature engineering; reframe
  around reproducibility and propose a better evaluation.
- **A2 shows seed variance dominates** → all existing conclusions, positive and
  negative, are underpowered; the contribution becomes methodological.
