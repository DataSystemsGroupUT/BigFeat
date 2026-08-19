# Benchmark Design for the Time-Series Publication

What to evaluate BigFeat's time-series feature generation on, and why the
current suite cannot demonstrate it.

Companion documents: [ARCHITECTURE.md](ARCHITECTURE.md) for how the subsystem
works, [CORRECTNESS_FIXES.md](CORRECTNESS_FIXES.md) for the correctness review,
[INVESTIGATION_PLAN.md](INVESTIGATION_PLAN.md) for the wider study plan.

---

## 1. The problem in one paragraph

BigFeat's time-series mode detects periodicity, derives window sizes, and
generates time-aware features that it **composes with other columns**. The
current benchmark — 25 datasets from the Monash archive — stores one numeric
column per series and nothing else. There is no second column to compose with,
so the benchmark exercises a small fraction of the mechanism. This is a
structural mismatch between capability and evaluation, not a tuning problem,
and no amount of re-running will fix it.

---

## 2. Evidence that the current suite is the wrong instrument

Three findings, each measured directly from the committed data.

### 2.1 There are no covariates

Every record in every dataset has the same shape:

```
{'target': [...], 'start': ..., 'item_id': ..., 'feat_static_cat': [...]}
```

One numeric channel. `feat_static_cat` is a constant series identifier, not a
time-varying signal.

BigFeat's central mechanism is composing operators **across columns** —
`rolling_mean(x1) - lag(x2)` is the kind of feature it exists to discover. With
a single channel per series, the search space collapses to unary transforms of
one column. We are running a combinatorial feature-composition engine on a
univariate problem.

### 2.2 Three datasets cannot show a full seasonal cycle

Periodicity detection needs roughly three complete cycles to work. Measured as
median series length divided by the expected seasonal period:

| Dataset | Freq | Median length | Expected period | Cycles visible |
|---|---|---|---|---|
| `nn5_weekly` | W | 105 | 52 | **2.0** |
| `web_traffic_weekly` | W | 106 | 52 | **2.0** |
| `electricity_weekly` | W | 148 | 52 | **2.8** |
| `car_parts_without_missing` | M | 39 | 12 | 3.2 |

Three datasets contain barely two periods per series. Detection cannot succeed
there, and a benchmark that includes them measures the detector's failure mode
rather than its capability.

### 2.3 An independent tool hits the same ceiling

On the same suite and harness, **OpenFE** — an established automated
feature-engineering library, entirely independent of BigFeat — finishes level
with the no-feature-engineering baseline:

- geometric-mean MASE ratio **1.0008**
- better on 12 of 22 datasets, worse on 10
- sign test **p = 0.42** — indistinguishable from a coin flip
- ratios cluster tightly in [0.99, 1.03], i.e. nothing moves in either direction

This measurement is unaffected by our correctness fixes, since OpenFE never
calls into BigFeat. It is the strongest available evidence that the ceiling
belongs to the benchmark rather than to any one method.

### 2.4 What is NOT the explanation

An earlier hypothesis was that a naive last-value forecast is already
near-optimal, leaving no headroom. Measured across all 25 datasets, that is
**too simple**: the median naive MASE is **2.25**, and only 7 of 25 datasets
have naive MASE below 1.5. There is real headroom on most datasets — the models
are beating naive comfortably.

The problem is therefore not "the task is trivially easy". It is that the
*feature engineering* has nothing to work with: one column, no covariates.
Recording this distinction matters, because it changes the fix from "find a
harder benchmark" to "find a benchmark with covariates".

---

## 3. Recommended evaluation

Three experiments. Together they support a complete claim: the mechanism works,
it helps where it should, and we know where it does not.

### Experiment A — Synthetic study with known ground truth

**Purpose.** Isolate the contribution. This is the only setting where the true
period is known, so detection can be scored directly rather than inferred from
downstream accuracy.

**Design.** Generate series with planted periods (7, 30, 91 days), sweeping:

| Factor | Levels |
|---|---|
| Planted period | 7, 30, 91 days |
| Noise level (SNR) | high, medium, low |
| Series length | 3, 5, 10, 20 cycles |
| Sampling regularity | regular, 10% missing, irregular |
| Number of covariates | 1, 3, 10 |
| Control condition | pure noise (no planted period) |

**Metrics.**

- **Detection accuracy** — is the planted period recovered within tolerance?
- **Window quality** — do the selected windows bracket the true period?
- **False-positive rate** — how often does it detect seasonality in the noise
  controls? This is the property that matters most and the one no real dataset
  can measure.
- **Downstream MASE** vs. an oracle that is handed the true period.

**Why it belongs in the paper.** It answers "does the detection work?"
separately from "does feature engineering help?", which the current design
conflates. The oracle comparison bounds how much of the achievable gain the
detector captures.

**Cost.** Hours. The test fixtures already do this at small scale.

### Experiment B — Multivariate regression with covariates

**Purpose.** The main result. Datasets where the target depends on several
time-varying signals, so cross-column composition can actually operate.

| Dataset | Source | Why it fits | Scale |
|---|---|---|---|
| **Beijing / UCI Air Quality** | UCI | PM2.5 from temperature, pressure, wind, dew point, rain. Strong daily and annual cycles. Canonical multivariate TS regression set. | ~44k rows |
| **Bike Sharing** | UCI | Demand from weather, humidity, windspeed, holiday flags. Clear daily + weekly structure, widely used, easy to reproduce. | ~17k rows |
| **Appliances Energy** | UCI | Energy use from 20+ sensor channels. Genuinely high-dimensional — the strongest case for composition. | ~20k rows, 28 cols |
| **Electricity Load Diagrams** | UCI / gluonts | Load with calendar and weather covariates; hourly, strong daily/weekly cycles. | 370 clients |
| **PeMS / METR-LA traffic** | Public | Speed from adjacent sensors; very strong daily/weekly periodicity. | Large |

`gluonts` already exposes 62 datasets including `electricity` and `traffic`, so
some of these need no new download infrastructure.

**Baselines to include.**

- No feature engineering (raw columns + calendar fields)
- BigFeat with `enable_time_series='no'` — isolates the time-series contribution
  from the arithmetic one
- OpenFE, tsfresh — independent AutoFE comparison
- A hand-crafted feature set (`lag_1`, `lag_7`, `rolling_7`, `rolling_30`) —
  this is the honest competitor, since it is what a practitioner would write

That last baseline is the one reviewers will ask for. Beating "no features" is
weak; beating "what a competent engineer writes by hand" is the actual claim.

### Experiment C — Monash as the declared hard case

**Purpose.** Honesty, and it strengthens rather than weakens the paper.

Keep 6-8 Monash datasets and report them explicitly as the adversarial setting:
univariate, no covariates, nothing to compose. If BigFeat ties the baseline
there, that is the expected outcome and should be stated as such.

This also lets you report the **decline behaviour** as a positive result: on
data with no exploitable structure, the system disables time-series features
rather than manufacturing them.

---

## 4. Metrics and protocol

**Primary metric.** MASE, for comparability with the forecasting literature.
Report RMSE and R² as secondary.

**Statistical treatment.** Paired across datasets, Wilcoxon signed-rank or a
critical-difference diagram. With 25 datasets and multiple configurations, an
unpaired mean comparison will find a winner by chance.

**Seeds.** At least 3, preferably 5. BigFeat is stochastic; single-seed results
cannot separate a method effect from run-to-run variance. **Measure seed
variance first** on three datasets before committing to the full run — if
within-method variance rivals between-method differences, the design needs
changing and that is worth learning in two hours rather than after ninety.

**Pre-registration.** Fix the primary metric, the comparison, the test and the
seed count *before* looking at results. State this in the paper.

---

## 5. Handling the obvious objection

Changing benchmarks after unfavourable results invites the charge of
venue-shopping. Pre-empt it directly:

1. **State the structural argument, not the outcome.** Monash records are
   univariate with no covariates; BigFeat composes across columns. The mismatch
   is a property of the data format, demonstrable without reference to any
   result.
2. **Cite the OpenFE tie.** An independent tool hits the same ceiling on the
   same suite. That is evidence about the benchmark, not about us.
3. **Keep Monash in the paper.** Reporting it as the declared hard case is the
   difference between diagnosis and avoidance.

---

## 6. Sequencing

```
Experiment A (synthetic)      hours    -> can start immediately
Seed-variance check           ~2 h     -> gates the expensive runs
Experiment B (multivariate)   days     -> the main result
Experiment C (Monash subset)  ~20 h    -> reduced grid, reported as hard case
```

Experiment A is the natural next step: cheapest, produces publishable figures,
requires no new data, and answers the question the current benchmark cannot
answer at all.

---

## 7. What would change this recommendation

- **Experiment A shows poor detection accuracy** — then the problem is the
  detectors, not the benchmark, and the priority shifts to fixing them.
- **Experiment B shows no gain despite covariates** — then the limitation is in
  feature *selection* or the search, not in the evaluation. That would point at
  hypothesis B2 in the investigation plan.
- **Seed variance dominates** — all existing conclusions, positive and negative,
  are underpowered, and the contribution becomes methodological.
