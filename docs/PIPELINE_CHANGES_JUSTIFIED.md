# Pipeline Changes, Justified

Every change made to the time-series pipeline across both rounds of work,
organised by pipeline stage, each with the evidence that motivated it and the
justification for the design chosen. Self-contained by intent — this is the
document to hand to someone who asks "what exactly did you change, and why?"

Deeper records: [CORRECTNESS_FIXES.md](CORRECTNESS_FIXES.md) (round 1, with
per-fix verification detail), [PIPELINE_IMPROVEMENTS.md](PIPELINE_IMPROVEMENTS.md)
(round 2 narrative), [SYNTHETIC_STUDY.md](SYNTHETIC_STUDY.md) and
[PILOT_BIKE.md](PILOT_BIKE.md) (the measurements).

The pipeline under discussion:

```
signal (+ target) ─▶ detectors (DFT · ACF · Lomb-Scargle)
                  ─▶ periodicity vote ─▶ stationarity gate
                  ─▶ fundamentals consensus
                  ─▶ windows · lags · cyclical encodings
                  ─▶ operator pool ─▶ feature generation ─▶ replay
```

---

## Stage 1 — Rolling execution

### 1.1 Windows are time spans, not row counts

**Change.** Rolling operations receive the detected `pd.Timedelta` and let
pandas select rows **by timestamp**. The old path converted every window to a
row count assuming one row per day.

**Evidence.** A "90-day" window used 90 rows at every sampling frequency.
Measured against true time-based rolling on the Monash datasets: **wrong on
100% of rows on every dataset tested** (mean absolute error 72,057 on
m1_monthly; 1,463,465 on m1_yearly — on monthly data, 90 rows is 7.5 years).
13 of the 25 benchmark datasets are monthly/quarterly/yearly, where calendar
units are not fixed durations, so a row count *cannot* be correct there.

**Justification.** The window's meaning must be invariant to sampling
frequency — "average the last week" is a statement about time, not about
array positions. Row-count rolling is also faster only superficially: on
monthly data the correct window touches 3 rows, not 90, and the fix measured
2.8× *faster* end to end.

### 1.2 Group before rolling

**Change.** With `groupby_cols` set, grouping happens before the rolling
window is applied. The old path rolled globally over the whole frame and
masked each group's first rows afterwards.

**Evidence.** A group's early rows averaged in the *preceding entity's*
values before being zeroed — store B's first "weekly average" contained
store A's numbers.

**Justification.** Entity isolation must be structural, not cosmetic. Masking
after the fact hides the contamination in exactly the rows where it is
hardest to notice, and only the rows it happens to zero.

### 1.3 Causal calendar means

**Change.** `weekday_mean` / `month_mean` use `shift(1).expanding().mean()`
within each calendar group. The old code used `transform('mean')` over the
whole column.

**Evidence.** Every row's value depended on rows dated after it — textbook
look-ahead leakage, present in **two** independent code paths (fixing only
the first would have left the default configuration leaking).

**Justification.** "How does this Tuesday compare to *previous* Tuesdays" is
the only version of this feature that exists at prediction time. A feature
that validation likes but production cannot compute is worse than no feature.

---

## Stage 2 — The detectors

### 2.1 ACF: bounded lags (round 1)

**Change.** Autocorrelation requires a minimum overlap of 30 samples and
caps `max_lag` at n/2. Previously lags ran to n−1.

**Evidence.** The correlation of two points is always exactly ±1, so white
noise produced |ACF| = 1.0 at long lags — and the detector reported **pure
noise as periodic at confidence 0.973**, with 163–363-day windows flowing
into feature generation.

**Justification.** n/2 with a minimum-overlap floor is the standard rule for
meaningful autocorrelation estimates. An automated system that always finds
seasonality is worse than none: it manufactures confidence.

### 2.2 DFT: peak-to-background confidence (round 1)

**Change.** Confidence = dominant peak over the spectrum **median**, mapped
to [0,1). Previously `1 − second/first` over the sorted spectrum.

**Evidence.** The two largest bins of a genuinely periodic signal are
neighbouring bins of the *same* peak, split by spectral leakage — so the old
metric scored a clean 7-day sine at 0.244 (below threshold, "not periodic")
and noise at 0.075. It measured approximately the inverse of its intent.

**Justification.** The median estimates the noise floor because only a
handful of bins carry real signal; peak-to-floor is the standard
periodogram-significance construction. After the fix: periodic 0.628 vs
noise 0.263 — a decisive margin where there was none.

### 2.3 ACF: fundamentals, not harmonics (round 2)

**Change.** Peaks are accepted in **ascending-lag order** (the fundamental is
the first peak clearing the floor), each verified by an ACF echo at a small
multiple, and each accepted period's harmonic train is masked before looking
for a second period. Previously peaks were ranked by height.

**Evidence.** For a signal with periods P and Q, ACF at common multiples
*exceeds* the fundamentals — every component realigns there. On a planted
7+30-day signal, ACF(210) = 0.997 vs ACF(7) = 0.652, so the height-ranked
top-3 were 210/91/301: all junk, both true periods missed. Across five
planted cases the height rule missed the fundamental in 3.

**Justification.** This matches how the literature treats the lag domain
(AUTOPERIOD uses ACF for *verification*, not unsupervised peak-picking).
Two refinements came from implementation and are recorded as spec
amendments: the echo rule is ANY-of {2L, 3L} — in a multi-period signal the
other component can sit anti-phase at exactly 2L and cancel the echo — and
ACF *cannot* recover the second period at all (the fast comb's tooth beside
it towers over it), so its contract is shortest-fundamental-first and
multi-period recovery belongs to the ensemble.

### 2.4 DFT: top-k spectral peaks (round 2)

**Change.** All spectral peaks above 3× the spectrum median are candidates,
**deduplicated in period space** (±15%), top-k kept. Previously a single
`np.argmax`.

**Evidence.** One argmax means a second genuine period is never proposed.
And without period-space dedup, the three tallest bins were all leakage
lobes of the *same* peak (~10.9/11.1/10.9 days on an 11-day signal), so
top-k alone still missed the second period.

**Justification.** Height-ranking is *correct* in the frequency domain —
fundamentals exceed their harmonics — the exact opposite of the lag domain
(§2.3). That asymmetry is documented at both code sites, because "make the
two detectors consistent" would have been precisely wrong.

### 2.5 Sampling rate measured from the data (round 1)

**Change.** `infer_sampling_rate` takes the median timestamp spacing and
snaps it to a frequency alias; the sample→days conversion accepts modern
pandas aliases and parses unknown codes instead of silently assuming daily.

**Evidence.** The ensemble path never passed a rate, so monthly observations
were treated as daily and every detected period came out 30–90× too small —
1–6-day windows for series sampled once a month. Separately, lowercase 'h'
fell through the alias table and inflated hourly periods 24×, silently.

**Justification.** The sampling rate is a property of the data; any default
is a guess that fails silently on exactly the frequencies (M/Q/Y) that
dominate the benchmark. The median resists gaps between entities.

---

## Stage 3 — Vote, gate, consensus

### 3.1 Unified periodicity verdict (round 1)

**Change.** One shared `assess_periodicity` in a detector base class:
periodic only if average confidence clears the threshold **and** at least
half the analysed columns individually clear it.

**Evidence.** Three same-named methods had silently diverged — only DFT
required the feature-level consensus, so for ACF/LS one strongly periodic
column among many noisy ones enabled time-series features for the whole
frame.

**Justification.** Methods that share a name must share semantics; the
stricter rule was kept because the failure mode it prevents (hallucinated
seasonality) is the expensive one.

### 3.2 Stationarity gate: ADF instead of lag-1 autocorrelation (round 2)

**Change.** Restricted mode (lag/diff operators only) triggers on an
Augmented Dickey–Fuller unit-root verdict (median p > 0.05 across sampled
columns). Previously: mean |lag-1 autocorrelation| > 0.85. statsmodels is an
optional dependency; the old heuristic remains as an explicit fallback.

**Evidence.** Lag-1 autocorrelation measures **smoothness**, and it
misclassified in both directions: a clean *stationary* 30-day seasonal
(lag-1 ≈ 0.97) was routed to restricted mode — stripping rolling operators
from exactly the data the subsystem exists for — while a noisy random walk
(covid_deaths, 0.725) sailed into the full pool, where a smoothing operator
was selected at a measured 5× MASE cost. ADF classified all six canonical
test series correctly.

**Justification.** Use the instrument built for the question. The gate asks
"does a rolling mean describe signal or trend here?", which is the
unit-root question, not the smoothness question. The threshold was not
retuned on the failing dataset — the *statistic* was replaced, per the
pre-registered rule against single-dataset tuning.

### 3.3 Target-aware detection (round 2)

**Change.** In `'auto'` mode the detectors analyse a local frame that
includes `y` as a synthetic first column, feeding the gate, the vote, and
the fundamentals stash. Strictly local: it cannot reach feature generation
or `transform()`.

**Evidence.** `_setup_time_series(X, y)` accepted `y` and never used it —
the only reference was the docstring. On a fixture where features carry a
5-day cycle and the target an 11-day one, the output lags were [1, 5]: the
predicted signal's own rhythm was invisible.

**Justification.** The periodicity that matters for prediction is the
target's. The argument was already plumbed; the change is to use it — with
guards for length mismatch, non-numeric and all-NaN targets so it degrades
to the old behaviour rather than failing.

### 3.4 One fundamentals consensus, three consumers (round 2)

**Change.** `_consense_fundamental_days` clusters the detectors' raw
(period, confidence) pairs — ±15% tolerance, confidence-weighted means,
clusters ranked by summed confidence — **once**, before window finalisation.
Windows, lags and cyclical encodings all consume the same protected list,
and no truncation step (ladder or pool) may drop a fundamental.

**Evidence.** Three separate failures converged here: pooled-window
truncation kept the n *smallest* windows (1–6-day windows on monthly data);
after that was fixed, quantile subsampling could still drop a fundamental
the lags proved was detected (6/78 synthetic cases); and lags re-clustered
independently to top-2, so pair lags recovered only 6/24.

**Justification.** A period seen by two detectors outranks one detector's
tall stray peak — cross-method agreement is the strongest signal available.
Deriving each consumer from its own re-clustering created disagreement
between outputs of the *same* detection. Notably, the full consensus
machinery was **deferred twice** ("no machinery without a measured failure
it would fix") and built only when the synthetic study produced that
failure — in exactly the slice the evidence demanded.

---

## Stage 4 — Feature construction

### 4.1 Lags equal detected cycles (round 2)

**Change.** `lag_periods = [1 step] + detected fundamentals`. Previously
positional picks from the window ladder (`windows[0], windows[1],
windows[mid]`).

**Evidence.** On the 7+30-day fixture, generation received lags [1, 3, 7]:
the 30-day fundamental absent, the 7 present only by accident of its
position in the pooled list.

**Justification.** A window *spans* a cycle; a lag *equals* one — "the same
point one cycle ago" is only meaningful at the detected period itself.
Deriving lags from window positions conflated two different roles.

### 4.2 Cyclical sin/cos encodings at detected periods (round 2)

**Change.** Two new operators computing `sin/cos(2π·t/P)` where `t` is the
timestamp against a **fixed epoch** and P a detected fundamental. No phase
encoding existed before (verified: the only sin/cos in the pool were
fallback shims).

**Evidence & justification.** Fourier terms are the standard regression
representation of seasonality (Hyndman & Athanasopoulos, fpp3 §7.4); they
hand tree models the phase information raw lags express only indirectly.
The fixed epoch makes the encoding a pure function of the timestamp — same
date, same value, at fit or transform, under any row order — i.e.
**leak-proof by construction**, verified byte-exact and shuffle-invariant.
It also gives detection a third consumer, tripling the value of getting P
right. Recipe replay required zero changes — evidence the recipes-not-values
representation is doing its job.

---

## How the whole set was verified

- **Every change landed test-first**, its acceptance test confirmed failing
  against the pre-change code. This caught two of our own fixtures being
  wrong (one passed on broken code because 4×11 = 44 sat inside a 45-day
  tolerance window — the ladder faked the recovery).
- **Ground truth, end to end** ([SYNTHETIC_STUDY.md](SYNTHETIC_STUDY.md)):
  78 planted-period cases + 6 noise controls, identical data across code
  versions. Window recovery 69% → 100%, lag recovery 46% → 100%, false
  positives 0/6 at every stage — the gains were never bought with
  sensitivity.
- **Real data** ([PILOT_BIKE.md](PILOT_BIKE.md)): with calendar columns
  present, nothing helps (including hand-crafted features — they hurt);
  with them removed, the pipeline recovers the structure automatically to
  within 6% of the calendar-equipped baseline. The bounded claim these
  changes support: *BigFeat recovers what an expert would encode by hand.*

## Changes considered and rejected

- **"Fixing" the replay operand swap** — verified load-bearing (the natural
  fix breaks 1,070/1,500 expression trees); guarded by a property test.
- **Retuning the 0.85 gate threshold** on the failing dataset — replaced the
  statistic instead (§3.2).
- **Making ACF and DFT rank peaks the same way** — the domains genuinely
  invert (§2.3 vs §2.4).
- **Full confidence-calibrated consensus, per-entity windows,
  successive-halving, FDR pre-filter, wiring the 42 idle indicators** — all
  deferred until a measured failure or review demands them; the
  deferral-with-tripwire pattern has fired correctly twice.
