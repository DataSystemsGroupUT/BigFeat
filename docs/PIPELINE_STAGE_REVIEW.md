# Stage-by-Stage Design Review — Time-Series Pipeline

Companion to [PIPELINE_GAPS.md](PIPELINE_GAPS.md) (which audited the
detection→pooling seam). This reviews every remaining stage: was the design
decision right, what does the literature do, what should change. Every verdict
is backed by a measurement run for this review or a cited method.

Verdict key: **sound** / **defensible** / **wrong-instrument** / **defect**.

---

## 1. Signal preprocessing — *defensible, one asymmetry*

`ffill→bfill→fillna(0)`, linear detrend, Hamming window (DFT only).

Standard practice. Two notes: Lomb-Scargle still **never detrends** (it drops
NaNs pairwise instead — correct — but a trending series leaks a spurious
long-period peak into LS specifically); and `fillna(0)` after ffill/bfill only
triggers on all-NaN columns, where zero is as good as anything. Change: give
LS the same linear detrend the others have. Small.

## 2. Stationarity gate — *wrong-instrument* (measured)

`avg_lag1 > 0.85` over the first 5 feature columns routes to restricted mode
(lag/diff only).

Lag-1 autocorrelation measures **smoothness**, not non-stationarity. Measured
against ADF/KPSS on six canonical series:

| Series | lag-1 | Gate fires? | Truly non-stationary (ADF)? |
|---|---|---|---|
| white noise | −0.02 | no | no |
| strong 7-day seasonal | 0.62 | no | no |
| **smooth 30-day seasonal** | **0.97** | **yes** | **no** |
| random walk | 0.97 | yes | yes |
| trend + seasonal | 0.89 | yes | yes |
| AR(1) φ=0.7 | 0.71 | no | no |

The gate **strips rolling operators from a clean stationary monthly cycle** —
the exact case the subsystem exists for — while the covid-style walk at 0.725
slips under (the regression found in the benchmark A/B). ADF classifies all
six correctly. `statsmodels` is already in the benchmark environment; as a
library dependency it is optional-importable with the current heuristic as
fallback. Also: "first 5 columns" is an arbitrary sample; test all feature
columns or a random sample. **This is the highest-value fix in this review.**

## 3. Binary consensus vote — *defensible mechanism, defective override*

`(2 of 3 detectors) OR (max_confidence > 0.7)`.

The 2-of-3 part is sound and matches the hybrid-method literature. Two
problems: (a) the three confidence scores remain **incommensurable**
(peak/median map, correlation height, √power), so the 0.7 override means a
different thing per detector; (b) the override lets a *single* detector at
0.71-on-its-own-scale overrule two explicit "no" votes — the exact
hallucination path the vote exists to block.

Literature: [AUTOPERIOD (Vlachos et al., SDM'05)](http://alumni.cs.ucr.edu/~mvlachos/pubs/sdm05.pdf)
does not vote at all — it takes periodogram candidates and **verifies each on
the ACF** (candidate valid only if it lands on an ACF hill).
[RobustPeriod (arXiv:2002.09535)](https://arxiv.org/pdf/2002.09535) likewise
couples frequency and time domains for multi-period detection. This
cross-domain *verification* is strictly stronger than independent voting and
composes naturally with the consensus stage in PIPELINE_FIXES_SPEC.md Fix 2:
DFT/LS propose, ACF verifies. Change: drop the single-detector override, or
calibrate all three scores against a synthetic noise corpus so 0.7 is a real
probability. Medium.

## 4. Pre-flight CV check — *defensible aim, wrong proxy*

Tail split, `LinearRegression` with/without simple rolling features, halve TS
weights if improvement < 5%.

Now that it executes (post-fix), its design is questionable: a linear model
scores features destined for a **tree ensemble** — a lag that only matters in
interaction shows zero linear gain; one split on ≤1000 tail rows is high
variance; 5% is arbitrary. Mitigation: it only *halves* weight, never
disables. Change: score with the downstream estimator class (a small RF), or
delete the stage and let selection do its job. Low priority.

## 5. Correlation acceptance gate — *defensible; earlier criticism partly withdrawn*

Candidate kept early if `|corr(f,y)| ≥ max |corr(parent,y)|`, else best of 3.

Measured: the "linear proxy misses interactions" worry (raised as D3 in the
investigation plan) is **mostly wrong** — once composed, an interaction
feature is linearly related to the target (`|corr(x1·x2, y)| = 0.99` when
`y = x1·x2`; XOR-style 0.996; quadratic 0.998). The gate passes the features
that matter.

The real weakness is narrower: on autocorrelated targets the parent series
itself correlates ≈1.0 with y, a bar no smoothed feature clears (roll-30:
0.65), so for TS candidates the gate **burns 3 candidate-generations per slot
without filtering** (best-attempt is kept regardless). Change: exempt TS
operators from the parent-beating test, or lower retries to 1 when the parent
bar exceeds ~0.95. Cheap CPU win, not a quality issue.

## 6. Operator-weight evolution — *sound decay, cliff-shaped penalty* (measured)

Decay ×0.8 (EMA — sound, standard), floor (sound, prevents starvation),
diversity penalty: increment ×0.1 when share > 50%.

Simulated 5 iterations with a genuinely-60%-useful operator in a 20-op pool:
its sampling probability converges to **0.111 vs 0.047 uniform** — capped
near 2× uniform *no matter how useful it is*, because the penalty is a cliff
at 50% share rather than graded damping. An operator that deserves 60% usage
and one that deserves 95% end in the same place. Change: replace the cliff
with sublinear credit (`increment = √count`) or a soft share cap (renormalise
any op above 40% down to 40%). Small.

## 7. Depth distribution 1/2^d — *defensible*

57% mass on depth 1, 29% on 2, 14% on 3. Matches the shape of what AutoFE
tools find useful (OpenFE searches depth-≤2 interactions almost exclusively;
deep expressions rarely survive selection anywhere). Possible refinement:
adapt weights toward depths that survive selection, symmetric to the operator
reweighting. Not a priority.

## 8. Feature selection — *defensible, one literature-backed alternative*

RF/LGBM importance ranking per iteration; stability re-scoring across
iterations; optional fANOVA.

Tree importances are biased toward high-cardinality and mutually correlated
features, and an importance *ranking* has no notion of "not significant".
[tsfresh's FRESH](https://www.sciencedirect.com/science/article/pii/S0925231218304843)
selects instead by per-feature hypothesis tests with **Benjamini–Yekutieli
FDR control** — a principled "keep nothing that isn't provably relevant"
filter. Cheap to add as a *pre-filter* before importance ranking (kill
features whose relevance is statistically indistinguishable from noise, then
rank survivors). This also directly addresses hypothesis B2 in the
investigation plan (selection as the bottleneck). Medium value, small cost.

## 9. Restricted / trend modes — *defensible, revisit after §2*

Hardcoded `windows=[1d,7d]`, `lags=[1d]`, weights ×3 for diff/pct_change.
Reasonable conservatism, but these paths currently trigger on the *wrong
inputs* because the gate above misfires. Fix the gate first; then re-derive
these from the detected fundamentals like everything else.

## 10. Block downsampling constants — *sound*

Padding = max window, min core = 3× max window, ~10 blocks, dedicated RNG.
All defensible; verified behaviour-preserving during the correctness round.
No change.

---

## What was designed RIGHT — decisions to keep and defend

A review that only lists flaws misleads. These original design decisions are
correct, several of them ahead of common practice, and should be kept — and
cited as strengths in the paper.

**The ensemble-of-detectors concept itself.** Running DFT, ACF and
Lomb-Scargle rather than picking one is the right call: the literature's best
methods (AUTOPERIOD, RobustPeriod) are explicitly multi-method because the
domains fail differently. The flaws found here are in the *combining rule*,
not the idea. Most AutoFE tools have no periodicity detection at all —
having three is a differentiator.

**Including Lomb-Scargle at all.** Almost no feature-engineering tool handles
irregularly sampled time series. LS is the standard instrument for it
(astronomy heritage), and its inclusion means the design anticipated
real-world data the Monash benchmark cannot even represent.

**Windows as real time spans (`Timedelta`), detected not configured.** The
decision that a window is a *duration* — with the sampling rate measured from
the data — is the architecturally correct one, and it is what most hand-rolled
pipelines get wrong (row-count windows). The original implementation had the
right interface; the correctness round fixed the execution, not the design.

**The multiscale ladder (P/2, P, 2P, 4P).** Expanding a detected period into
a harmonic family is well-founded: seasonal structure expresses itself at
multiple related scales, and this is what rescued DFT's single-peak limitation
in the two-period test. It needs the fundamental-protection guard (Fix 5),
but the mechanism is right.

**TS operators join the arithmetic pool instead of replacing it.** Letting
the search compose `rolling_mean(x1) - lag(x2)` across both operator families
is the design's most distinctive idea — tsfresh generates TS features but
cannot compose them with covariates; OpenFE composes but has no temporal
operators. BigFeat is the only one of the three that does both.

**Recipes, not values.** Storing features as replayable
(operator, source, params) recipes is what makes train/test consistency
*provable* — the fit≡transform invariant test exists only because the
representation allows it. Also the basis of interpretability claims.

**Confidence-scaled operator weighting.** Down-weighting TS operators when
detection confidence is low (clip(conf/0.7, 0.5, 2.0)) is graded trust —
better than the binary on/off most systems use. The same graded idea should
replace the cliff-shaped diversity penalty (§6).

**Elitism + importance selection + decay reweighting.** The generation loop
is a textbook-correct bandit-flavoured hill-climb: protect the best, explore
the rest, let credit fade. Nothing in the literature review suggests a
different loop — only the parameter shapes (§5, §6) need touching.

**Contiguous-block downsampling with warm-up padding.** Sampling blocks
rather than rows to preserve temporal continuity — with per-block history
prefixes and seam guards — is a genuinely thoughtful piece of design that
most large-scale TS pipelines omit entirely.

**The instinct to decline.** Restricted mode, trend mode, and the pre-flight
check all exist because the designer understood that hallucinated seasonality
is worse than none. The *instruments* need replacing (§2, §4), but the
safety-tier architecture they plug into is right, and rare in AutoFE.

---

## Priority order (merging with PIPELINE_FIXES_SPEC.md)

| Rank | Change | Verdict basis |
|---|---|---|
| 1 | Fixes 1–3 from the spec (ACF peaks, consensus stage, lags) | measured defects |
| 2 | §2 Replace stationarity gate with ADF/KPSS (optional statsmodels) | measured misclassification both directions |
| 3 | §3 Drop or calibrate the single-detector override; adopt AUTOPERIOD-style propose-and-verify inside the consensus stage | literature + scale mismatch |
| 4 | §8 FDR pre-filter before importance ranking | literature (FRESH) |
| 5 | §6 Graded diversity damping · §5 TS retry exemption · §1 LS detrend | measured, small |
| 6 | §4 Pre-flight scoring model · §7 adaptive depth | low priority |

## Proposed redesign — what I would build to make this pipeline powerful

Beyond repairing flaws, these are the changes I would make to the *design*.
Ordered by expected-power-per-cost. R1–R3 are cheap enough to land alongside
the spec fixes; R4–R6 are the next research increment; R7–R8 are
architecture-level and pay off mostly in the paper's ablation story.

### R1 — Detect on the target, not only the features *(cheap, likely the single biggest win)*

Verified: `_setup_time_series(self, X, y=None)` accepts `y` and never uses
it — the only reference is the docstring. Detection is entirely
**target-blind**, yet the periodicity that matters for prediction is the
target's. A feature column can carry a strong 24 h cycle while y follows a
weekly one; today the windows would be tuned to the wrong rhythm.

Change: run detection on `y` first (it is available in `fit`), and give
target-detected periods priority weight in the consensus stage; feature-column
periods remain as secondary evidence. One argument already plumbed — the fix
is to actually use it.

### R2 — Cyclical calendar encodings at detected periods *(cheap)*

Verified: no `sin/cos` operator exists (the two `sin` occurrences are
fallback shims inside `weekday_mean`/`month_mean`). Yet
`sin(2πt/P), cos(2πt/P)` at each detected period P is the standard
forecasting representation of seasonality — smooth, leak-proof by
construction (depends only on the timestamp), and it hands tree models the
phase information that raw lags express only indirectly. Add one operator
pair parameterised by the consensus fundamentals. This also gives the
detector output a second consumer, doubling the value of getting P right.

### R3 — FDR-controlled relevance pre-filter before importance ranking *(cheap; from §8)*

FRESH-style per-feature hypothesis tests with Benjamini–Yekutieli control,
applied to candidates *before* the RF/LGBM ranking. Kills
statistically-noise features that tree importances rank anyway, and directly
instruments investigation-plan hypothesis B2 (is selection the bottleneck?).

### R4 — One detection algorithm instead of three voters *(medium; subsumes spec Fix 2 + §3)*

Restructure detection as AUTOPERIOD-style **propose → verify → consense**:

1. DFT and Lomb-Scargle *propose* candidate periods (top-k spectral peaks;
   LS covers irregular sampling).
2. ACF *verifies*: a candidate survives only if it lands on an ACF hill —
   using ACF for what it is reliable at (confirmation) instead of what it is
   weak at (unsupervised peak-picking, the Fix-1 defect).
3. The consensus stage (spec Fix 2) clusters and de-harmonics survivors.

This converts the vote-combining problem (§3) into a pipeline where each
method plays its strong suit, and deletes the incommensurable-confidence
override entirely.

### R5 — Calibrate confidences into probabilities *(medium)*

Generate a synthetic corpus (planted periods × SNR × length, plus pure-noise
controls — the same harness as BENCHMARK_DESIGN Experiment A), fit a
per-detector isotonic map from raw score to P(genuinely periodic), and ship
the calibration curves as package data. After this, `confidence_threshold=0.5`
means the same thing on every detector, and the safety tiers (§9) key off a
real probability. Turns Experiment A from evaluation into a build step.

### R6 — Per-entity detection for panel data *(medium)*

Today one global window set serves every series in the frame; DFT samples 5
groups, ACF/LS one. A panel where store A is weekly and store B monthly gets
a compromise set that fits neither. The recipe representation already carries
per-operator params, so per-group windows are representable without format
changes. Design: detect per entity (or cluster entities by period signature
to bound cost), store `window_sizes` as {group → ladder}, fall back to global
for unseen groups at transform time.

### R7 — Successive-halving candidate evaluation *(medium)*

Generation currently scores every candidate with a full RF/LGBM fit per
iteration. Score on a 20% temporal subsample first, promote the top half,
re-score survivors on the full data. Standard multi-fidelity practice; the
saved budget funds a larger `gen_size`, which matters more than scoring
precision on early-round noise.

### R8 — Make the stages explicit objects *(architecture; enables the paper)*

`Detector → Consensus → Parameteriser → OperatorPool → Search → Selector` as
replaceable components rather than regions of a 664-line `fit`. The
scientific payoff is the **ablation table the paper needs**: each proposal
above becomes a row (pipeline with/without target-aware detection, with/
without verification stage, …) instead of a code fork. This is the
refactoring that was deliberately deferred in ARCHITECTURE.md — the ablation
requirement is the "specific need" that now justifies it.

### Explicitly considered and rejected

- **Learned/neural period detection** — needs labelled corpora we do not
  have, kills interpretability, and the classical estimators above are not
  the bottleneck.
- **Replacing the hill-climb with a full GA (crossover/mutation)** — the
  loop's simplicity is a strength (§ "what was designed right"); search is
  not where the measured losses are.
- **Wiring all 42 `local_utils` indicators now** — do it *after* R3 exists,
  so the FDR filter absorbs the noise a 3× pool expansion brings; behind the
  same confidence gating as the current TS operators.

---

## Corrections to earlier documents

- INVESTIGATION_PLAN.md D3 called the correlation acceptance test "a weak
  proxy for what the importance scorer rewards". Measured here: composed
  features pass the gate fine; the issue is only the parent-bar on
  autocorrelated targets. D3 should be read down accordingly.
- PIPELINE_GAPS.md's fixture remains the acceptance harness for ranks 1–3.

## Sources

Period detection (§3, R4):

- [AUTOPERIOD — On Periodicity Detection and Structural Periodic Similarity, Vlachos et al., SDM 2005](http://alumni.cs.ucr.edu/~mvlachos/pubs/sdm05.pdf) — the propose-on-periodogram / verify-on-ACF-hill pattern adopted in R4.
- [RobustPeriod: Robust Time-Frequency Mining for Multiple Periodicity Detection, Wen et al.](https://arxiv.org/pdf/2002.09535) — coupled time/frequency treatment of multiple simultaneous periods.

Feature selection (§8, R3):

- [tsfresh / FRESH: Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests, Christ et al., Neurocomputing 2018](https://www.sciencedirect.com/science/article/pii/S0925231218304843) — per-feature hypothesis testing with Benjamini–Yekutieli FDR control; the model for the R3 pre-filter. (The BY procedure itself: Benjamini & Yekutieli, *The control of the false discovery rate in multiple testing under dependency*, Annals of Statistics 29(4), 2001.)
- [tsfresh feature filtering documentation](https://tsfresh.readthedocs.io/en/latest/text/feature_filtering.html)

Cyclical encodings (R2):

- [Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.), §7.4 "Some useful predictors" — Fourier terms for seasonality](https://otexts.com/fpp3/useful-predictors.html) — the standard sin/cos representation of seasonal structure proposed as an operator pair in R2.

Multi-fidelity evaluation (R7):

- [Jamieson & Talwalkar, *Non-stochastic Best Arm Identification and Hyperparameter Optimization*, AISTATS 2016 (arXiv:1502.07943)](https://arxiv.org/abs/1502.07943) — successive halving.
- [Li et al., *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*, JMLR 2018 (arXiv:1603.06560)](https://arxiv.org/abs/1603.06560)

Confidence calibration (R5):

- [scikit-learn user guide: probability calibration (isotonic regression)](https://scikit-learn.org/stable/modules/calibration.html) — the calibration machinery proposed for mapping raw detector scores to P(periodic).

Stationarity testing (§2) uses the ADF and KPSS implementations from
`statsmodels` (`statsmodels.tsa.stattools.adfuller` / `kpss`), measured
directly in this review rather than cited.
