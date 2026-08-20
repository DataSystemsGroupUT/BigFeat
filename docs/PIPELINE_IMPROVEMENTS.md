# Pipeline Improvements — the Detection→Consumption Series

The consolidated record of the improvement series on the time-series
pipeline: eight commits, `a9ff2b9..9633dd0`, following the correctness round.

Companions: [CORRECTNESS_FIXES.md](CORRECTNESS_FIXES.md) (the defect round
that preceded this), [PIPELINE_GAPS.md](PIPELINE_GAPS.md) /
[PIPELINE_STAGE_REVIEW.md](PIPELINE_STAGE_REVIEW.md) (the audits that
motivated it), [PIPELINE_FIXES_SPEC.md](PIPELINE_FIXES_SPEC.md) (the
pre-registered designs, with amendments recorded as implementation taught
us better), [SYNTHETIC_STUDY.md](SYNTHETIC_STUDY.md) (the ground-truth
measurement).

**Distinction from the correctness round:** that round fixed code that was
*wrong* (leakage, corruption, inert seeds). This series improved code that
was *working as written but designed poorly* — harmonics selected over
fundamentals, smoothness mistaken for non-stationarity, detection blind to
the target. Every change here was driven by a measured failure and verified
by an acceptance test confirmed failing against the pre-change code.

---

## The pipeline, before and after

```
BEFORE  signal ─▶ ACF: tallest peaks (harmonics)     ─┐
                 DFT: single argmax (one period)      ├─▶ pooled windows,
                 LS : top-3                           ┘   quantile-truncated
        lag-1 autocorr > 0.85 gate (smoothness)           (drops fundamentals)
        target y: ignored                                 lags: positional picks
                                                          encodings: none

AFTER   signal + TARGET ─▶ ACF: first significant peak, echo-verified,
                                harmonic train masked
                           DFT: top-k spectral peaks, leakage-deduped
                           LS : top-3 (unchanged)
                                │
                    _consense_fundamental_days  (±15% clustering,
                                │                confidence-weighted)
              ┌─────────────────┼──────────────────┐
        windows (ladder +       lags               cyclical sin/cos
        pool, fundamentals      [1, P1, P2, P3]    encodings at each P
        NON-DROPPABLE)
        ADF stationarity gate (statsmodels optional; lag-1 fallback)
```

One consensus, three consumers — detection output feeds windows, lags and
phase encodings from a single protected list.

---

## The changes

### 1. ACF accepts fundamentals, not harmonics (`a9ff2b9`)

**Was:** peaks sorted by height. For multi-period signals the ACF at common
multiples *exceeds* the fundamentals (every component realigns there:
ACF(210)=0.997 vs ACF(7)=0.652 on a 7+30d signal), so the top-3 were
harmonics — 210/91/301 on the fixture; 360/120/330 on planted [30].

**Now:** candidates walked in ascending-lag order (the fundamental is the
first peak clearing the floor); each accepted lag's harmonic train is masked;
every candidate must show an ACF echo at a small multiple, which rejects
isolated noise spikes.

**Learned in implementation (spec amendments):** the echo rule is ANY-of
{2L, 3L}, not both — in a multi-period signal the other component can sit
anti-phase at exactly 2L and cancel the echo ({12,52}: ACF(24)=0.018). And
ACF *cannot* recover the second period by physics: the slow fundamental is
not an ACF local maximum at all (the fast comb's tooth beside it towers over
it), so the lag-domain contract is shortest-fundamental-first and
multi-period recovery belongs to the ensemble.

### 2. DFT proposes top-k spectral peaks; the ladder protects fundamentals (`830583a`)

**Was:** a single `np.argmax` — only the strongest component ever proposed.
And `_generate_multiscale_windows` truncated ascending, so a detected period
could be dropped in favour of its own derived sub-harmonics (detected
{8,11,30} → ladder [4,5,8,11,15,16,22,30,…] cut at slot six, losing 30).

**Now:** peaks above a 3×-median noise floor, **deduplicated in period space**
(spectral leakage splits one peak across adjacent FFT bins — without dedup
the top-3 were three lobes of the same peak), top-k by height. Height-sorting
is *correct* in the frequency domain — fundamentals exceed their harmonics —
the exact opposite of the lag domain, which is why changes 1 and 2 point in
opposite directions. Ladder truncation now drops derived harmonics before
detected periods.

Landed together because 2 is invisible without the ladder guard: the second
period DFT newly detects was being truncated by the ladder built from it.

### 3. Lags come from detected fundamentals (`20cf981`)

**Was:** `lag_periods = [windows[0], windows[1], windows[mid]]` — positional
picks from the pooled ladder. On the fixture: `[1, 3, 7]`, the 30-day
fundamental absent and 7 present only by accident of position.

**Now:** each detector stashes its raw `(period, confidence)` pairs
(`last_detected_periods`); both setup paths derive lags from the clustered
fundamentals: `[1 step, P1, P2, …]`. A lag should EQUAL a detected cycle
("same point one cycle ago"); a window merely spans one. Fixture:
`[1,3,7] → [1,30,7]`.

The full consensus machinery of the original Fix-2 design was **deferred**
here with the recorded note *"no machinery without a measured failure it
would fix"* — changes 1+2 had already cured the pooling pollution at its
source (`[1,4,7,14,105,210] → [1,3,5,7,28,41]` end-to-end).

### 4. Stationarity gate: ADF instead of lag-1 autocorrelation (`4fc84d7`)

**Was:** restricted mode triggered by mean |lag-1 autocorr| > 0.85 — a
measure of *smoothness*, wrong in both directions: it fired on a clean
STATIONARY 30-day seasonal (lag-1 ≈ 0.97), stripping rolling operators from
exactly the data the subsystem exists for, while a noisy random walk
(covid_deaths: 0.725) slipped under to the full pool at a measured 5× MASE
cost.

**Now:** Augmented Dickey–Fuller per sampled column; non-stationary when the
median p fails to reject the unit root. `statsmodels` optional
(`bigfeat[stationarity]`); the lag-1 heuristic remains as explicit fallback
and `avg_lag1` stays exposed. The trend-mode's secondary lag-1 trigger folded
into the same verdict. ADF classified all six canonical test series
correctly where the old rule got two wrong.

### 5. R1 — detection sees the target (`ea88e10`)

**Was:** `_setup_time_series(X, y)` accepted `y` and never used it (the only
reference was the docstring). Detection was target-blind: features with a
5-day cycle and a target with an 11-day cycle produced lags `[1, 5]` — the
predicted signal's rhythm invisible.

**Now:** in `'auto'` mode the detectors analyse a local frame with `y` as a
synthetic first column, feeding the gate, the vote, the windows and the lag
stash. Strictly local — it cannot leak into feature generation or
`transform()`. Guarded for length mismatch / non-numeric / all-NaN targets.

### 6. R2 — cyclical sin/cos encodings at detected periods (`4388af0`)

**Was:** no phase encoding existed anywhere in the pool, despite
`sin/cos(2πt/P)` being the standard regression representation of seasonality
(fpp3 §7.4).

**Now:** two operators whose phase is a pure function of the timestamp
against a fixed epoch — same date, same value, at fit or transform, any row
order, any entity: leak-proof by construction. Periods drawn from the
detected fundamentals. Pool 15→17 in full mode only. Recipe replay handled
the new operators with zero changes — the recipes-not-values design paying
off.

### 7. Both synthetic-study gaps closed (`9633dd0`, after `ca20e89` measured them)

The study exposed that the pooled-window quantile subsample could still drop
a fundamental that detection had recovered (6/78 cases — the lags proved the
period was found), and that pair lags re-clustered to top-2 (6/24 recovery).
This was the **measured failure that lifted change 3's deferral**: the pooled
path got the same fundamental protection as the ladder, and lags now consume
the same protected 3-fundamental list as windows and encodings. The old lag
deriver became an orphan within its own series and was removed.

---

## Measured end to end

Ground truth: 78 planted-period cases ({7, 30, 91, 7+30, 11+31} × 3 SNR ×
3 lengths × 2 seeds) + 6 noise controls, identical data per version
([SYNTHETIC_STUDY.md](SYNTHETIC_STUDY.md)):

| Metric | Before series | After changes 1–6 | After change 7 |
|---|---|---|---|
| Windows recover ALL planted periods | 69% | 92% | **100%** |
| Lags contain ALL planted periods | 46% | 77% | **100%** |
| False positives on pure noise | 0/6 | 0/6 | **0/6** |

Attribution is per-commit: pairs 5/24→22/24 (changes 1+2), single-period
lags 36/54→54/54 (change 3), low-SNR 16/26→25/26 (change 1's verification).
The gains were not bought with sensitivity — noise rejection never moved.

**Caveat, stated where the score is:** 100% is a property of THIS grid
(regular sampling, sinusoids, ±15% tolerance). The next move for the study
is extending the grid — irregular sampling, non-sinusoidal shapes,
amplitude drift — not celebrating the number.

The running fixture (planted 7+30 days) across the series:

| Quantity | Before | After |
|---|---|---|
| ACF accepted lags | 210, 91, 301 | 7 (30 via ensemble) |
| Ensemble windows | 1, 4, 7, 14, **105, 210** | brackets 7 AND 30, none > 60 |
| `lag_periods` | 1, 3, 7 | 1, 7, 30 |
| Planted [11, 31] via DFT | 11 only | 11 and 31 |

## Method notes

- **Every change was test-first**, with the acceptance test confirmed
  failing against the pre-change code. Two test fixtures were themselves
  caught being wrong this way: the spec's [11,45] pair passed on unfixed
  code because 4×11=44 sat inside the 45-day tolerance (the ladder faked
  the recovery), and an early ACF test demanded a recovery the lag domain
  cannot deliver by physics.
- **Deferral with a tripwire worked as designed.** The consensus machinery
  was refused twice for lack of a measured failure, then implemented in
  exactly the slice the synthetic study demanded — on the day the study
  produced the evidence.
- **Golden movements were diagnostic.** Changes touching only misclassified
  routes left the goldens intact (ADF, R1); the pool-size change moved
  exactly the full-pool golden and not the restricted-mode one.

## Not changed, deliberately

Deferred with reasons on record: full confidence-weighted consensus beyond
the protected-fundamentals slice (R5 calibration is its prerequisite),
per-entity windows (R6), successive-halving evaluation (R7), the FDR
selection pre-filter (R3 — next candidate, instruments hypothesis B2), the
diversity-penalty and retry-budget shapes (§6/§5), and wiring
`local_utils.py`'s 42 indicators (waits on R3 so the filter can absorb a
3× pool expansion).
