# Implementation Spec — Pipeline Fixes 1–5

Concrete design for the five changes recommended in
[PIPELINE_GAPS.md](PIPELINE_GAPS.md). This is the working spec for the
implementation, kept local until the team has reviewed it.

Shared regression fixture for the whole series (from the audit):

```python
# planted periods 7 and 30 days, both strong, SNR high
n = 730; t = np.arange(n)
sig = 10*np.sin(2*np.pi*t/7) + 8*np.sin(2*np.pi*t/30) + rng.randn(n)*0.5
```

Acceptance for the series as a whole: end-to-end `'auto'` windows must bracket
BOTH periods (today: `[1,4,7,14,105,210]` — 7 yes, 30 no), and `lag_periods`
must contain both fundamentals (today: `[1,4,14]` — neither).

---

## Fix 1 — ACF: first-significant-peak rule

**File:** `bigfeat/acf_window_detector.py`, `_find_acf_peaks`.

**Problem (measured):** peaks sorted by height; for multi-period signals the
ACF at common multiples exceeds the fundamentals (ACF(210)=0.997 vs
ACF(7)=0.652 on the fixture). Fundamental missed in 3 of 5 planted cases.

**Design.**

1. Keep `find_peaks` with the existing height/prominence floors.
2. Replace the height sort with **ascending-lag order**: the fundamental is
   the first peak clearing the floor.
3. **Multiple-verification:** accept peak at lag L only if the ACF also shows
   local maxima near 2L (and 3L when 3L <= max_lag), within +-15% of the
   multiple and above height/2. A true period repeats at its multiples; a
   noise spike does not.
4. After accepting L, **mask its harmonic train** (k*L +- 15%) and repeat on
   the remaining peaks to find a second independent period. Keep at most 3
   accepted fundamentals (unchanged cap).
5. Confidence for each accepted period = its ACF height (unchanged scale).

**Acceptance tests** (add to `tests/test_detectors.py`):

- Fixture: accepted lags contain 7 and 30 (+-1), and contain NO lag > 60.
- Planted [30] alone: top accepted lag in 27..33 (today: 360).
- Planted [12, 52]: both recovered (today: neither).
- Existing white-noise tests stay green (rule change must not create false
  positives; the verification step should *reduce* them).

**Non-goals:** do not touch `_compute_acf` (the overlap bound from the
correctness round stays as is).

**Amendments from implementation (2026-08-19):**

1. *Echo rule is ANY-of {2L, 3L}, not all.* In a multi-period signal the
   other component can sit near anti-phase at exactly 2L and cancel the
   echo: planted {12, 52} gives ACF(24) = 0.018 (cos(2*pi*24/52) ~ -0.97)
   while 3L = 36 shows 0.322. Requiring 2L rejects true fundamentals.
2. *ACF standalone cannot recover the SECOND period, by physics.* The slow
   fundamental is not an ACF local maximum at all -- the fast component's
   comb tooth adjacent to it towers over it (28 vs 30; 48/60 vs 52). The
   lag-domain contract is therefore: shortest fundamental first, harmonics
   masked, noise spikes rejected. Both-period recovery is the ensemble's
   job, which REORDERS THE SERIES: Fix 4 (DFT top-k) must land before
   Fix 2's fixture acceptance can pass, since DFT's single argmax also
   returns only the stronger component today.
3. Acceptance tests adjusted accordingly (shortest-first + no-harmonics
   instead of both-recovered for the standalone ACF tests); the
   both-periods assertion moves to the Fix-2 end-to-end test.

---

## Fix 2 — Period-consensus stage before laddering

**File:** `bigfeat/window_detector_base.py` (new shared method) plus the
`'auto'` pooling block in `bigfeat_base.py` (`_setup_time_series`).

**Problem (measured):** pooling merges *windows* from all yes-voting
detectors with equal weight; quantile sampling preserves extremes, so one bad
detector's harmonics are guaranteed slots. Fixture end-to-end:
`[1,4,7,14,105,210]`.

**Design.** New method on `BaseWindowDetector` (module-level function is also
fine):

```python
def consense_periods(candidates):
    """candidates: list of (period_days, confidence, detector_name).
    Returns list of (period_days, weight) fundamentals."""
```

1. **Cluster:** sort by period; greedily group candidates within +-15%
   relative tolerance. Cluster period = confidence-weighted mean.
2. **Score:** cluster weight = sum of member confidences; +50% bonus if >= 2
   distinct detectors are in the cluster (cross-method agreement is the
   strongest signal available).
3. **De-harmonic:** for clusters P < Q, if Q/P is within +-10% of an integer
   2..6, fold Q's weight *into* P and drop Q. The fundamental keeps the
   evidence; the ladder will regenerate deliberate harmonics.
4. **Select:** keep clusters by descending weight, max 3.

**Wiring change in `_setup_time_series` ('auto' path):** instead of calling
each yes-voter's `smart_window_selection` and pooling the returned windows,
collect each detector's raw `(periods, confidences)` from
`detect_optimal_windows`, run `consense_periods`, and build ONE ladder from
the consensus fundamentals. Delete the quantile subsample of pooled windows
(the ladder of <=3 fundamentals is already <= n_windows in practice; if it
exceeds, see Fix 5).

**Amendment from implementation (2026-08-19):** after Fixes 1, 4 and 5
landed, the fixture's window acceptance ALREADY passes end-to-end
(`[1,3,5,7,28,41]` -- both periods bracketed, no junk): the pooling
pollution was cured at its source, garbage-in rather than pooling-rule.
The full consense_periods machinery is therefore DEFERRED pending a
measured failure it would fix; **update 2026-08-19 (later): the synthetic
study (docs/SYNTHETIC_STUDY.md, gap 1) has now measured exactly such a
failure -- the pooled-window quantile subsample drops fundamentals that
detection recovered (6/78 cases). The deferral is lifted for the narrow
slice that protects fundamentals in the pooled path;** what survives of Fix 2 is the
`last_detected_periods` stash (implemented, used by Fix 3) and the
end-to-end regression test pinning the window set. Revisit alongside
R5 calibration if confidence-weighting becomes measurable.

**Interface note:** detectors currently return `(windows, confidence_dict)` —
the raw periods are internal. Smallest change: have `detect_optimal_windows`
also stash `self.last_detected_periods` (list of (period, conf)); the
ensemble reads that. Avoids breaking the public return shape used by the
'yes' path and the tests.

**Acceptance tests:**

- Unit: `consense_periods([(7,.9,'dft'),(7.3,.8,'ls'),(210,.99,'acf'),
  (30,.7,'ls'),(28,.6,'dft')])` -> fundamentals {7ish, 29ish}, no 210.
- Fixture end-to-end: windows bracket both 7 and 30; nothing > 60.
- Noise: still declines (consensus of zero periods -> unchanged fallback).
- Goldens: `reg_ts_periodic` / `reg_ts_nonstationary` will move — regenerate
  deliberately and explain in the commit.

---

## Fix 3 — Lags from detected fundamentals

**File:** `bigfeat_base.py` — both copies of the positional derivation
(`:399` and `:631`; collapse into one helper while there).

**Problem (measured):** `lag_periods = [w[0], w[1], w[mid]]` gave `[1,4,14]`
on the fixture; neither detected period present.

**Design.**

```python
def _derive_lag_periods(self, fundamentals):
    lags = [pd.Timedelta(days=1)]          # always: previous observation
    lags += [pd.Timedelta(days=round(p)) for p, _ in fundamentals[:2]]
    return dedupe_preserving_order(lags)[:3]
```

Windows *span* a cycle; lags *equal* one. When no fundamentals exist
(defaults path, 'yes' with user windows), fall back to the current positional
rule so behaviour there is unchanged.

**Acceptance:** fixture -> `lag_periods` contains 7d and 30d. Trend-mode path
(`[1d]`) unchanged.

---

## Fix 4 — DFT returns top-k peaks

**File:** `bigfeat/dft_window_detector.py:249` (the `np.argmax`).

**Design:** run `scipy.signal.find_peaks` on the magnitude spectrum with
`height = median(mags) * 3` (same noise-floor logic as the confidence
metric), take up to 3 peaks by height — height-sorting is CORRECT in the
frequency domain, where harmonics are weaker than fundamentals (unlike lag
domain; this asymmetry is why Fix 1 and Fix 4 go opposite directions —
worth a comment in code). Convert each to days as now; feed all into the
consensus stage.

**Acceptance:** planted [11, 45] (non-multiples, the case the ladder cannot
rescue): DFT alone recovers both within +-15%.

---

## Fix 5 — Fundamentals are non-droppable in the ladder

**File:** `window_detector_base.py`, `_generate_multiscale_windows`.

**Design:** ladder returns fundamentals first, derived harmonics after; any
truncation to `n_windows` removes derived entries before fundamentals.
Smallest implementation: build `fundamental_windows` and `derived_windows`
separately, concatenate, truncate from the tail of `derived`.

Largely defensive once Fix 2 lands (<=3 fundamentals x {/2, x2, x4} can still
exceed 6). Acceptance: with fundamentals {7, 30, 91} and n_windows=4, all
three fundamentals survive.

---

## Sequencing and verification

```
Fix 1 (ACF)  ->  Fix 2 (consensus)  ->  Fix 3 (lags)
                     |
Fix 4 (DFT top-k) ---+---> Fix 5 (ladder guard)
```

One commit per fix, each with its acceptance tests, each verified to FAIL
against the pre-fix code before committing (the lesson from the correctness
round: a regression test proves nothing until seen failing for the right
reason). Goldens regenerate once, at Fix 2, with the diff explained.

Expected end state on the fixture:

| Quantity | Today | After |
|---|---|---|
| ACF accepted lags | 210, 91, 301 | 7, 30 |
| Ensemble windows | 1,4,7,14,105,210 | brackets 7 AND 30, none > 60 |
| lag_periods | 1, 4, 14 | 1, 7, 30 |
| Planted [11,45] via DFT | 11 only | 11 and 45 |
