# Pipeline Audit — Gaps Between Detection and Feature Generation

An instrumented walk through the time-series pipeline, looking for information
lost between stages. Every finding below was measured, not inferred from
reading; the probe scripts are reproducible from the listings.

Pipeline under audit:

```
signal -> peak extraction -> period sanitisation -> multiscale ladder
       -> ensemble pooling -> window set -> lag derivation -> generation
```

**Headline:** the stages are individually reasonable, but the seams between
them discard exactly the information the next stage needs. On a clean signal
with two planted periods (7 and 30 days), the ensemble's final window set is
`[1, 4, 7, 14, 105, 210]` — the 30-day period is lost entirely and two of six
slots carry spurious harmonics.

---

## Finding 1 — ACF selects harmonics over fundamentals (defect)

**Where:** `acf_window_detector.py`, `_find_acf_peaks` — peaks are sorted by
height and the top 3 kept.

**Why it is wrong:** for a signal with periods P and Q, the ACF at common
multiples (n·P, n·Q, lcm) is *higher* than at the fundamentals, because every
component realigns there. On the 7+30-day test signal, ACF(210)=0.997 vs
ACF(7)=0.652 — so height-sorting picks 210, 91, 301.

**Measured:** across five planted-period cases, the tallest-3 rule misses the
fundamental in **3 of 5**, returning multiples instead (360=12·30, 210=7·30,
156=12·13):

| Planted | Top-3 by height | Fundamental found |
|---|---|---|
| [7] | 203, 7, 91 | yes |
| [30] | 360, 120, 330 | **no** |
| [7, 30] | 210, 119, 329 | **no** |
| [12, 52] | 156, 312, 360 | **no** |
| [24, 168] | 336, 168, 360 | yes |

**Fix direction:** prefer the *lowest-lag* significant peak, not the tallest —
the fundamental is always the first peak clearing the significance floor.
Standard practice is to take the first peak above a confidence band, then
verify its multiples also show peaks (which distinguishes a true period from a
noise spike). Cost: small, localised to `_find_acf_peaks`.

## Finding 2 — Pooling ignores confidence (design gap; the detection→pool seam)

**Where:** `bigfeat_base.py`, the `'auto'` pooling block.

**What happens now:** windows from every detector that voted "periodic" are
merged with equal weight, deduped, and quantile-sampled down to `n_windows`.
A window from a detector at confidence 0.99 counts exactly the same as one
from a detector at 0.31 — and quantile sampling then *preserves the extremes*,
so one bad detector's long-harmonic junk is guaranteed representation.

**Measured end-to-end:** on the 7+30-day signal, DFT and LS both return clean
sets bracketing the truth (`[4,7,14,28]`, `[4,7,14,15,28,30]`); ACF returns
`[46,91,105,150,182,210]`. The pooled result is `[1,4,7,14,105,210]`:
ACF's junk survives, the 30-day period does not.

**Fix direction — this is the missing processing stage the pooling seam
needs.** Before laddering/pooling, cluster the pooled *periods* (not windows)
across detectors:

1. Collect raw detected periods with their per-detector confidence.
2. Cluster periods that agree within ±15% across detectors; a period seen by
   two detectors is far more credible than a tall peak seen by one.
3. Suppress near-multiples: if P and k·P are both present, keep P (the
   fundamental) and let the ladder regenerate the harmonics deliberately.
4. Weight surviving periods by summed confidence, THEN build the ladder.

This inserts a *period-consensus stage* between extraction and ladder
generation — currently consensus is only applied to the binary
periodic/not-periodic vote, never to the periods themselves.

## Finding 3 — Lag periods never see the detected periods (design gap)

**Where:** `bigfeat_base.py:399` — `lag_periods = [windows[0], windows[1],
windows[mid]]`.

**What happens now:** lags are picked *positionally* from the pooled window
ladder. On the 7+30 signal the generation stage received lags `[1, 4, 14]` —
neither detected period is among them, and lag-7 is the single most valuable
feature for weekly-periodic data (the "same day last week" comparison).

**Why the seam matters:** windows and lags serve different purposes. A window
wants to *span* a cycle (rolling mean over one week); a lag wants to *equal*
one (compare to exactly one cycle ago). Deriving lags from the window ladder
conflates the two.

**Fix direction:** set `lag_periods` from the detected fundamentals directly —
`[1 step, P1, P2, ...]` — and keep the ladder for windows only. Cost: small.

## Finding 4 — DFT keeps one peak; ACF/LS keep three (asymmetry)

**Where:** `dft_window_detector.py:249` uses `np.argmax` (single dominant
peak); ACF and LS keep top-3.

**Measured impact:** smaller than expected. On the two-period signal DFT still
recovers both, because 28 ≈ 4·7 appears via the harmonic ladder — the 30-day
component is found by accident of arithmetic, not by detection. On periods
that are not near-multiples of each other (e.g. 11 and 45) this luck runs out.

**Fix direction:** have DFT return the top-k spectral peaks above the noise
floor (it already computes the full magnitude spectrum; only the argmax line
discards it). Then all three detectors feed the same consensus stage from
Finding 2.

## Finding 5 — The ladder can outvote the data (interaction, lower priority)

`_generate_multiscale_windows` expands each period into {P/2, P, 2P, 4P}. With
several detected periods the candidate set easily exceeds `n_windows=6`, and
the quantile subsample then decides which *derived* harmonics survive at the
expense of *detected* fundamentals. After Findings 1–2 the input to the ladder
is clean enough that this mostly resolves itself, but the ladder should mark
fundamentals as non-droppable during subsampling.

---

## What is NOT broken

For balance, seams checked and found sound:

- **Sampling-rate inference** correctly recovers M/Q/Y/D from timestamp
  spacing (verified on Monash data).
- **Time-based rolling** agrees exactly with ground-truth pandas rolling at
  every frequency (error 0.0000; see CORRECTNESS_FIXES.md §2.9).
- **The binary periodic/not-periodic consensus** behaves correctly: enables on
  clean periodic signals, declines on noise.
- **Causality and entity isolation** hold across all operator paths (91 tests).

---

## Recommended order of work

| # | Change | Type | Cost | Depends on |
|---|---|---|---|---|
| 1 | ACF: first-significant-peak rule + multiple-verification | defect fix | small | — |
| 2 | Period-consensus stage (cluster, de-harmonic, confidence-weight) | new stage | medium | 1 |
| 3 | Lags from detected fundamentals, not window positions | defect fix | small | 2 |
| 4 | DFT top-k peaks | enhancement | small | 2 |
| 5 | Fundamentals non-droppable in ladder subsampling | enhancement | small | 2 |

Item 2 is the answer to "is any more processing needed between period
extraction and window pool generation": yes — a consensus stage over the
periods themselves, which none of the current stages performs.

**Verification:** the synthetic study from
[BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md) Experiment A doubles as the test
harness for all five changes — planted periods give ground truth for exactly
the recovery rates measured here. The two-period case above should become a
fixture: current recovery is 7d=yes / 30d=no; after 1–3 both must be yes.
