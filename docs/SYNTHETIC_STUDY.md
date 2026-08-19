# Synthetic Study — Detection Quality Before and After the Improvement Series

Experiment A of [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md), first run. Series
with KNOWN planted periods are swept over period structure, noise and length,
plus pure-noise controls, and both code versions are scored on identical data:

- **PRE** = `b05b103` (all correctness fixes, before the improvement series)
- **POST** = `4388af0` (Fixes 1/3/4/5, ADF gate, R1, R2)

Harness: `testing/Benchmarking/synthetic_study.py`. Raw results:
`benchmark_results_synthetic/`. Grid: {7, 30, 91, 7+30, 11+31} × SNR
{0.5, 2.5, 5.0 noise on amplitude 10} × {3, 5, 10 cycles} × 2 seeds
= 78 planted cases + 6 noise controls per version. Recovery tolerance ±15%.

## Headline

| Metric | PRE | POST (fix series) | POST (+ gap fixes) |
|---|---|---|---|
| Windows recover ALL planted periods | 54/78 (69%) | 72/78 (92%) | **78/78 (100%)** |
| Lags contain ALL planted periods | 36/78 (46%) | 60/78 (77%) | **78/78 (100%)** |
| TS enabled on planted signals | 77/78 | 78/78 | 78/78 |
| **False positives on pure noise** | **0/6** | **0/6** | **0/6** |

The improvement is not bought with sensitivity: noise rejection is unchanged
at every stage. The third column is the study doing its job as an acceptance
harness -- both gaps it exposed were closed the same day
(`benchmark_results_synthetic/post_gapfix.json`): fundamentals are now
protected through the pooled-window subsample exactly as the ladder protects
them (Fix 5, one level up), and the lag list consumes the same protected
3-fundamental consensus as the windows and the cyclical encodings -- one
detection output, three consumers, no re-clustering.

A perfect score on a synthetic grid is a statement about THIS grid (regular
sampling, sinusoidal components, ±15% tolerance), not about detection in
general -- the grid should now be extended (irregular sampling, non-sinusoidal
shapes, amplitude drift) rather than celebrated.

## Attribution by condition

| Condition | PRE win | POST win | PRE lag | POST lag |
|---|---|---|---|---|
| single periods | 49/54 | 50/54 | 36/54 | **54/54** |
| period PAIRS | 5/24 | **22/24** | 0/24 | 6/24 |
| high SNR | 19/26 | 23/26 | 13/26 | 20/26 |
| mid SNR | 19/26 | 24/26 | 13/26 | 20/26 |
| low SNR | 16/26 | **25/26** | 10/26 | 20/26 |
| 3 cycles | 15/18 | 18/18 | 8/18 | 18/18 |
| 5 cycles | 22/30 | 27/30 | 14/30 | 24/30 |
| 10 cycles | 17/30 | 27/30 | 14/30 | 18/30 |

Reading the deltas against the commits:

- **Pairs 5/24 → 22/24** is Fixes 1+4+5 (ACF fundamentals, DFT top-k, ladder
  protection): multi-period signals were the measured blind spot.
- **Single-period lags 36/54 → 54/54 (perfect)** is Fix 3: lags now EQUAL the
  detected cycle instead of being positional ladder picks.
- **Low SNR 16/26 → 25/26** is the echo-verification step of Fix 1: noise
  spikes no longer masquerade as periods, so the surviving candidates are
  real ones.

## Two honest gaps the study exposed — CLOSED same day, see headline

**1. The window subsampling can still drop a fundamental — Fix 2's deferred
consensus stage now has its measured failure.** In 6 remaining window
failures (e.g. `single_7`, 10 cycles: windows `[1,4,9,16,31,35]` while the
LAGS correctly contain 7), the fundamental survives detection — the lag
derivation proves it — but the ensemble's pooled-window quantile subsample
drops it. Fix 2 was deferred with the note "no machinery without a measured
failure it would fix" (PIPELINE_FIXES_SPEC.md). This is that failure: the
pooled-window path needs the same fundamental-protection the ladder got in
Fix 5, which is a slice of the consensus design.

**2. Pair LAGS remain weak (0/24 → 6/24).** Windows recover pairs at 92% but
the lag list holds only the top-2 confidence clusters, and with two true
periods plus residual harmonic candidates in the stash, the second
fundamental often ranks third. Candidate fix: derive lags from the SAME
protected fundamentals list the windows now use (`detected_fundamentals`,
max 3) rather than re-clustering to 2 — one line, but it belongs with the
gap-1 fix so both consume one consensus output.

## Scope notes

Regular sampling only (the fixes under test do not touch irregularity;
Lomb-Scargle's irregular-sampling advantage needs its own sweep). Two seeds;
the pre/post deltas (23pp, 31pp) dwarf seed noise at this grid size, but the
per-condition rows carry n as fractions for that reason. This study doubles
as the calibration corpus scaffold for R5.
