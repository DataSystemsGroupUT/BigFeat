# Experiment B Pilot — UCI Bike Sharing

First multivariate accuracy measurement on the fixed code
([BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md) Experiment B). Harness:
`testing/Benchmarking/pilot_bike.py`, analysis pre-registered in its header
before any result existed. Raw results: `benchmark_results_pilot/`.

Setup: 17,378 hourly rows (2011–2012), temporal 80/20 split, identical
RandomForest everywhere, every arm sees the same information set (weather at
t, target history strictly before t). Five BigFeat seeds; the two non-BigFeat
arms are deterministic. Primary metric MAE.

## Pre-registered run

| Arm | MAE | vs baseline |
|---|---|---|
| baseline (weather + calendar + `cnt_prev`) | **31.38** | — |
| handcrafted (+ lag/roll 24 h, 168 h) | 33.10 | worse |
| bigfeat_no · 5 seeds | 31.56 [31.21–32.00] | ≈ tie |
| bigfeat_auto · 5 seeds | 32.74 [31.24–33.92] | worse |

**Verdict under the pre-registered rule** (mean < handcrafted AND seed max <
handcrafted): **fails** — the mean passes, the worst seed (33.92) does not.

The most informative row is `handcrafted`: the competent-engineer features
*hurt* (31.38 → 33.10). On this dataset **no feature engineering helps, human
or automated**, when the baseline already holds `cnt_prev` plus calendar
columns — the OpenFE-ties-baseline pattern reproduced on multivariate data,
and direct support for hypothesis B1 (a tree ensemble with a strong
information set absorbs temporal structure internally; added columns dilute).

Detection itself was correct on every seed — windows `[1,2,3,7,…]` days, the
true daily and weekly cycles — but *redundant*: the baseline's `hr` and
`weekday` columns encode those cycles for free. Detection quality and
detection usefulness separated cleanly.

## Exploratory run — calendar removed (post-hoc, clearly fenced)

The pre-registered result begs one question: does detection pay when the
temporal structure is NOT hand-encoded? Variant: drop the cyclic calendar
columns (`hr`, `weekday`, `mnth`, `season`) from **every** arm
(`--no-calendar`; non-cyclic context and weather retained).

| Arm (no calendar) | MAE | vs its baseline |
|---|---|---|
| baseline | 81.00 | — |
| bigfeat_no · 5 seeds | 80.88 [80.33–81.05] | ≈ tie |
| handcrafted | 46.17 | −43% |
| **bigfeat_auto · 5 seeds** | **33.36 [29.77–38.08]** | **−59%** |

Applying the same decision rule to this run: mean 33.36 < 46.17 **and** worst
seed 38.08 < 46.17 — **passes both prongs**. Every seed detected the true
periods (windows `[1,2,3,5,8,97]` days; lags `[1,3,5]`).

The comparison that matters most spans the two runs:

> **bigfeat_auto with no calendar knowledge (33.36) lands within 6% of the
> full calendar-equipped baseline (31.38), and its two best seeds beat it
> (29.77, 29.84).** Automated detection substituted almost completely for
> expert temporal encoding — and recovered far more of it than the
> hand-crafted lag/rolling set did (46.17).

## What the pilot establishes

1. **The value proposition is real but bounded.** BigFeat's time-series
   pipeline does not beat a baseline that already encodes the temporal
   structure — nothing does, on this data. Its measured value is
   **substituting for temporal feature engineering that hasn't been done**:
   −59% MAE over the uninformed baseline, −28% over a competent hand.
2. **This is the honest framing for the paper**: not "beats all baselines"
   but "automatically recovers what an expert would encode, verified against
   ground truth (SYNTHETIC_STUDY.md) and now priced on real data."
3. **Seed variance is the cost to report**: the auto arm spans 8.3 MAE across
   seeds in the no-calendar run (29.8–38.1). Real, and honest error bars are
   mandatory in any headline number.
4. `bigfeat_no` tracked its baseline in both runs: arithmetic composition
   contributed nothing here; the entire effect is the time-series pipeline.

## Caveats

One dataset; one downstream model (RF); the exploratory run is post-hoc and
labelled as such — it sets the hypothesis for the remaining Experiment B
datasets (Air Quality, Appliances Energy, Electricity), where the
calendar-ablated design should be pre-registered from the start. Fit cost:
190–275 s per auto fit at this size.
