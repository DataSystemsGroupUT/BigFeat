# ⚠ These results are stale

**Run dates:** 2026-01-24 to 2026-02-12
**Correctness fixes landed:** 2026-08-04

Every BigFeat number in this directory was produced by code that has since been
found to have six defects making its output wrong rather than merely
suboptimal — including rolling windows that spanned the wrong time range on
**100% of rows across all 25 datasets**, a seasonality guard that had **never
once executed**, and a `random_state` parameter that did nothing.

See [`../docs/CORRECTNESS_FIXES.md`](../docs/CORRECTNESS_FIXES.md).

## What this means

| File | BigFeat columns | `openfe` / `tsfresh` / `baseline` columns |
|---|---|---|
| `benchmark_summary.csv` | **Do not cite** | Still valid |
| `*_results.json` | **Do not cite** | Still valid |
| `average_rankings.csv`, `best_config_counts.csv` | **Do not cite** — rankings mix both | — |
| `cd_diagram.*`, `*_impact.*`, `efficiency_frontier.*` | **Do not cite** — derived from BigFeat rows | — |

`openfe`, `tsfresh` and `baseline` never call into BigFeat, so the fixes cannot
have changed them. Those columns remain a valid measurement against the same
datasets and harness.

The figures in particular are misleading: `detector_confidence_impact` and
`stationarity_impact` plot quantities that the fixes changed substantially
(DFT's confidence metric measured roughly the inverse of what it claimed; ACF
reported white noise as periodic at 0.973 confidence).

## Replacing them

See [`../docs/INVESTIGATION_PLAN.md`](../docs/INVESTIGATION_PLAN.md) Phase A.
Do not delete this directory until the replacement run exists — the comparison
between old and new is itself informative about what the fixes changed.
