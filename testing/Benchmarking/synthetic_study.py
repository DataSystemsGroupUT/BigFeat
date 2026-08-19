"""Experiment A: synthetic study with planted periods (BENCHMARK_DESIGN.md).

Ground-truth evaluation of the detection pipeline: series with KNOWN planted
periods, swept over period structure, noise and length, plus pure-noise
controls. Because the truth is known, detection can be scored directly --
recovery rate, lag fidelity, false-positive rate -- rather than inferred from
downstream model accuracy.

Usage:
    python synthetic_study.py <bigfeat_import_root> <label> <out.json>

The import-root argument lets the same harness score two checkouts of the
library (e.g. before/after an improvement series) on identical data.
"""
import json
import sys
import warnings

warnings.filterwarnings('ignore')

IMPORT_ROOT, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, IMPORT_ROOT)

import numpy as np
import pandas as pd

import bigfeat.bigfeat_base as bb
assert bb.__file__.startswith(IMPORT_ROOT), f"wrong bigfeat: {bb.__file__}"

TOL = 0.15          # a period P counts as recovered within +-15%
SEEDS = (0, 1)

# --- the grid -------------------------------------------------------------
# (name, planted_periods, amplitudes)
SIGNALS = [
    ("single_7",   [7],        [10]),
    ("single_30",  [30],       [10]),
    ("single_91",  [91],       [10]),
    ("pair_7_30",  [7, 30],    [10, 8]),
    ("pair_11_31", [11, 31],   [10, 7]),
]
NOISE = {"high_snr": 0.5, "mid_snr": 2.5, "low_snr": 5.0}
CYCLES = (3, 5, 10)


def make(planted, amps, noise, cycles, seed):
    rs = np.random.RandomState(seed)
    n = max(60, int(max(planted) * cycles))
    t = np.arange(n)
    sig = sum(a * np.sin(2 * np.pi * t / p) for p, a in zip(planted, amps))
    sig = sig + rs.randn(n) * noise
    X = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="D"),
        "v": sig,
        "u": rs.rand(n),          # a non-temporal covariate
    })
    y = pd.Series(np.roll(sig, -1))
    return X, y


def make_noise(n, seed):
    rs = np.random.RandomState(seed)
    X = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="D"),
        "v": rs.randn(n) * 10,
        "u": rs.rand(n),
    })
    return X, pd.Series(rs.randn(n))


def hit(days, p):
    return any(abs(d - p) <= TOL * p for d in days)


def run_case(X, y):
    bf = bb.BigFeat(task_type="regression", enable_time_series="auto",
                    datetime_col="date", verbose=False)
    bf.fit(X, y, gen_size=2, iterations=1, random_state=0)
    wins = [w.days for w in (bf.window_sizes or [])] if bf.enable_time_series else []
    lags = [getattr(l, "days", l) for l in (bf.lag_periods or [])] \
        if bf.enable_time_series else []
    return dict(ts=bool(bf.enable_time_series), windows=wins, lags=lags,
                strategy=str(getattr(bf, "detection_strategy", None)))


rows = []
for name, planted, amps in SIGNALS:
    for snr_name, noise in NOISE.items():
        for cycles in CYCLES:
            if len(planted) > 1 and cycles < 5:
                continue          # a pair needs room for the slow component
            for seed in SEEDS:
                X, y = make(planted, amps, noise, cycles, seed)
                try:
                    r = run_case(X, y)
                except Exception as e:
                    r = dict(ts=None, windows=[], lags=[],
                             strategy=f"ERROR:{type(e).__name__}")
                r.update(kind="planted", signal=name, planted=planted,
                         snr=snr_name, cycles=cycles, seed=seed, n=len(X))
                r["win_hits"] = [hit(r["windows"], p) for p in planted]
                r["lag_hits"] = [hit(r["lags"], p) for p in planted]
                rows.append(r)
                print(f"[{LABEL}] {name:11s} {snr_name:8s} c={cycles:2d} s={seed} "
                      f"ts={str(r['ts']):5s} win={str(r['windows']):28s} "
                      f"lag={r['lags']}", flush=True)

for n in (90, 365, 910):
    for seed in SEEDS:
        X, y = make_noise(n, seed)
        try:
            r = run_case(X, y)
        except Exception as e:
            r = dict(ts=None, windows=[], lags=[],
                     strategy=f"ERROR:{type(e).__name__}")
        r.update(kind="noise", signal=f"noise_{n}", planted=[], snr="-",
                 cycles=0, seed=seed, n=n, win_hits=[], lag_hits=[])
        rows.append(r)
        print(f"[{LABEL}] noise n={n:4d} s={seed} ts={r['ts']}", flush=True)

json.dump(rows, open(OUT, "w"), indent=1)

planted_rows = [r for r in rows if r["kind"] == "planted" and r["ts"] is not None]
noise_rows = [r for r in rows if r["kind"] == "noise" and r["ts"] is not None]
full = [r for r in planted_rows if all(r["win_hits"])]
lagf = [r for r in planted_rows if all(r["lag_hits"])]
print(f"\n[{LABEL}] SUMMARY over {len(planted_rows)} planted + {len(noise_rows)} noise cases")
print(f"[{LABEL}]   windows recover ALL planted periods : "
      f"{len(full)}/{len(planted_rows)} ({100*len(full)/len(planted_rows):.0f}%)")
print(f"[{LABEL}]   lags contain ALL planted periods    : "
      f"{len(lagf)}/{len(planted_rows)} ({100*len(lagf)/len(planted_rows):.0f}%)")
print(f"[{LABEL}]   TS enabled on planted signals       : "
      f"{sum(1 for r in planted_rows if r['ts'])}/{len(planted_rows)}")
print(f"[{LABEL}]   FALSE POSITIVES on pure noise       : "
      f"{sum(1 for r in noise_rows if r['ts'])}/{len(noise_rows)}")
