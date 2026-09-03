"""Experiment B pilot: UCI Bike Sharing (hourly), four arms, five seeds.

PRE-REGISTERED ANALYSIS -- written before any result was produced:

* Task: predict cnt(t) from exogenous weather at t plus target history
  strictly before t (cnt shifted >= 1). Every arm sees the identical
  information set; the arms differ ONLY in feature engineering.
* Split: temporal, first 80% train / last 20% test. No shuffling.
* Model: RandomForestRegressor(n_estimators=200, random_state=0) for every
  arm, so differences are attributable to features alone.
* Primary metric: MAE on the test split. RMSE secondary.
* Arms:
    baseline     raw covariates + calendar columns + cnt_prev
    handcrafted  baseline + lag 24h/168h and rolling 24h/168h of cnt_prev
                 (what a competent engineer writes for hourly demand data)
    bigfeat_no   BigFeat enable_time_series='no' on the baseline columns
    bigfeat_auto BigFeat enable_time_series='auto' with the timestamp
* Seeds: BigFeat random_state in {0..4}. baseline/handcrafted are
  deterministic and run once.
* Comparison: per-seed MAE deltas vs baseline AND vs handcrafted; report
  mean +- spread across seeds. The claim under test is bigfeat_auto vs
  HANDCRAFTED -- beating "no features" is weak, beating a competent hand
  is the actual bar.
* Decision rule, fixed in advance: bigfeat_auto "helps" only if its mean
  MAE beats handcrafted AND the seed spread does not straddle it.

Usage: python pilot_bike.py <out.json>
"""
import json
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, '/home/victoryst/projects/BigFeat-clean')
import bigfeat.bigfeat_base as bb

OUT = sys.argv[1] if len(sys.argv) > 1 else 'pilot_bike_results.json'
SEEDS = (0, 1, 2, 3, 4)

# ---------------------------------------------------------------------------
df = pd.read_csv('testing/Benchmarking/datasets/uci/hour.csv')
df['timestamp'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
df = df.sort_values('timestamp').reset_index(drop=True)

# information set: exogenous at t + target history <= t-1
df['cnt_prev'] = df['cnt'].shift(1)
df = df.dropna().reset_index(drop=True)

# --no-calendar: EXPLORATORY variant, post-hoc relative to the
# pre-registered analysis above. It removes the cyclic calendar columns
# (hr, weekday, mnth, season) from EVERY arm to test whether detected
# periods can SUBSTITUTE for hand-encoded calendar knowledge -- the
# pre-registered run showed detection finding the true 1d/7d cycles that
# the baseline's hr/weekday columns already encoded for free.
NO_CAL = '--no-calendar' in sys.argv
CAL = (['holiday', 'workingday', 'weathersit'] if NO_CAL else
       ['hr', 'weekday', 'mnth', 'holiday', 'workingday', 'season', 'weathersit'])
EXO = ['temp', 'atemp', 'hum', 'windspeed']
BASE = EXO + CAL + ['cnt_prev']
if NO_CAL:
    print("EXPLORATORY VARIANT: cyclic calendar columns removed from all arms",
          flush=True)

cut = int(len(df) * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]
y_tr, y_te = tr['cnt'], te['cnt']
print(f"rows={len(df)}  train={len(tr)}  test={len(te)}  "
      f"span {df.timestamp.min().date()} .. {df.timestamp.max().date()}", flush=True)


def score(Xtr, Xte, label):
    m = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    m.fit(np.nan_to_num(np.asarray(Xtr, dtype=float)), y_tr)
    p = m.predict(np.nan_to_num(np.asarray(Xte, dtype=float)))
    mae = float(mean_absolute_error(y_te, p))
    rmse = float(np.sqrt(mean_squared_error(y_te, p)))
    print(f"  {label:28s} MAE {mae:8.3f}   RMSE {rmse:8.3f}", flush=True)
    return dict(label=label, mae=mae, rmse=rmse, n_features=int(Xtr.shape[1]))


results = []

# --- arm 1: baseline ------------------------------------------------------
results.append(score(tr[BASE], te[BASE], 'baseline'))

# --- arm 2: handcrafted ---------------------------------------------------
hc = df.copy()
for L in (24, 168):
    hc[f'lag_{L}'] = hc['cnt_prev'].shift(L - 1)      # cnt at t-L
    hc[f'roll_{L}'] = hc['cnt_prev'].rolling(L, min_periods=1).mean()
HC = BASE + [f'lag_{L}' for L in (24, 168)] + [f'roll_{L}' for L in (24, 168)]
hc = hc.fillna(0)
results.append(score(hc.iloc[:cut][HC], hc.iloc[cut:][HC], 'handcrafted'))

# --- arms 3 & 4: BigFeat --------------------------------------------------
def bigfeat_arm(mode, seed):
    cols = (['timestamp'] if mode == 'auto' else []) + BASE
    kw = dict(task_type='regression', verbose=False)
    if mode == 'auto':
        kw.update(enable_time_series='auto', datetime_col='timestamp')
    else:
        kw.update(enable_time_series='no')
    bf = bb.BigFeat(**kw)
    t0 = time.time()
    Xtr = bf.fit(tr[cols], y_tr, gen_size=5, iterations=3, random_state=seed)
    Xte = bf.transform(te[cols])
    r = score(Xtr, Xte, f'bigfeat_{mode} seed={seed}')
    r.update(seed=seed, fit_seconds=round(time.time() - t0, 1),
             ts_enabled=bool(getattr(bf, 'enable_time_series', False)),
             windows=[w.days for w in (bf.window_sizes or [])]
                     if getattr(bf, 'enable_time_series', False) else [],
             lags=[getattr(l, 'days', l) for l in (bf.lag_periods or [])]
                  if getattr(bf, 'enable_time_series', False) else [])
    return r

for seed in SEEDS:
    results.append(bigfeat_arm('no', seed))
for seed in SEEDS:
    results.append(bigfeat_arm('auto', seed))

json.dump(results, open(OUT, 'w'), indent=1)

# --- pre-registered summary ------------------------------------------------
base = next(r for r in results if r['label'] == 'baseline')['mae']
hand = next(r for r in results if r['label'] == 'handcrafted')['mae']
for mode in ('no', 'auto'):
    m = [r['mae'] for r in results if r['label'].startswith(f'bigfeat_{mode}')]
    print(f"\nbigfeat_{mode}: MAE mean {np.mean(m):.3f}  "
          f"[{min(m):.3f} .. {max(m):.3f}] over {len(m)} seeds")
    print(f"  vs baseline    {base:8.3f}: {'BETTER' if np.mean(m) < base else 'worse'}")
    print(f"  vs handcrafted {hand:8.3f}: {'BETTER' if np.mean(m) < hand else 'worse'}")
print(f"\ndecision rule: bigfeat_auto helps iff mean < handcrafted "
      f"and max(seed MAE) < handcrafted", flush=True)
