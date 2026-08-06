# BigFeat Comprehensive Time Series Benchmark

Comprehensive evaluation of BigFeat's automated feature engineering on 29 time series datasets from the AutoGluon-TimeSeries paper.

## What It Tests

### Configurations Compared

1. **Baseline**: No BigFeat, only basic features
2. **BigFeat** with all combinations of:
   - **Time Series Mode** (`enable_time_series`):
     - `'auto'`: Auto-detect periodicity
     - `'yes'`: Force enable time series features
     - `'no'`: Standard features only
   - **Window Detector** (`window_detector`):
     - `'dft'`: Discrete Fourier Transform
     - `'acf'`: Autocorrelation Function
     - `'lomb_scargle'`: Lomb-Scargle Periodogram

Total: **8 configurations** (1 baseline + 7 BigFeat variants)

*Note: For `enable_time_series='no'`, detector is irrelevant, so only tested once*

### Metrics Measured

#### Performance Metrics
- **MASE** (Mean Absolute Scaled Error) - primary metric
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (R-squared score)

#### Resource Metrics
- **Wall Time**: Total elapsed time
- **CPU Time**: Actual CPU seconds used
- **CPU Percent**: CPU utilization percentage
- **Memory Start/End**: Memory usage in MB
- **Memory Delta**: Memory increase during execution
- **Memory Peak**: Maximum memory used

#### Feature Engineering Metrics
- Number of features generated
- Number of original features
- Time series detection strategy
- Window sizes detected
- Detection confidence scores

## Installation

```bash
# Install dependencies
pip install gluonts pandas numpy scikit-learn psutil

# Install BigFeat (adjust path as needed)
pip install -e /path/to/bigfeat
```

## Usage

### Quick Test (3 small datasets)
```bash
python bigfeat_comprehensive_benchmark.py --quick-test
```

### Test Specific Datasets
```bash
python bigfeat_comprehensive_benchmark.py --datasets m1_yearly tourism_yearly
```

### Test All Datasets
```bash
python bigfeat_comprehensive_benchmark.py
```

### Custom Configuration
```bash
python bigfeat_comprehensive_benchmark.py \
    --datasets m4_monthly m4_weekly \
    --estimator gbm \
    --output-dir ./my_results \
    --verbose
```

## Command Line Options

- `--datasets`: List of specific datasets to test (default: all 29)
- `--output-dir`: Directory for results (default: `./benchmark_results`)
- `--estimator`: Model to use (`rf`, `ridge`, `gbm`; default: `rf`)
- `--time-limit`: Max seconds per dataset (default: 3600)
- `--quick-test`: Run on 3 small datasets for testing
- `--verbose`: Print detailed progress (default: True)

## Datasets

All 29 datasets from AutoGluon-TimeSeries paper:

**M-Competitions**:
- m1_monthly, m1_quarterly, m1_yearly
- m3_monthly, m3_other, m3_quarterly, m3_yearly
- m4_daily, m4_hourly, m4_monthly, m4_quarterly, m4_weekly, m4_yearly

**Other Forecasting Benchmarks**:
- car_parts_without_missing
- cif_2016
- covid_deaths
- electricity_hourly, electricity_weekly
- fred_md
- hospital
- kdd_cup_2018_without_missing
- nn5_daily_without_missing, nn5_weekly
- pedestrian_counts
- tourism_monthly, tourism_quarterly, tourism_yearly
- vehicle_trips_without_missing
- kaggle_web_traffic_weekly

## Output Files

### Individual Dataset Results
`{dataset_name}_results.json` - Complete results for each dataset including:
- Metadata (frequency, prediction length, etc.)
- Baseline performance
- All BigFeat variant performances
- Resource usage for each configuration
- Detection statistics

### Summary CSV
`benchmark_summary.csv` - Aggregated results across all datasets with columns:
- Dataset metadata
- `baseline_*`: Baseline metrics
- `{mode}_{detector}_*`: BigFeat configuration metrics
  - `_mase`, `_mae`, `_r2`: Performance
  - `_time`, `_cpu_pct`, `_mem_mb`: Resources
  - `_n_features`: Feature count
  - `_ts_enabled`, `_ts_strategy`, `_ts_confidence`: Detection info

## Example Output

```
================================================================================
BigFeat Comprehensive Time Series Benchmark
================================================================================
Datasets to test: 3
Estimator: rf
BigFeat configurations: 7
  - Baseline (no BigFeat)
  - BigFeat: enable_time_series='auto', window_detector='dft'
  - BigFeat: enable_time_series='auto', window_detector='acf'
  - BigFeat: enable_time_series='auto', window_detector='lomb_scargle'
  - BigFeat: enable_time_series='yes', window_detector='dft'
  - BigFeat: enable_time_series='yes', window_detector='acf'
  - BigFeat: enable_time_series='yes', window_detector='lomb_scargle'
  - BigFeat: enable_time_series='no', window_detector='dft'
================================================================================

[1/3] Processing: m1_yearly
  ✓ Loaded 181 timesteps across 181 series

1. Running Baseline (no feature engineering)...
  ✓ Baseline
    MASE: 2.3456
    Features: 15
    Time: 0.52s
    CPU: 95.2%
    Memory: 12.3 MB

2. Running BigFeat-auto-dft...
  ✓ BigFeat-auto-dft
    MASE: 1.9234
    Features: 47
    Time: 4.23s
    CPU: 87.3%
    Memory: 45.6 MB
    TS Strategy: dft_detected
    TS Confidence: 1.85

...

================================================================================
Performance Summary
================================================================================

Average MASE by method:
  Method                           MASE
  ------------------------------ ----------
  Baseline                          2.3456
  BF-auto_dft                       1.9234
  BF-auto_acf                       2.0123
  ...

Wins against Baseline (lower MASE):
  Method                           Wins       Rate
  ------------------------------ ---------- ----------
  BF-auto_dft                      2/3        66.7%
  BF-auto_acf                      1/3        33.3%
  ...

Average Runtime (seconds):
  Method                         Time (s)    CPU %  Mem (MB)
  ------------------------------ ---------- ---------- ----------
  Baseline                             0.52       95.2       12.3
  BF-auto_dft                          4.23       87.3       45.6
  ...
```

## Analyzing Results

### Load Summary
```python
import pandas as pd

df = pd.read_csv('benchmark_results/benchmark_summary.csv')

# Compare MASE scores
print(df[['dataset', 'baseline_mase', 'auto_dft_mase', 'auto_acf_mase']].head())

# Best configuration per dataset
configs = ['auto_dft', 'auto_acf', 'auto_lomb_scargle', 
           'yes_dft', 'yes_acf', 'yes_lomb_scargle', 'no_dft']
mase_cols = [f'{c}_mase' for c in configs]
df['best_config'] = df[mase_cols].idxmin(axis=1)
print(df['best_config'].value_counts())

# Resource usage comparison
print(df[['baseline_time', 'auto_dft_time', 'auto_acf_time']].mean())
```

### Visualization Example
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Win rate comparison
win_rates = []
for config in configs:
    wins = (df[f'{config}_mase'] < df['baseline_mase']).sum()
    total = df[f'{config}_mase'].notna().sum()
    win_rates.append({'config': config, 'win_rate': wins/total*100})

win_df = pd.DataFrame(win_rates)
plt.figure(figsize=(10, 6))
sns.barplot(data=win_df, x='config', y='win_rate')
plt.xticks(rotation=45)
plt.ylabel('Win Rate vs Baseline (%)')
plt.title('BigFeat Configuration Performance')
plt.tight_layout()
plt.savefig('win_rates.png')
```

## Notes

- **Time Series Detection**: In `'auto'` mode, detectors analyze the data and decide whether to enable time series features based on detected periodicity
- **Resource Usage**: Measured using `psutil` for accurate CPU and memory tracking
- **Reproducibility**: Fixed random seed (42) for consistent results
- **Timeout**: Each dataset has a time limit (default: 1 hour)
- **Graceful Failure**: Individual configuration failures don't stop the benchmark

## Troubleshooting

### Import Error
```python
# If BigFeat not in path, add it:
import sys
sys.path.append('/path/to/bigfeat')
from bigfeat.bigfeat_base import BigFeat
```

### Memory Issues
```bash
# Reduce dataset size or run fewer configurations
python bigfeat_comprehensive_benchmark.py --quick-test
```

### Dataset Download Issues
```bash
# GluonTS may need to download datasets on first run
# Ensure internet connection and sufficient disk space
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{bigfeat_benchmark,
  title={BigFeat Time Series Benchmark},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/bigfeat}
}
```

## License

MIT License - See LICENSE file for details
---

## Practical notes

Added after running the suite during the correctness review documented in
[`../../docs/CORRECTNESS_FIXES.md`](../../docs/CORRECTNESS_FIXES.md).

### Runtime

A full run is **roughly 172 hours**, derived from the wall times in the
committed `benchmark_results/*.json`:

| Method | Datasets | Mean wall time | Total |
|---|---|---|---|
| `bigfeat_yes_dft` | 24 | 6005 s | 40.0 h |
| `bigfeat_yes_lomb_scargle` | 24 | 5998 s | 40.0 h |
| `bigfeat_auto_ensemble` | 24 | 5635 s | 37.6 h |
| `bigfeat_yes_acf` | 24 | 5512 s | 36.7 h |
| `bigfeat_no_standard` | 24 | 1428 s | 9.5 h |
| `tsfresh` | 18 | 980 s | 4.9 h |
| `openfe` | 22 | 517 s | 3.2 h |
| `baseline` | 24 | 111 s | 0.7 h |

Scope a run before starting one:

```bash
python benchmark.py --datasets m1_monthly tourism_quarterly \
                    --methods baseline auto_ensemble \
                    --output-dir ./benchmark_results
python benchmark.py --quick-test    # 3 small datasets
```

### Sampling regularity of the datasets

All 25 datasets declare a regular frequency, and the GluonTS format stores only
a `start` timestamp plus a dense `target` array — timestamps are reconstructed
as `start + i * freq`, so **every series is uniformly sampled by
construction**. Irregular sampling is not representable in this format.

However, 13 of the 25 have **non-uniform spacing in days**, because calendar
units are not fixed durations:

| Frequency | Datasets | Distinct gaps |
|---|---|---|
| M (monthly) | 8 | 4 (28–31 days) |
| Q (quarterly) | 3 | 3 (90–92 days) |
| Y (yearly) | 3 | 2 (365–366 days) |
| D / W / h | 12 | 1 (uniform) |

This is exactly where a row-count approximation of a time window cannot be
correct, and it was the source of the most consequential defect found in the
review (`CORRECTNESS_FIXES.md` §2.9).

Two consequences worth knowing:

- **The Lomb-Scargle detector has no dataset here that exercises its
  advantage.** It exists specifically for irregularly sampled data, which this
  archive cannot represent. Claims about its relative merit need a different
  data source.
- **16 of 25 datasets are ragged** (unequal series lengths), which matters for
  any per-series aggregation.

### Repository size

The `datasets/` and `datasets/exported_datasets/` trees hold roughly **2.3 GB**,
with several individual CSVs over 200 MB — above GitHub's 100 MB per-file
limit. They are regenerable:

```bash
python datasets/download_datasets.py
```

Prefer regenerating over committing them.
