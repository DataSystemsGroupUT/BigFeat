# BigFeat: Scalable and Interpretable Automated Feature Engineering Framework

## What is BigFeat?
BigFeat is a scalable and interpretable automated feature engineering framework designed to enhance the quality of input features to maximize predictive performance based on a user-defined metric. It supports both **classification** and **regression** tasks, employing a dynamic feature generation and selection mechanism to construct expressive, interpretable features that improve prediction performance.

## Input/Output
BigFeat takes original input features and returns a collection of base and engineered features expected to enhance predictive performance for either classification or regression tasks.

## Setup and Installation

### Prerequisites
Ensure you have Python 3.8+ installed. BigFeat requires specific versions of Python packages as listed in the `requirements.txt` file.

### Installation Steps

1. **Clone the Repository** (if applicable):
   ```bash
   git clone https://github.com/DataSystemsGroupUT/BigFeat.git
   cd BigFeat
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt              # library runtime deps
   pip install -r requirements-benchmark.txt    # optional: benchmarking suite
   ```

   `requirements.txt` pins the six packages the library itself imports:
   `lightgbm`, `numpy`, `pandas`, `psutil`, `scikit-learn`, `scipy`. The
   benchmark file adds the heavier evaluation-only dependencies (`gluonts`,
   `openfe`, `stumpy`, `tsfresh`, `yfinance`, `matplotlib`, `seaborn`,
   `statsmodels`, `pyarrow`).

4. **Install BigFeat**:
   ```bash
   pip install .          # or: pip install -e .  for development
   ```

5. **Run the tests** (optional but recommended):
   ```bash
   pip install pytest
   pytest                 # 91 tests, ~85 s
   pytest -m "not slow"   # fast subset, ~58 s
   ```
   See [`tests/README.md`](tests/README.md) for what the suite covers.

## Usage

BigFeat can be used to generate and select features for both classification and regression tasks. Below is an example demonstrating how to run BigFeat on test datasets.

### Example Code

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, r2_score
import bigfeat.bigfeat_base as bigfeat
import sklearn.preprocessing as preprocessing

def run_tst(df_path, target_ft, random_state, task_type='classification'):
    df = pd.read_csv(df_path)
    # Encode categorical columns
    object_columns = df.select_dtypes(include='object')
    if len(object_columns.columns):
        df[object_columns.columns] = object_columns.apply(preprocessing.LabelEncoder().fit_transform)
    X = df.drop(columns=target_ft)
    y = df[target_ft]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Initialize BigFeat for classification or regression
bf = bigfeat.BigFeat(task_type='classification')  # Use 'regression' for regression tasks

# Example datasets (replace with your dataset paths and target columns)
datasets = [
    ("data/shuttle.csv", "class", "classification"),
    ("data/blood-transfusion-service-center.csv", "Class", "classification"),
    ("data/credit-g.csv", "class", "classification"),
    ("data/kc1.csv", "defects", "classification"),
    ("data/nomao.csv", "Class", "classification"),
    ("data/eeg_eye_state.csv", "Class", "classification"),
    ("data/gina.csv", "class", "classification"),
    ("data/sonar.csv", "Class", "classification"),
    ("data/arcene.csv", "Class", "classification"),
    ("data/madelon.csv", "Class", "classification"),
    # Add regression datasets as needed
]

for dataset, target, task_type in datasets:
    print(f"\nProcessing dataset: {dataset}")
    X_train, X_test, y_train, y_test = run_tst(dataset, target, random_state=0, task_type=task_type)
    
    # Configure BigFeat for the task
    bf = bigfeat.BigFeat(task_type=task_type)
    
    # Fit BigFeat
    res = bf.fit(
        X_train, y_train,
        gen_size=5,
        random_state=0,
        iterations=5,
        estimator='avg',
        feat_imps=True,
        split_feats=None,
        check_corr=False,
        selection='fAnova',
        combine_res=True
    )
    
    # Evaluate performance
    if task_type == 'classification':
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Original F1 Score: {f1_score(y_test, y_pred):.4f}")
        
        clf = LogisticRegression(random_state=0).fit(bf.transform(X_train), y_train)
        y_pred_bf = clf.predict(bf.transform(X_test))
        print(f"BigFeat F1 Score: {f1_score(y_test, y_pred_bf):.4f}")
        
    else:  # regression
        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print(f"Original R² Score: {r2_score(y_test, y_pred):.4f}")
        
        reg = LinearRegression().fit(bf.transform(X_train), y_train)
        y_pred_bf = reg.predict(bf.transform(X_test))
        print(f"BigFeat R² Score: {r2_score(y_test, y_pred_bf):.4f}")
```

### Key Parameters for `BigFeat.fit`
- `gen_size`: Number of features to generate per iteration.
- `random_state`: Seed for reproducibility.
- `iterations`: Number of feature generation iterations.
- `estimator`: Method for feature importance ('avg' uses RandomForest and LightGBM).
- `feat_imps`: Whether to use feature importance for guiding generation.
- `split_feats`: Strategy for splitting features ('comb' or 'splits').
- `check_corr`: Whether to check and remove highly correlated features.
- `selection`: Feature selection method ('stability' or 'fAnova').
- `combine_res`: Whether to combine results across iterations.

`random_state` fully determines the output: two fits with the same seed produce
identical features without the caller needing to seed NumPy globally.

## Time Series Features

BigFeat can detect temporal structure in a dataset and generate time-aware
features (rolling statistics, lags, differences, EWM, seasonal decomposition)
alongside the standard arithmetic ones.

```python
bf = bigfeat.BigFeat(
    task_type='regression',
    enable_time_series='auto',   # 'auto' | 'yes' | 'no'
    datetime_col='timestamp',
    groupby_cols=['item_id'],    # one entity per series
)
X_train_feats = bf.fit(X_train, y_train, gen_size=5, iterations=3, random_state=0)
X_test_feats  = bf.transform(X_test)
```

### How windows are chosen

In `'auto'` mode BigFeat runs three periodicity detectors — DFT, ACF and
Lomb-Scargle — and enables time-series features only if at least two agree the
data is periodic (or one is highly confident). Window sizes are pooled from the
agreeing detectors.

Windows are genuine time spans, not row counts: a 90-day window covers 90 days
of timestamps regardless of whether the data is sampled hourly, daily or
monthly, and regardless of calendar units having unequal lengths. The sampling
rate is measured from the data rather than assumed.

Rolling operations never cross an entity boundary when `groupby_cols` is set,
and all operators are causal — a row's features depend only on rows dated at or
before it.

Strongly non-stationary data (unit-root/trending series) is detected with an
Augmented Dickey–Fuller test and routed to a restricted operator set, since
rolling means on a trending series describe the trend rather than the signal.
This needs `statsmodels` (`pip install bigfeat[stationarity]`); without it a
lag-1 autocorrelation heuristic is used instead.

### Key time-series parameters

| Parameter | Default | Meaning |
|---|---|---|
| `enable_time_series` | `'auto'` | `'auto'` detects periodicity; `'yes'` forces on (requires `datetime_col`); `'no'` disables |
| `datetime_col` | `None` | Timestamp column; auto-detected in `'auto'` mode |
| `groupby_cols` | `[]` | Entity columns — rolling windows never span two entities |
| `window_detector` | `'ensemble'` | `'dft'`, `'acf'`, `'lomb_scargle'` or `'ensemble'` |
| `confidence_threshold` | `0.5` | Minimum periodicity confidence to enable TS features |
| `min_window_days` / `max_window_days` | `1` / `365` | Bounds on detected windows |
| `n_windows` | `6` | Number of window sizes to keep |
| `enable_downsampling` | `False` | Memory-aware block sampling for feature *discovery*; `transform` still returns every row |

### Regression targets

For strictly positive, highly skewed regression targets (skew > 2), `fit`
log-transforms the target internally before scoring feature importances. This
does not modify your `y` array, but it does affect which features are selected.
Check `bf.target_log_transformed` and use `bf.inverse_transform_target(preds)`
if you train a downstream model on the same transformed target.

## Development

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the library works:
  the generation loop, how features are represented and replayed, the
  time-series subsystem, and where the rough edges are. **Start here if you
  are modifying the code.**
- [`docs/BENCHMARK_DESIGN.md`](docs/BENCHMARK_DESIGN.md) — what to evaluate
  the time-series subsystem on, and why the current Monash suite cannot
  exercise it.
- [`docs/PILOT_BIKE.md`](docs/PILOT_BIKE.md) — first multivariate accuracy
  pilot: feature engineering is redundant when calendar columns are given,
  and substitutes for them (−59% MAE) when they are not.
- [`docs/PIPELINE_IMPROVEMENTS.md`](docs/PIPELINE_IMPROVEMENTS.md) — the
  detection-quality improvement series: what changed at each pipeline
  stage and the ground-truth before/after (69%→100% period recovery).
- [`docs/CORRECTNESS_FIXES.md`](docs/CORRECTNESS_FIXES.md) — correctness review
  of the time-series subsystem: defects found, how each was verified, and the
  benchmark impact. **Read this before relying on results collected before the
  fixes**, which measured different behaviour than their configuration
  describes.
- [`tests/README.md`](tests/README.md) — test suite layout, design rationale,
  and known coverage gaps.
- [`testing/Benchmarking/`](testing/Benchmarking/) — evaluation harness over
  the Monash time-series archive.

## Cite Us
If you use BigFeat in your research, please cite the following paper:

```bib
@inproceedings{eldeeb2022bigfeat,
  title={BigFeat: Scalable and Interpretable Automated Feature Engineering Framework},
  author={Eldeeb, Hassan and Amashukeli, Shota and ElShawi, Radwa},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={515--524},
  year={2022},
  organization={IEEE}
}
```
