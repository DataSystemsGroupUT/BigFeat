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
   Use the provided `requirements.txt` to install all required packages with their exact versions:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:
   ```
   bigfeat==0.1
   joblib==1.4.2
   lightgbm==4.6.0
   numpy==2.2.5
   pandas==2.2.3
   python-dateutil==2.9.0.post0
   pytz==2025.2
   scikit-learn==1.6.1
   scipy==1.15.2
   six==1.17.0
   threadpoolctl==3.6.0
   tzdata==2025.2
   ```

4. **Install BigFeat**:
   If not already installed via `requirements.txt`, install BigFeat locally:
   ```bash
   pip install .
   ```

   Alternatively, install directly from the source:
   ```bash
   pip install ./BigFeat
   ```

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
