"""
evaluation.py
-------------
Responsible for model evaluation and selecting the best estimator.
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, r2_score, make_scorer


def select_estimator(x, y, task_type='classification', n_jobs=-1, estimators_names=None, random_state=42):
    """
    Select the best estimator based on cross-validation
    
    Parameters:
    -----------
    x : array-like
        Feature matrix
    y : array-like
        Target vector
    task_type : str, default='classification'
        The type of machine learning task. Either 'classification' or 'regression'.
    n_jobs : int, default=-1
        The number of jobs to run in parallel.
    estimators_names : list or None
        List of estimator names to try. If None, uses appropriate defaults.
    random_state : int, default=42
        Seed used by the random number generator.
    
    Returns:
    --------
    model : estimator
        Fitted best estimator
    """
    if estimators_names is None:
        if task_type == 'classification':
            estimators_names = ['dt', 'lr']
        else:
            estimators_names = ['dt_reg', 'lr_reg']
    
    estimators_dic = {
        'dt': DecisionTreeClassifier(
            random_state=random_state, min_samples_leaf=1, max_features=None, ccp_alpha=0.0
        ),
        'lr': LogisticRegression(random_state=random_state),
        'rf': RandomForestClassifier(
            n_jobs=n_jobs, random_state=random_state, min_samples_leaf=1, max_features='sqrt'
        ),
        'lgb': LGBMClassifier(random_state=random_state, verbosity=-1),
        
        'dt_reg': DecisionTreeRegressor(
            random_state=random_state, min_samples_leaf=1, max_features=None, ccp_alpha=0.0
        ),
        'lr_reg': LinearRegression(),
        'rf_reg': RandomForestRegressor(
            n_jobs=n_jobs, random_state=random_state, min_samples_leaf=1, max_features='sqrt'
        ),
        'lgb_reg': LGBMRegressor(random_state=random_state, verbosity=-1)
    }
    
    models_score = {}

    for estimator in estimators_names:
        model = estimators_dic[estimator]
        
        if task_type == 'classification':
            scorer = make_scorer(f1_score)
        else:
            scorer = make_scorer(r2_score)
            
        models_score[estimator] = cross_val_score(model, x, y, cv=3, scoring=scorer).mean()
        
    best_estimator = max(models_score, key=models_score.get)
    best_model = estimators_dic[best_estimator]
    best_model.fit(x, y)
    
    return best_model