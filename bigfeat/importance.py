"""
importance.py
-------------
Responsible for training initial models to extract feature importances.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
import lightgbm as lgb


def get_feature_importances(x, y, estimator, random_state, task_type, sample_count=1, sample_size=3, n_jobs=1):
    """Return feature importances by specified method"""
    rng = np.random.default_rng(random_state)
    n_rows = x.shape[0]
    importance_sum = np.zeros(x.shape[1])
    total_estimators = []
    
    for _ in range(sample_count):
        sampled_ind = rng.choice(n_rows, size=n_rows // sample_size, replace=False)
        sampled_x = x[sampled_ind]
        sampled_y = np.take(y, sampled_ind)
        
        # Different behavior based on task type
        if estimator == "rf":
            if task_type == 'classification':
                estm = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, min_samples_leaf=1, max_features='sqrt')
            else:  # regression
                estm = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, min_samples_leaf=1, max_features='sqrt')
            
            estm.fit(sampled_x, sampled_y)
            total_importances = estm.feature_importances_
            estimators = estm.estimators_
            total_estimators += estimators
            
        elif estimator == "avg":
            # For classification
            if task_type == 'classification':
                clf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, min_samples_leaf=1, max_features='sqrt')
                clf.fit(sampled_x, sampled_y)
                rf_importances = clf.feature_importances_
                estimators = clf.estimators_
                total_estimators += estimators
                
                # LightGBM for classification
                train_data = lgb.Dataset(sampled_x, label=sampled_y)
                param = {'num_leaves': 31, 'objective': 'binary', 'verbose': -1}
                param['metric'] = 'auc'
                
            # For regression
            else:
                clf = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, min_samples_leaf=1, max_features='sqrt')
                clf.fit(sampled_x, sampled_y)
                rf_importances = clf.feature_importances_
                estimators = clf.estimators_
                total_estimators += estimators
                
                # LightGBM for regression
                train_data = lgb.Dataset(sampled_x, label=sampled_y)
                param = {'num_leaves': 31, 'objective': 'regression', 'verbose': -1}
                param['metric'] = 'rmse'
            
            # Common LightGBM code for both tasks
            num_round = 2
            bst = lgb.train(param, train_data, num_round)
            lgb_imps = bst.feature_importance(importance_type='gain')
            lgb_imps /= lgb_imps.sum()
            total_importances = (rf_importances + lgb_imps) / 2
            
        importance_sum += total_importances
        
    return importance_sum, total_estimators


def get_weighted_feature_importances(x, y, random_state, task_type, n_jobs=-1):
    """Return feature importances weighted by model performance"""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state)
    
    # Choose appropriate model based on task type
    if task_type == 'classification':
        estm = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, min_samples_leaf=1, max_features='sqrt')
    else:  # regression
        estm = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, min_samples_leaf=1, max_features='sqrt')
        
    estm.fit(x_train, y_train)
    model = estm
    imps = np.zeros((len(model.estimators_), x.shape[1]))
    scores = np.zeros(len(model.estimators_))
    
    for i, each in enumerate(model.estimators_):
        # Different scoring metrics based on task type
        if task_type == 'classification':
            y_probas_train = each.predict_proba(x_test)[:, 1]
            score = roc_auc_score(y_test, y_probas_train)
        else:  # regression
            y_pred_train = each.predict(x_test)
            score = r2_score(y_test, y_pred_train)
            
        imps[i] = each.feature_importances_
        scores[i] = score
        
    weights = scores / scores.sum()
    return np.average(imps, axis=0, weights=weights)