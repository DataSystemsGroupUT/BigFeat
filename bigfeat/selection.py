"""
selection.py
------------
Responsible for feature filtering and selection (Correlation & fAnova).
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, f_classif


def check_correlations(feats):
    """ Check correlations among the selected features """
    cor_thresh = 0.8
    corr_matrix = pd.DataFrame(feats).corr().abs()
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(mask)
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > cor_thresh)]
    # remove the feature with lower importance if corr > cor_thresh
    # to_drop = []
    # for c in tri_df.columns:
    #     if any(corr_matrix[c] > cor_thresh):
    #         for c_, cor_val in enumerate(corr_matrix[c].values):
    #             if cor_val > cor_thresh and c != c_:
    #                 if self.ig_vector_gen[c_] < self.ig_vector_gen[c] and c_ not in to_drop:
    #                     to_drop.append(c_)

    feats = pd.DataFrame(feats).drop(to_drop, axis=1)
    return feats.values, to_drop


def fit_fanova(X, y, task_type, n_feats):
    """
    Use the appropriate feature selection method (fAnova) based on task type.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    task_type : str
        Either 'classification' or 'regression'.
    n_feats : int
        Number of top features to select.
        
    Returns:
    --------
    transformed_feats : array-like
        The reduced feature matrix.
    selector : SelectKBest object
        The fitted selector to be used later in transform.
    """
    if task_type == 'classification':
        selector = SelectKBest(f_classif, k=n_feats)
    else:  # regression
        selector = SelectKBest(f_regression, k=n_feats)
        
    transformed_feats = selector.fit_transform(X, y)
    
    return transformed_feats, selector