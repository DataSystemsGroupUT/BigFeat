import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import bigfeat.local_utils as local_utils
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from functools import partial


class BigFeat:
    """Base BigFeat Class for both classification and regression tasks"""

    def __init__(self, task_type='classification'):
        """
        Initialize the BigFeat object
        
        Parameters:
        -----------
        task_type : str, default='classification'
            The type of machine learning task. Either 'classification' or 'regression'.
        """
        self.n_jobs = -1
        self.operators = [np.multiply, np.add, np.subtract, np.abs, np.square]
        self.binary_operators = [np.multiply, np.add, np.subtract]
        self.unary_operators = [np.abs, np.square, local_utils.original_feat]
        self.task_type = task_type
        
        # Validate task_type input
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be either 'classification' or 'regression'")

    def fit(self, X, y, gen_size=5, random_state=0, iterations=5, estimator='avg', 
            feat_imps=True, split_feats=None, check_corr=True, selection='stability', combine_res=True):
        """ Generated Features using test set """
        self.selection = selection
        self.imp_operators = np.ones(len(self.operators))
        self.operator_weights = self.imp_operators / self.imp_operators.sum()
        self.gen_steps = []
        self.n_feats = X.shape[1]
        self.n_rows = X.shape[0]
        self.ig_vector = np.ones(self.n_feats) / self.n_feats
        self.comb_mat = np.ones((self.n_feats, self.n_feats))
        self.split_vec = np.ones(self.n_feats)
        # Set RNG seed if provided for numpy
        self.rng = np.random.RandomState(seed=random_state)
        gen_feats = np.zeros((self.n_rows, self.n_feats * gen_size))
        iters_comb = np.zeros((self.n_rows, self.n_feats * iterations))
        depths_comb = np.zeros(self.n_feats * iterations)
        ids_comb = np.zeros(self.n_feats * iterations, dtype=object)
        ops_comb = np.zeros(self.n_feats * iterations, dtype=object)
        self.feat_depths = np.zeros(gen_feats.shape[1])
        self.depth_range = np.arange(3) + 1
        self.depth_weights = 1 / (2 ** self.depth_range)
        self.depth_weights /= self.depth_weights.sum()
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        if feat_imps:
            self.ig_vector, estimators = self.get_feature_importances(X, y, estimator, random_state)
            self.ig_vector /= self.ig_vector.sum()
            for tree in estimators:
                paths = self.get_paths(tree, np.arange(X.shape[1]))
                self.get_split_feats(paths, self.split_vec)
            self.split_vec /= self.split_vec.sum()
            # self.split_vec = StandardScaler().fit_transform(self.split_vec.reshape(1, -1), {'var_':5})
            if split_feats == "comb":
                self.ig_vector = np.multiply(self.ig_vector, self.split_vec)
                self.ig_vector /= self.ig_vector.sum()
            elif split_feats == "splits":
                self.ig_vector = self.split_vec
        for iteration in range(iterations):
            self.tracking_ops = []
            self.tracking_ids = []
            gen_feats = np.zeros((self.n_rows, self.n_feats * gen_size))
            self.feat_depths = np.zeros(gen_feats.shape[1])
            for i in range(gen_feats.shape[1]):
                dpth = self.rng.choice(self.depth_range, p=self.depth_weights)
                ops = []
                ids = []
                gen_feats[:, i] = self.feat_with_depth(X, dpth, ops, ids)  # ops and ids are updated
                self.feat_depths[i] = dpth
                self.tracking_ops.append(ops)
                self.tracking_ids.append(ids)
            self.tracking_ids = np.array(self.tracking_ids + [[]], dtype='object')[:-1]
            self.tracking_ops = np.array(self.tracking_ops + [[]], dtype='object')[:-1]
            imps, estimators = self.get_feature_importances(gen_feats, y, estimator, random_state)
            total_feats = np.argsort(imps)
            feat_args = total_feats[-self.n_feats:]
            gen_feats = gen_feats[:, feat_args]
            self.tracking_ids = self.tracking_ids[feat_args]
            self.tracking_ops = self.tracking_ops[feat_args]
            self.feat_depths = self.feat_depths[feat_args]
            depths_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.feat_depths
            ids_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.tracking_ids
            ops_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.tracking_ops
            iters_comb[:, iteration * self.n_feats:(iteration + 1) * self.n_feats] = gen_feats
            for i, op in enumerate(self.operators):
                for feat in self.tracking_ops:
                    for feat_op in feat:
                        if op == feat_op[0]:
                            self.imp_operators[i] += 1
            self.operator_weights = self.imp_operators / self.imp_operators.sum()
        if selection == 'stability' and iterations > 1 and combine_res:
            imps, estimators = self.get_feature_importances(iters_comb, y, estimator, random_state)
            total_feats = np.argsort(imps)
            feat_args = total_feats[-self.n_feats:]
            gen_feats = iters_comb[:, feat_args]
            self.tracking_ids = ids_comb[feat_args]
            self.tracking_ops = ops_comb[feat_args]
            self.feat_depths = depths_comb[feat_args]

        if selection == 'stability' and check_corr:
            gen_feats, to_drop_cor = self.check_correlations(gen_feats)
            self.tracking_ids = np.delete(self.tracking_ids, to_drop_cor)
            self.tracking_ops = np.delete(self.tracking_ops, to_drop_cor)
            self.feat_depths = np.delete(self.feat_depths, to_drop_cor)
        gen_feats = np.hstack((gen_feats, X))

        if selection == 'fAnova':
            # Use the appropriate feature selection method based on task type
            if self.task_type == 'classification':
                self.fAnova_best = SelectKBest(f_classif, k=self.n_feats)
            else:  # regression
                self.fAnova_best = SelectKBest(f_regression, k=self.n_feats)
            gen_feats = self.fAnova_best.fit_transform(gen_feats, y)

        return gen_feats

    def transform(self, X):
        """ Produce features from the fitted BigFeat object """
        X = self.scaler.transform(X)
        self.n_rows = X.shape[0]
        gen_feats = np.zeros((self.n_rows, len(self.tracking_ids)))
        for i in range(gen_feats.shape[1]):
            dpth = self.feat_depths[i]
            op_ls = self.tracking_ops[i].copy()
            id_ls = self.tracking_ids[i].copy()
            gen_feats[:, i] = self.feat_with_depth_gen(X, dpth, op_ls, id_ls)
        gen_feats = np.hstack((gen_feats, X))
        if self.selection == 'fAnova':
            gen_feats = self.fAnova_best.transform(gen_feats)
        return gen_feats

    def select_estimator(self, X, y, estimators_names=None):
        """
        Select the best estimator based on cross-validation
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        estimators_names : list or None
            List of estimator names to try. If None, uses appropriate defaults.
        
        Returns:
        --------
        model : estimator
            Fitted best estimator
        """
        # Use appropriate default estimators based on task type
        if estimators_names is None:
            if self.task_type == 'classification':
                estimators_names = ['dt', 'lr']
            else:  # regression
                estimators_names = ['dt_reg', 'lr_reg']
        
        # Define available estimators based on task type
        estimators_dic = {
            # Classification estimators
            'dt': DecisionTreeClassifier(),
            'lr': LogisticRegression(),
            'rf': RandomForestClassifier(n_jobs=self.n_jobs),
            'lgb': LGBMClassifier(),
            # Regression estimators
            'dt_reg': DecisionTreeRegressor(),
            'lr_reg': LinearRegression(),
            'rf_reg': RandomForestRegressor(n_jobs=self.n_jobs),
            'lgb_reg': LGBMRegressor()
        }
        
        models_score = {}

        for estimator in estimators_names:
            model = estimators_dic[estimator]
            
            # Use appropriate scoring metric based on task type
            if self.task_type == 'classification':
                scorer = make_scorer(f1_score)
            else:  # regression
                scorer = make_scorer(r2_score)
                
            models_score[estimator] = cross_val_score(model, X, y, cv=3, scoring=scorer).mean()
            
        best_estimator = max(models_score, key=models_score.get)
        best_model = estimators_dic[best_estimator]
        best_model.fit(X, y)
        return best_model

    def get_feature_importances(self, X, y, estimator, random_state, sample_count=1, sample_size=3, n_jobs=1):
        """Return feature importances by specified method"""

        importance_sum = np.zeros(X.shape[1])
        total_estimators = []
        for sampled in range(sample_count):
            sampled_ind = np.random.choice(np.arange(self.n_rows), size=self.n_rows // sample_size, replace=False)
            sampled_X = X[sampled_ind]
            sampled_y = np.take(y, sampled_ind)
            
            # Different behavior based on task type
            if estimator == "rf":
                if self.task_type == 'classification':
                    estm = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
                else:  # regression
                    estm = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
                
                estm.fit(sampled_X, sampled_y)
                total_importances = estm.feature_importances_
                estimators = estm.estimators_
                total_estimators += estimators
                
            elif estimator == "avg":
                # For classification
                if self.task_type == 'classification':
                    clf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
                    clf.fit(sampled_X, sampled_y)
                    rf_importances = clf.feature_importances_
                    estimators = clf.estimators_
                    total_estimators += estimators
                    
                    # LightGBM for classification
                    train_data = lgb.Dataset(sampled_X, label=sampled_y)
                    param = {'num_leaves': 31, 'objective': 'binary', 'verbose': -1}
                    param['metric'] = 'auc'
                    
                # For regression
                else:
                    clf = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
                    clf.fit(sampled_X, sampled_y)
                    rf_importances = clf.feature_importances_
                    estimators = clf.estimators_
                    total_estimators += estimators
                    
                    # LightGBM for regression
                    train_data = lgb.Dataset(sampled_X, label=sampled_y)
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

    def get_weighted_feature_importances(self, X, y, estimator, random_state):
        """Return feature importances weighted by model performance"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state)
        
        # Choose appropriate model based on task type
        if self.task_type == 'classification':
            estm = RandomForestClassifier(random_state=random_state, n_jobs=self.n_jobs)
        else:  # regression
            estm = RandomForestRegressor(random_state=random_state, n_jobs=self.n_jobs)
            
        estm.fit(X_train, y_train)
        ests = estm.estimators_
        model = estm
        imps = np.zeros((len(model.estimators_), X.shape[1]))
        scores = np.zeros(len(model.estimators_))
        
        for i, each in enumerate(model.estimators_):
            # Different scoring metrics based on task type
            if self.task_type == 'classification':
                y_probas_train = each.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_probas_train)
            else:  # regression
                y_pred_train = each.predict(X_test)
                score = r2_score(y_test, y_pred_train)
                
            imps[i] = each.feature_importances_
            scores[i] = score
            
        weights = scores / scores.sum()
        return np.average(imps, axis=0, weights=weights)

    def feat_with_depth(self, X, depth, op_ls, feat_ls):
        """ Recursively generate a new features """
        if depth == 0:
            feat_ind = self.rng.choice(np.arange(len(self.ig_vector)), p=self.ig_vector)
            feat_ls.append(feat_ind)
            return X[:, feat_ind]
        depth -= 1
        op = self.rng.choice(self.operators, p=self.operator_weights)
        if op in self.binary_operators:
            feat_1 = self.feat_with_depth(X, depth, op_ls, feat_ls)
            feat_2 = self.feat_with_depth(X, depth, op_ls, feat_ls)
            op_ls.append((op, depth))
            return op(feat_1, feat_2)
        elif op in self.unary_operators:
            feat_1 = self.feat_with_depth(X, depth, op_ls, feat_ls)
            op_ls.append((op, depth))
            return op(feat_1)

    def feat_with_depth_gen(self, X, depth, op_ls, feat_ls):
        """ Reproduce generated features with new data """
        if depth == 0:
            feat_ind = feat_ls.pop()
            return X[:, feat_ind]
        depth -= 1
        op = op_ls.pop()[0]
        if op in self.binary_operators:
            feat_1 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls)
            feat_2 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls)
            return op(feat_2, feat_1)
        elif op in self.unary_operators:
            feat_1 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls)
            return op(feat_1)

    def check_correlations(self, feats):
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

    def get_paths(self, clf, feature_names):
        """ Returns every path in the decision tree"""
        tree_ = clf.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        path = []
        path_list = []

        def recurse(node, depth, path_list):
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                path_list.append(path.copy())
            else:
                name = feature_name[node]
                path.append(name)
                recurse(tree_.children_left[node], depth + 1, path_list)
                recurse(tree_.children_right[node], depth + 1, path_list)
                path.pop()

        recurse(0, 1, path_list)

        new_list = []
        for i in range(len(path_list)):
            if path_list[i] != path_list[i - 1]:
                new_list.append(path_list[i])
        return new_list

    def get_combos(self, paths, comb_mat):
        """ Fills Combination matrix with values """
        for i in range(len(comb_mat)):
            for pt in paths:
                if i in pt:
                    comb_mat[i][pt] += 1

    def get_split_feats(self, paths, split_vec):
        """ Fills split vector with values """
        for i in range(len(split_vec)):
            for pt in paths:
                if i in pt:
                    split_vec[i] += 1
