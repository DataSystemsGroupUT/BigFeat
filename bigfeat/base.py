"""
base.py
-------
The main orchestrator class for BigFeat.
"""
import numpy as np
import ray
# debug comment
# 1111111111
from sklearn.preprocessing import MinMaxScaler
import bigfeat.local_utils as local_utils
from .config import initialize_ray
from .distributed_tasks import remote_generate_batch, remote_get_importance

# Importing from our refactored modules
from .generator import feat_with_depth, feat_with_depth_gen
from .importance import get_feature_importances
from .tree_utils import get_paths, get_split_feats
from .selection import check_correlations, fit_fanova
from .evaluation import select_estimator as eval_select_estimator


class BigFeat:
    """Base BigFeat Class for both classification and regression tasks"""
    def __init__(self, task_type='classification', options=None):
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be either 'classification' or 'regression'")

        self.task_type = task_type
        self.options = options or {} 
        self.n_jobs = -1
        self.fanova_best = None
        
        self.operators = [np.multiply, np.add, np.subtract, np.abs, np.square]
        self.binary_operators = [np.multiply, np.add, np.subtract]
        self.unary_operators = [np.abs, np.square, local_utils.original_feat]

    def _calculate_initial_importance(self, x_ref, y_ref, random_state, split_feats):
        future = remote_get_importance.remote(
            x_ref, y_ref, "avg", self.task_type, random_state, self.n_jobs
        )
        importance_sum, splits_sum = ray.get(future)
        
        ig_vector = importance_sum / (importance_sum.sum() + 1e-9)
        split_vec = splits_sum / (splits_sum.sum() + 1e-9)
        
        if split_feats == "comb":
            ig_vector = np.multiply(ig_vector, split_vec)
            ig_vector /= (ig_vector.sum() + 1e-9)
        elif split_feats == "splits":
            ig_vector = split_vec
        return ig_vector


    def _get_initial_ig(self, x_ref, y_ref, random_state, split_feats):
        """Helper to calculate initial feature importance."""
        future = remote_get_importance.remote(
            x_ref, y_ref, "avg", self.task_type, random_state, self.n_jobs
        )
        importance_sum, splits_sum = ray.get(future)
        ig_vector = importance_sum / (importance_sum.sum() + 1e-9)
        
        if split_feats in ["comb", "splits"]:
            split_vec = splits_sum / (splits_sum.sum() + 1e-9)
            if split_feats == "comb":
                ig_vector = np.multiply(ig_vector, split_vec)
            else:
                ig_vector = split_vec
            ig_vector /= (ig_vector.sum() + 1e-9)
        return ig_vector

    def _generate_iteration_batches(self, x_ref, iteration, num_to_gen, batch_size, random_state):
        """Helper to manage Ray tasks batching."""
        gen_futures = []
        for i in range(0, num_to_gen, batch_size):
            curr_size = min(batch_size, num_to_gen - i)
            batch_depths = [self.rng.choice(self.depth_range, p=self.depth_weights) for _ in range(curr_size)]
            gen_futures.append(remote_generate_batch.remote(
                x_ref, batch_depths, random_state + i + iteration,
                self.ig_vector, self.operators, self.operator_weights,
                self.binary_operators, self.unary_operators))
        return gen_futures

    def _update_weights(self, selected_ops):
        """Update operator weights based on their usage in selected features."""
        for i_op, op in enumerate(self.operators):
            for feat in selected_ops:
                if any(op == f_op[0] for f_op in feat):
                    self.imp_operators[i_op] += 1
        self.operator_weights = self.imp_operators / self.imp_operators.sum()

    def fit(self, x, y, gen_size=5, random_state=0, iterations=5, estimator='avg', 
            feat_imps=True, split_feats=None, check_corr=True, selection='stability', combine_res=True):
        initialize_ray(self.options)
        self.rng = np.random.default_rng(seed=random_state)
        self.n_feats, self.n_rows = x.shape[1], x.shape[0]
        self.selection, self.imp_operators = selection, np.ones(len(self.operators))
        self.operator_weights = self.imp_operators / self.imp_operators.sum()
        self.ig_vector = np.ones(self.n_feats) / self.n_feats
        self.depth_range = np.arange(3) + 1
        self.depth_weights = (1 / (2 ** self.depth_range)) / (1 / (2 ** self.depth_range)).sum()
        
        self.scaler = MinMaxScaler()
        x_scaled = self.scaler.fit_transform(x)
        x_ref, y_ref = ray.put(x_scaled), ray.put(y)
        
        if feat_imps:
            self.ig_vector = self._get_initial_ig(x_ref, y_ref, random_state, split_feats)

        num_to_gen = self.n_feats * gen_size
        batch_size = max(1, num_to_gen // (int(ray.cluster_resources().get("CPU", 1)) * 2))
        iters_comb = np.zeros((self.n_rows, self.n_feats * iterations))
        depths_comb = np.zeros(self.n_feats * iterations)
        ids_comb, ops_comb = [None] * (self.n_feats * iterations), [None] * (self.n_feats * iterations)

        for iteration in range(iterations):
            futures = self._generate_iteration_batches(x_ref, iteration, num_to_gen, batch_size, random_state)
            flattened = [item for batch in ray.get(futures) for item in batch]
            gen_feats_iter = np.column_stack([res[0] for res in flattened])
            
            imps_iter, _ = get_feature_importances(gen_feats_iter, y, estimator, random_state, self.task_type, n_jobs=self.n_jobs)
            feat_args = np.argsort(imps_iter)[-self.n_feats:]
            
            start, end = iteration * self.n_feats, (iteration + 1) * self.n_feats
            iters_comb[:, start:end] = gen_feats_iter[:, feat_args]
            depths_comb[start:end] = np.array([flattened[idx][3] for idx in feat_args])
            
            for k, idx in enumerate(feat_args):
                ids_comb[start + k], ops_comb[start + k] = flattened[idx][2], flattened[idx][1]
            
            self._update_weights([flattened[idx][1] for idx in feat_args])

        # Final selection
        if selection == 'stability' and iterations > 1 and combine_res:
            imps_f, _ = get_feature_importances(iters_comb, y, estimator, random_state, self.task_type, n_jobs=self.n_jobs)
            feat_args = np.argsort(imps_f)[-self.n_feats:]
            gen_feats, self.tracking_ids, self.tracking_ops = iters_comb[:, feat_args], [ids_comb[i] for i in feat_args], [ops_comb[i] for i in feat_args]
            self.feat_depths = depths_comb[feat_args]
        else:
            gen_feats, self.tracking_ids, self.tracking_ops = iters_comb[:, -self.n_feats:], ids_comb[-self.n_feats:], ops_comb[-self.n_feats:]
            self.feat_depths = depths_comb[-self.n_feats:]

        if selection == 'stability' and check_corr:
            gen_feats, to_drop = check_correlations(gen_feats)
            self.tracking_ids = [it for i, it in enumerate(self.tracking_ids) if i not in to_drop]
            self.tracking_ops = [it for i, it in enumerate(self.tracking_ops) if i not in to_drop]
            self.feat_depths = np.delete(self.feat_depths, to_drop)
            
        gen_feats = np.hstack((gen_feats, x_scaled))
        if selection == 'fAnova':
            gen_feats, self.fanova_best = fit_fanova(gen_feats, y, self.task_type, self.n_feats)

        return gen_feats








    def transform(self, x):
        """
        Produce features from the fitted BigFeat object.
        """
        x_scaled = self.scaler.transform(x)
        rows_count = x_scaled.shape[0]
        gen_feats = np.zeros((rows_count, len(self.tracking_ids)))

        for i in range(gen_feats.shape[1]):
            dpth = self.feat_depths[i]
            # Copy lists to prevent modifying the fitted state during pop()
            op_ls = self.tracking_ops[i].copy()
            id_ls = self.tracking_ids[i].copy()

            gen_feats[:, i] = feat_with_depth_gen(
                x_scaled, dpth, op_ls, id_ls, 
                self.binary_operators, self.unary_operators
            )

        gen_feats = np.hstack((gen_feats, x_scaled))

        if self.selection == 'fAnova':
            gen_feats = self.fanova_best.transform(gen_feats)

        return gen_feats

    def select_estimator(self, x, y, estimators_names=None):
        """
        Select the best estimator based on cross-validation.
        
        Parameters:
        -----------
        x : array-like
            Feature matrix.
        y : array-like
            Target vector.
        estimators_names : list or None
            List of estimator names to try.
        
        Returns:
        --------
        model : estimator
            Fitted best estimator.
        """
        return eval_select_estimator(
            x, y, 
            task_type=self.task_type, 
            n_jobs=self.n_jobs, 
            estimators_names=estimators_names
        )