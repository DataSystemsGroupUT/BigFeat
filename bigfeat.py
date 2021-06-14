import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import local_utils
#from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
from numba import jit


class BigFeat:
    """Base BigFeat Class"""

    def __init__(self):
        self.n_jobs = -1
        self.operators = [np.multiply, np.add, np.subtract, np.abs,np.square]
        self.binary_operators = [np.multiply, np.add, np.subtract]
        self.unary_operators = [np.abs,np.square,local_utils.original_feat]

    #@jit
    def fit(self,X,y,gen_size=5,random_state=0, feat_imps = False, split_feats = None):
        self.tracking_ops = []
        self.tracking_ids = []
        self.gen_steps = []
        self.n_feats = X.shape[1]
        self.n_rows = X.shape[0]
        self.ig_vector = np.ones(self.n_feats)/self.n_feats
        self.comb_mat = np.ones((self.n_feats,self.n_feats))
        self.split_vec = np.ones(self.n_feats)
        #Set RNG seed if provided for numpy
        self.rng = np.random.RandomState(seed=random_state)
        gen_feats = np.zeros((self.n_rows, self.n_feats*gen_size))
        self.op_order = np.zeros(self.n_feats*gen_size, dtype='object')
        self.feat_depths = np.zeros(gen_feats.shape[1])
        self.depth_range = np.arange(3)+1
        self.depth_weights = 1/(2**self.depth_range)
        self.depth_weights /= self.depth_weights .sum()

        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        if feat_imps:
            self.ig_vector, estimators = self.get_feature_importances(X,y,None,random_state)
            self.ig_vector /= self.ig_vector.sum()
            for tree in estimators:
                paths = self.get_paths(tree,np.arange(X.shape[1]))
                self.get_split_feats(paths, self.split_vec)
                self.split_vec /= self.split_vec.sum()
            if split_feats == "comb":
                self.ig_vector = np.multiply(self.ig_vector,self.split_vec)
                self.ig_vector /= self.ig_vector.sum()
            elif split_feats == "splits":
                self.ig_vector = self.split_vec

        for i in range(gen_feats.shape[1]):
            #dpth  = 3
            dpth = self.rng.choice(self.depth_range,p=self.depth_weights)
            ops = []
            ids = []
            gen_feats[:,i] = self.feat_with_depth(X,dpth,ops,ids)
            self.feat_depths[i] = dpth
            self.tracking_ops.append(ops)
            self.tracking_ids.append(ids)
        self.tracking_ids = np.array(self.tracking_ids,dtype='object')
        self.tracking_ops = np.array(self.tracking_ops,dtype='object')
        #self.op_order = np.hstack((self.op_order,np.arange(self.n_feats)))
        #gen_feats = np.hstack((gen_feats,X))

        if False:
            gen_feats, to_drop_cor = self.check_corolations(gen_feats)
            self.op_order = np.delete(self.op_order,to_drop_cor) 


        #OG SELECTION

        imps, estimators = self.get_feature_importances(gen_feats,y,None,random_state)
        #imps = self.get_weighted_feature_importances(gen_feats,y,None,random_state)
        total_feats = np.argsort(imps)
        feat_args = total_feats[-self.n_feats:]
        gen_feats = gen_feats[:,feat_args]
        self.tracking_ids = self.tracking_ids[feat_args]
        self.tracking_ops = self.tracking_ops[feat_args]
        self.feat_depths = self.feat_depths[feat_args]

        #SEQ SELECTOIN

        #feat_args = self.seq_importances(gen_feats,y,random_state)
        #gen_feats = gen_feats[:,feat_args]
        #self.op_order = self.op_order[feat_args]
        ###

        #print('-----------------')
        #gen_feats = np.hstack((gen_feats,X))
        #self.op_order = np.hstack((self.op_order,np.arange(self.n_feats)))
        gen_feats = np.hstack((gen_feats,X))

        if False:
            gen_feats, to_drop_cor = self.check_corolations(gen_feats)
            self.op_order = np.delete(self.op_order ,to_drop_cor) 
        return gen_feats



    def produce(self,X):
        X = self.scaler.transform(X)
        self.n_rows = X.shape[0]
        gen_feats = np.zeros((self.n_rows, len(self.tracking_ids)))
        for i in range(gen_feats.shape[1]):
            dpth = self.feat_depths[i]
            op_ls = self.tracking_ops[i].copy()
            id_ls = self.tracking_ids[i].copy()
            gen_feats[:,i] = self.feat_with_depth_gen(X,dpth,op_ls,id_ls)
        gen_feats = np.hstack((gen_feats,X))
        return gen_feats


    def get_feature_importances(self,X,y,estimator,random_state):
        """Return feature importances by specifeid method """
        estm = RandomForestClassifier(random_state=random_state,n_jobs=self.n_jobs)
        estm.fit(X,y)
        return estm.feature_importances_, estm.estimators_

    def get_weighted_feature_importances(self,X,y,estimator,random_state):
        """Return feature importances by specifeid method """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state)
        
        estm = RandomForestClassifier(random_state=random_state,n_jobs=self.n_jobs)
        estm.fit(X_train,y_train)
        ests = estm.estimators_
        model = estm
        imps = np.zeros((len(model.estimators_),X.shape[1]))
        scores = np.zeros(len(model.estimators_))
        for i,each in enumerate(model.estimators_):
            y_probas_train = each.predict_proba(X_test)[:, 1]
            roc_train = roc_auc_score(y_test, y_probas_train)
            #print(roc_train)
            imps[i]=each.feature_importances_
            scores[i] = roc_train
        #return np.array((roc_train, roc_test))
        weights = scores/scores.sum()
        return np.average(imps,axis=0, weights=weights)

    #@jit(nopython=True)
    def gen_feat(self, X):
        feat_ind_1 = self.rng.choice(np.arange(len(self.ig_vector )),p=self.ig_vector)
        feat_ind_2 = self.rng.choice(np.arange(len(self.ig_vector )),p=self.ig_vector)
        op = self.rng.choice(self.operators)
        if op in self.binary_operators:
            return op,feat_ind_1,feat_ind_2
        elif op in self.unary_operators:
            return op,feat_ind_1

    #@jit
    def feat_with_depth(self, X, depth, op_ls, feat_ls):
        if depth == 0:
            feat_ind = self.rng.choice(np.arange(len(self.ig_vector )),p=self.ig_vector)
            feat_ls.append(feat_ind)
            return X[:,feat_ind]
        depth -= 1
        op = self.rng.choice(self.operators)
        if op in self.binary_operators:
            feat_1 = self.feat_with_depth(X,depth,op_ls, feat_ls)
            feat_2 = self.feat_with_depth(X,depth,op_ls, feat_ls)
            op_ls.append((op,depth))
            return op(feat_1,feat_2)
        elif op in self.unary_operators:
            feat_1 = self.feat_with_depth(X,depth,op_ls, feat_ls)
            op_ls.append((op,depth))
            return op(feat_1)

    def feat_with_depth_gen(self, X, depth,op_ls, feat_ls):
        if depth == 0:
            #print('aaa')
            feat_ind = feat_ls.pop()
            return X[:,feat_ind]
        #print('bbbb')
        depth -= 1
        op = op_ls.pop()[0]
        #print(op)
        if op in self.binary_operators:
            feat_1 = self.feat_with_depth_gen(X,depth, op_ls, feat_ls)
            feat_2 = self.feat_with_depth_gen(X,depth, op_ls, feat_ls)
            return op(feat_2,feat_1)
        elif op in self.unary_operators:
            feat_1 = self.feat_with_depth_gen(X,depth, op_ls, feat_ls)
            return op(feat_1)


    def check_corolations(self,feats):
        cor_thresh = 0.8
        corr_matrix = pd.DataFrame(feats).corr().abs()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        tri_df = corr_matrix.mask(mask)
        to_drop = [c for c in tri_df.columns if any(tri_df[c] > cor_thresh)]
        feats = pd.DataFrame(feats).drop(to_drop,axis=1)
        return feats.values,to_drop


    def get_paths(self,clf, feature_names):
        """ Returns every path in the decision tree"""

        tree_ = clf.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        path = []
        path_list = []
        def recurse(node, depth,path_list):
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                path_list.append(path.copy())
            else:
                name = feature_name[node]
                path.append(name)
                recurse(tree_.children_left[node], depth + 1,path_list)
                recurse(tree_.children_right[node], depth + 1,path_list)
                path.pop()
        recurse(0, 1,path_list) 

        new_list = []
        for i in range(len(path_list)):
            if path_list[i] != path_list[i-1]:
                new_list.append(path_list[i])
        return new_list

    def get_combos(self,paths,comb_mat):
        """ Filles Combination matrix with values """

        for i in range(len(comb_mat)):
            for pt in paths:
                if i in pt:
                    comb_mat[i][pt]+=1


    def get_split_feats(self,paths,split_vec):
        """ Filles Combination matrix with values """

        for i in range(len(split_vec)):
            for pt in paths:
                if i in pt:
                    split_vec[i] += 1

    #def seq_importances(self, X,y, random_state=0):
        #estm = RandomForestClassifier(random_state=random_state,n_jobs=self.n_jobs)
    #    sfs = SequentialFeatureSelector(estm, n_features_to_select=self.n_feats)
    #    sfs.fit(X, y)
    #    return sfs.get_support()