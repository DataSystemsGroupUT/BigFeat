import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import local_utils
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class BigFeat:
    """Base BigFeat Class"""

    def __init__(self):
        self.n_jobs = -1
        self.operators = [np.multiply, np.add, np.subtract, np.abs,np.square]
        self.binary_operators = [np.multiply, np.add, np.subtract]
        self.unary_operators = [np.abs,np.square,local_utils.original_feat]

        
        pass

    def fit(self,X,y,gen_size=5,random_state=0):
        self.gen_steps = []
        self.n_feats = X.shape[1]
        self.n_rows = X.shape[0]
        self.ig_vector = np.ones(self.n_feats)/self.n_feats
        #Set RNG seed if provided for numpy
        self.rng = np.random.RandomState(seed=random_state)
        gen_feats = np.zeros((self.n_rows, self.n_feats*gen_size))
        self.op_order = np.zeros(self.n_feats*gen_size, dtype='object')

        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        for i in range(gen_feats.shape[1]):
            self.op_order[i] = self.gen_feat(X)
            if len(self.op_order[i]) == 3:
                gen_feats[:,i] = self.op_order[i][0](X[:,self.op_order[i][1]],X[:,self.op_order[i][2]])
            elif len(self.op_order[i]) == 2:
                gen_feats[:,i] = self.op_order[i][0](X[:,self.op_order[i][1]])
            else:
                print('____EROR_____')
            #gen_feats[:,i] = self.gen_feat(X)

        #self.op_order = np.hstack((self.op_order,np.arange(self.n_feats)))
        #gen_feats = np.hstack((gen_feats,X))

        if True:
            gen_feats, to_drop_cor = self.check_corolations(gen_feats)
            self.op_order = np.delete(self.op_order,to_drop_cor) 



        #OG SELECTION

        imps = self.get_feature_importances(gen_feats,y,None,random_state)
        #imps = self.get_weighted_feature_importances(gen_feats,y,None,random_state)
      
        total_feats = np.argsort(imps)
        feat_args = total_feats[-self.n_feats:]
        gen_feats = gen_feats[:,feat_args]
        self.op_order = self.op_order[feat_args]

        #SEQ SELECTOIN

        #feat_args = self.seq_importances(gen_feats,y,random_state)
        #gen_feats = gen_feats[:,feat_args]
        #self.op_order = self.op_order[feat_args]
        ###


        #print(gen_feats[0,:6])
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
        gen_feats = np.zeros((self.n_rows, len(self.op_order)))

        for i in range(len(self.op_order)):
            if type(self.op_order[i]) == int:
                gen_feats[:,i] = X[:,self.op_order[i]]
            elif len(self.op_order[i]) == 3:
                gen_feats[:,i] = self.op_order[i][0](X[:,self.op_order[i][1]],X[:,self.op_order[i][2]])
            elif len(self.op_order[i]) == 2:
                gen_feats[:,i] = self.op_order[i][0](X[:,self.op_order[i][1]])
                #print(gen_feats[0,:6])

            else:
                print('____EROR_____')

        gen_feats = np.hstack((gen_feats,X))
        return gen_feats

    def get_feature_importances(self,X,y,estimator,random_state):
        """Return feature importances by specifeid method """
        estm = RandomForestClassifier(random_state=random_state,n_jobs=self.n_jobs)
        estm.fit(X,y)
        return estm.feature_importances_

    
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


    def gen_feat(self, X):
        feat_ind_1 = self.rng.choice(np.arange(len(self.ig_vector )),p=self.ig_vector)
        feat_ind_2 = self.rng.choice(np.arange(len(self.ig_vector )),p=self.ig_vector)
        op = self.rng.choice(self.operators)
        if op in self.binary_operators:
            return op,feat_ind_1,feat_ind_2
        elif op in self.unary_operators:
            return op,feat_ind_1

        #return op(X[:,feat_ind_1],X[:,feat_ind_2])
    

    def check_corolations(self,feats):
        cor_thresh = 0.8
        corr_matrix = pd.DataFrame(feats).corr().abs()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        tri_df = corr_matrix.mask(mask)
        to_drop = [c for c in tri_df.columns if any(tri_df[c] > cor_thresh)]
        feats = pd.DataFrame(feats).drop(to_drop,axis=1)
        return feats.values,to_drop

    def seq_importances(self, X,y, random_state=0):
        #estm = RandomForestClassifier(random_state=random_state,n_jobs=self.n_jobs)
        sfs = SequentialFeatureSelector(estm, n_features_to_select=self.n_feats)
        sfs.fit(X, y)
        return sfs.get_support()