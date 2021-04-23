import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


class BigFeat:
    """Base BigFeat Class"""

    def __init__(self):
        self.n_jobs = 1
        self.operators = [np.multiply, np.add, np.subtract, np.abs,np.square]
        self.binary_operators = [np.multiply, np.add, np.subtract]
        self.unary_operators = [np.abs,np.square]

        
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

            #gen_feats[:,i] = self.gen_feat(X)




        imps = self.get_feature_importances(X,y,None,random_state)
        total_feats = np.argsort(imps)
        feat_args = total_feats[-self.n_feats:]
        gen_feats = gen_feats[:,feat_args]
        self.op_order = self.op_order[feat_args]
        #print(gen_feats[0,:6])
        #print('-----------------')
        return gen_feats

    def produce(self,X):
        X = self.scaler.transform(X)
        self.n_rows = X.shape[0]
        gen_feats = np.zeros((self.n_rows, len(self.op_order)))

        for i in range(len(self.op_order)):
            if len(self.op_order[i]) == 3:
                gen_feats[:,i] = self.op_order[i][0](X[:,self.op_order[i][1]],X[:,self.op_order[i][2]])
            elif len(self.op_order[i]) == 2:
                gen_feats[:,i] = self.op_order[i][0](X[:,self.op_order[i][1]])
                #print(gen_feats[0,:6])


        return gen_feats

    def get_feature_importances(self,X,y,estimator,random_state):
        """Return feature importances by specifeid method """
        estm = RandomForestClassifier(random_state=random_state,n_jobs=self.n_jobs)
        estm.fit(X,y)
        return estm.feature_importances_

    def gen_feat(self, X):
        feat_ind_1 = self.rng.choice(np.arange(len(self.ig_vector )),p=self.ig_vector)
        feat_ind_2 = self.rng.choice(np.arange(len(self.ig_vector )),p=self.ig_vector)
        op = self.rng.choice(self.operators)
        if op in self.binary_operators:
            return op,feat_ind_1,feat_ind_2
        elif op in self.unary_operators:
            return op,feat_ind_1

        #return op(X[:,feat_ind_1],X[:,feat_ind_2])
        