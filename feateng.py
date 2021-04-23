import pandas as pd
import numpy as np
from numpy.random import RandomState
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from datetime import datetime
import local_utils

class BigFeat:
    """Base BigFeat Class"""

    def __init__(self):
        self.selected_operators = None
        self.selected_feat_ids = None
        self.selected_feat_ord = None
    
    def fit(self,X,y, iterations=1,gen_size=10,imp_estimator='rf',random_state=0,optimizer='smart',corr_do=False,n_jobs=1):
        """ Performs Feature Engineering """

        self.n_jobs = n_jobs
        self.iterations = iterations
        #Set RNG seed if provided for numpy
        rng = RandomState(seed=random_state)
        #define operators
        operators = [np.multiply, np.add, np.subtract, np.abs,np.square, local_utils.unary_cube ,local_utils.unary_sqrtabs, local_utils.group_by]
        self.binary_operators = [np.multiply, np.add, np.subtract,local_utils.group_by]
        self.categorical_operators =[local_utils.group_by] 
        self.unary_operators = [ local_utils.unary_logabs, np.exp,np.abs,np.square, local_utils.unary_multinv, local_utils.unary_cube ,local_utils.unary_sqrtabs]
        n_feats = X.shape[1]
        n_rows = X.shape[0]
        self.n_feats = n_feats
        self.n_rows = n_rows

        #Perform Mean Encoding
        unq_count = np.zeros(X.shape[1])
        for c in range(X.shape[1]):
            unq_count[c] = len(X.iloc[:,c].unique())
        categorical = unq_count<50
        if categorical.sum():
            X_comp = X.iloc[:,categorical].copy()
            X_comp['Target'] = y
            for c in X_comp.drop(columns=['Target']).columns:
                Mean_encoded_subject = X_comp.groupby([c])['Target'].mean().to_dict() 
                X[c] =  X_comp[c].map(Mean_encoded_subject)
        else:
            #If no categorical features in teh dataset, remove categorical operators form the list
            [operators.remove(x) for x in self.categorical_operators] 
            [self.binary_operators.remove(x) for x in self.categorical_operators] 
        self.categorical = categorical==True
        #normilize features
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        #initialize and normilize vectors for importances
        comb_mat = np.ones((X.shape[1],X.shape[1]))
        ig_vector = np.ones_like(X[0])
        if optimizer=='smart':
            #If smart is on, use random forest to calculate importances
            ig_vector, estimators = self.get_feature_importances(X,y,imp_estimator,random_state,n_jobs)
            #And calculate combinatations
            for tree in estimators:
                if imp_estimator=='gb':
                    paths = self.get_paths(tree[0],np.arange(X.shape[1]))
                elif imp_estimator=='rf' or imp_estimator =='mi' or imp_estimator=='avg':
                    paths = self.get_paths(tree,np.arange(X.shape[1]))
                self.get_combos(paths,comb_mat)

        self.comb_mat = comb_mat
        imp_operators = np.ones(len(operators))
        norm_ig = ig_vector/ig_vector.sum()
        self.norm_cat_ig = ig_vector.copy()
        self.norm_cat_ig[~self.categorical] = 0
        if self.norm_cat_ig.sum():
            self.norm_cat_ig /= ig_vector[self.categorical].sum()
        #initilize zero matrix to populate with generated features
        ng_feats = n_feats * gen_size
        gen_feats = np.zeros((n_rows,ng_feats))
        iter_best_feats = []
        iter_ids = []
        iter_ops = []
        iter_ord = []
        depth_ls = [1, 1, 1, 1, 1, 1, 2]

        for iteration in range(iterations):
            gen_feats = np.zeros((n_rows,ng_feats))
            feat_ops = np.zeros(ng_feats,dtype=np.object)
            feat_ids = np.zeros(ng_feats,dtype=np.object)
            feat_ord = np.zeros(ng_feats,dtype=np.object)
            norm_operators = imp_operators/imp_operators.sum()
            is_last_iter = iteration == iterations-1
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Start Time =", current_time)
            print('Iteration : ',iteration)
            print('Progress: ',0,'%', end="\r")
            depth =depth_ls[iteration]
            #generate features
            print('Progress: ',int(depth/(iteration+2)*100),'%',end="\r")
            for i in range(ng_feats):
                op_list = []
                base_feats = []
                reconst_list = []
                gen_feats[:,i],_ =self.gen_feat(X, norm_ig, operators, norm_operators, depth, op_list, base_feats,reconst_list,rng=rng)
                feat_ops[i] = op_list
                feat_ids[i] = base_feats
                feat_ord[i] = reconst_list
            gen_feats = np.nan_to_num(gen_feats,posinf=999,neginf=-999)
            print('Progress: ',100,'%')

            to_drop_cor = []
            if corr_do:
                gen_feats, to_drop_cor = self.check_corolations(gen_feats)
            feat_ops = np.delete(feat_ops,to_drop_cor) 
            feat_ids = np.delete(feat_ids,to_drop_cor) 
            feat_ord = np.delete(feat_ord,to_drop_cor) 

            if imp_estimator !='mi':
                imps, _ = self.get_feature_importances(gen_feats,y,imp_estimator,random_state,n_jobs)
                #sort by importance
                total_feats = np.argsort(imps)
                feat_args = total_feats[-n_feats:]
            else:
                feat_args = self.mu_selector(gen_feats,y,n_feats)

            self.selected_operators = feat_ops[feat_args]
            self.selected_feat_ids = feat_ids[feat_args]
            self.selected_feat_ord = feat_ord[feat_args]
            imp_operators = imp_operators/imp_operators.sum()
            if not is_last_iter:
            #modify operator importance vector
                if optimizer=='smart':
                    curr_imp_operators = np.zeros(len(operators))
                    for i, op in enumerate(operators):
                        for feat in feat_ops[feat_args]:
                            if op in feat:
                                curr_imp_operators[i] += 1

                    curr_imp_operators = curr_imp_operators/curr_imp_operators.sum()
                    imp_operators = ((iteration * imp_operators) + curr_imp_operators)/(iteration+1)
            
            iter_best_feats.append(gen_feats[:,feat_args])
            iter_ops.append(feat_ops[feat_args])
            iter_ids.append(feat_ids[feat_args])
            iter_ord.append(feat_ord[feat_args])

        print("Performing Final Selecton...")
        # Start Last iter
        gen_feats = np.concatenate(iter_best_feats,axis=1)
        feat_ops = np.concatenate(iter_ops)
        feat_ids = np.concatenate(iter_ids)
        feat_ord = np.concatenate(iter_ord)
        if corr_do:
            gen_feats, to_drop_cor = self.check_corolations(gen_feats)
        feat_ops = np.delete(feat_ops,to_drop_cor) 
        feat_ids = np.delete(feat_ids,to_drop_cor) 
        feat_ord = np.delete(feat_ord,to_drop_cor) 
        gen_feats = np.hstack((gen_feats,X))
        feat_ops_n = np.zeros(len(feat_ops)+n_feats,dtype=np.object)
        feat_ops_n[:-n_feats] = feat_ops
        feat_ops_n[-n_feats:] = 1
        feat_ops = feat_ops_n
        feat_ids_n = np.zeros(len(feat_ids)+n_feats,dtype=np.object)
        feat_ids_n[:-n_feats] = feat_ids
        feat_ids_n[-n_feats:] = np.arange(n_feats)
        feat_ids = feat_ids_n
        feat_ord_n = np.zeros(len(feat_ord)+n_feats,dtype=np.object)
        feat_ord_n[:-n_feats] = feat_ord
        feat_ord_n[-n_feats:] = 1
        feat_ord = feat_ord_n
        imps, _ = self.get_feature_importances(gen_feats,y,imp_estimator,random_state,n_jobs)
        #sort by importance
        total_feats = np.argsort(imps)
        feat_args = total_feats[-n_feats:]
        self.selected_operators = feat_ops[feat_args]
        self.selected_feat_ids = feat_ids[feat_args]
        self.selected_feat_ord = feat_ord[feat_args]       
        #Select final featutres and return
        selected_feats = gen_feats[:,feat_args]
        return selected_feats

    def gen_feat(self, X, norm_ig, operators, norm_operators, depth, op_list,base_feats, reconst_list, feature_1 = None, parent_op = None,rng=np.random):
        """ Recursivly generate features using operators"""

        #chek for termination
        if depth == 0:
            #choose feature based on nomralized information gain vector

            #Check if feature 1 was already selected
            if feature_1 != None:
                #If it is combine  information gain and combinations vector for new weights
                comb_vect = self.comb_mat[feature_1]
                joined_vect = np.multiply(comb_vect,norm_ig)
                joined_vect = joined_vect / joined_vect.sum()
            else:
                #otherwise use information gain
                #check if operator is group by and requires categorical feature
                if parent_op in self.categorical_operators:
                    joined_vect = self.norm_cat_ig
                else:
                    joined_vect = norm_ig
            joined_vect = np.ones_like(joined_vect)/np.ones_like(joined_vect).sum()
            feat_ind = rng.choice(np.arange(len(norm_ig)),p=joined_vect)
            base_feats.append(feat_ind)
            #Return selected feature
            return  X[:,feat_ind], feat_ind
        #decrease depth and call recrsivly
        depth -= 1
        #select operator to be used on the current recursion level
        op = operators[rng.choice(np.arange(len(norm_operators)),p=norm_operators)]
        if op in self.categorical_operators:
            f1,og_feat_1 = self.gen_feat( X, norm_ig, operators, norm_operators, 0, op_list, base_feats, reconst_list,parent_op=op,rng=rng)
            f2,_ = self.gen_feat( X, norm_ig, operators, norm_operators, depth, op_list, base_feats, reconst_list, feature_1 = og_feat_1, rng=rng)
            #apply selected operetator to selected features
            feat = op(f1,f2)
        elif op in self.binary_operators:
            #recursivly generate 2 features
            f1,og_feat_1 = self.gen_feat( X, norm_ig, operators, norm_operators, depth, op_list, base_feats,reconst_list,parent_op=op,rng=rng)
            f2,_ = self.gen_feat( X, norm_ig, operators, norm_operators, depth, op_list, base_feats,reconst_list,feature_1 = og_feat_1, rng=rng)
            #apply selected operetator to selected features
            feat = op(f1,f2)
        elif op in self.unary_operators:
            #recursivly generate 1 feature
            f1,og_feat_1 = self.gen_feat( X, norm_ig, operators, norm_operators, depth, op_list, base_feats,reconst_list, rng=rng)
            #apply selected operetator to selected features
            feat = op(f1)
        else:
            print('Unknown Operator')
        #append selected opertator to the list for later use and return generated feature
        op_list.append(op)
        reconst_list.append((op,depth))
        return feat,og_feat_1

    def get_feature_importances(self,X,y,estimator,random_state,n_jobs,sample_count=5, sample_size=3):
        """Return feature importances by specifeid method """
        

        importance_sum = np.zeros(X.shape[1])
        total_estimators = []
        for sampled in range(sample_count):
            sampled_ind = np.random.choice(np.arange(self.n_rows),size=self.n_rows//sample_size,replace=False)
            sampled_X = X[sampled_ind]
            #sampled_y = y[sampled_ind]
            sampled_y = np.take(y,sampled_ind)

            if estimator == "rf" or estimator == 'mi':
                estm = RandomForestClassifier(random_state=random_state,n_jobs=n_jobs)
                estm.fit(sampled_X,sampled_y)
                total_importances = estm.feature_importances_
                estimators = estm.estimators_
                total_estimators += estimators
            elif estimator =="gb":
                estm = GradientBoostingClassifier(random_state=random_state)
                estm.fit(sampled_X,sampled_y)
                total_importances = estm.feature_importances_
                estimators = estm.estimators_
                #Different for gb as estimators are in differnt format
                total_estimators = estimators
            elif estimator =="avg":
                clf = RandomForestClassifier(random_state=random_state,n_jobs=n_jobs)
                clf.fit(sampled_X, sampled_y)
                rf_importances = clf.feature_importances_
                estimators = clf.estimators_
                total_estimators += estimators
                
                train_data = lgb.Dataset(sampled_X, label=sampled_y)
                param = {'num_leaves': 31, 'objective': 'binary'}
                param['metric'] = 'auc'
                param = {} 
                num_round = 2
                bst = lgb.train(param, train_data, num_round)
                lgb_imps = bst.feature_importance(importance_type='gain')
                lgb_imps /= lgb_imps.sum()
                total_importances = (rf_importances + lgb_imps) /2
            importance_sum +=total_importances
       # return total_importances,estimators
        return importance_sum,total_estimators

    def mu_selector(self,X,y,n_features):
        F = np.ones(X.shape[1])
        S = np.zeros(X.shape[1])
        selected = np.zeros(n_features)
        for ft in range(n_features):
            MU_s =  np.ones(X.shape[1])*-1
            for i in range(len(F)):
                if F[i]:
                    I_y = normalized_mutual_info_score(X[:,i],y)
                    I_s = 0
                    for j in range(len(S)):
                        if S[j]:
                            I_s += normalized_mutual_info_score(X[:,i],X[:,j])
                    m = (S==1).sum()
                    if m:
                        MU = I_y - (I_s)/m
                    else:
                        MU = I_y
                    MU_s[i] = MU
            mx_MI = MU_s.argmax()
            selected[ft]=mx_MI
            F[mx_MI] = 0
            S[mx_MI] = 1
        return selected.astype(int)

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

    def check_base_corr(self, base_feats,gen_feat):
        cor_thresh = 0.95
        for i in range(base_feats.shape[1]):
            cor_val = np.abs(np.corrcoef(base_feats[:,i],gen_feat))[0,1]
            if cor_val > cor_thresh:
                return False
        return True

    def check_corolations(self,feats):
        cor_thresh = 0.9
        corr_matrix = pd.DataFrame(feats).corr().abs()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        tri_df = corr_matrix.mask(mask)
        to_drop = [c for c in tri_df.columns if any(tri_df[c] > cor_thresh)]
        feats = pd.DataFrame(feats).drop(to_drop,axis=1)
        print(feats.shape)
        return feats.values,to_drop

    def check_corolations_post(self,feats,total_feats,gen_feats):
        """ Removes highly corolated features after runing"""

        cor_thresh =  0.999
        all_gen = gen_feats[:, total_feats]  
        corr_sel = np.abs(np.corrcoef(feats,rowvar=False))
        corr_tot = np.abs(np.corrcoef(all_gen,rowvar=False))
        for i in range(self.n_feats):
            if np.sum(corr_sel[:,i]>cor_thresh)>1:
                ind = total_feats[-self.n_feats+i]
                for j in range(self.n_feats):
                    new_ind = corr_tot[:,-self.n_feats-j][-self.n_feats:]
                    if np.sum(new_ind>cor_thresh)>1:
                        feats[:,i] = all_gen[:,-self.n_feats-j]
                        break                    
        return feats

    def produce(self,X):
        """ Produces features from base features once they have been generated """

        final_feats = np.zeros_like(X)
        feat_tmp = np.zeros((X.shape[0],2**self.iterations))
        X = self.scaler.transform(X)
        for i in range(len(self.selected_feat_ord)):
            ords = self.selected_feat_ord[i]
            feats = self.selected_feat_ids[i]
            for depth in range(self.iterations):
                base_ind = 0
                feat_ind = 0
                if ords == 1:
                    feat_tmp[:,0] = X[:,feats]
                    continue
                for each in ords:
                    if each[1] == depth:
                        if each[1] == 0:
                            if each[0] in self.binary_operators:
                                feat1 = X[:,feats[base_ind]]
                                feat2 = X[:,feats[base_ind+1]]
                                feat_tmp[:,feat_ind] = each[0](feat1,feat2)
                                base_ind += 2
                                feat_ind +=1
                            if each[0] in self.unary_operators:
                                feat1 = X[:,feats[base_ind]]
                                feat_tmp[:,feat_ind] = each[0](feat1)
                                base_ind += 1
                                feat_ind +=1
                        else:
                            if each[0] in self.binary_operators:
                                feat1 = feat_tmp[:,base_ind]
                                feat2 = feat_tmp[:,base_ind+1]
                                feat_tmp[:,feat_ind] = each[0](feat1,feat2)
                                base_ind += 2
                                feat_ind +=1
                            if each[0] in self.unary_operators:
                                feat1 = feat_tmp[:,base_ind]
                                feat_tmp[:,feat_ind] = each[0](feat1)
                                base_ind += 1
                                feat_ind +=1
            final_feats[:,i] = feat_tmp[:,0]
        return final_feats

    def compare_feats(self,X,new_X,y,optimizer,random_state=None):
        """ Compares Results of models with original features to generated ones"""

        org_feats = self.rf_test(X,y,random_state=random_state)
        print("Original Features:",org_feats)
        new_feats = self.rf_test(new_X,y,random_state=random_state)
        print(optimizer,"Features:",new_feats)
        return org_feats, new_feats

    def rf_test(self,X,y,random_state=None):
        """ Trains simple RF model on provided data for testing purposes"""

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        rf = RandomForestClassifier(random_state=random_state,n_jobs=self.n_jobs)
        rf.fit(X_train,y_train)
        y_hat = rf.predict(X_test)
        score = accuracy_score(y_hat, y_test)
        y_hat_train = rf.predict(X_train)
        score_train = accuracy_score(y_hat_train, y_train)
        return score,score_train


if __name__ == "__main__":
    ft = BigFeat()
    #df_path =(r"data/eeg_eye_state.csv", "Class")
    #df_path = (r"data/gina.csv", "class")
    df_path = (r"data/banknote.csv", "Class")
    target_ft = df_path[1]
    df = pd.read_csv(df_path[0])
    X = df.drop(columns=target_ft)
    y = df[target_ft]

    res = ft.fit(X,y, iterations=7, gen_size=10,imp_estimator='rf',random_state=0,optimizer="smart")
    ts = ft.produce(X)


