# What is BigFeat?
BigFeat is a scalable and interpretable automated feature engineering framework that optimizes for improving the quality of input features with the aim to maximize the predictive performance according to a user-defined metric.
BigFeat employs a dynamic feature generation and selection mechanism that constructs a set of expressive features that improve the prediction performance while retaining interpretability.


# input/output:
Bigfeat takes the original input features and returns a collection of base and engineered features expected to improve the predictive performance.

# Installation:
  - **pip** : pip install bigfeat 
  - **local installation**: pip install BigFeat/.

# Requirement:
  - 'pandas',
  - 'numpy',
  - 'scikit-learn',
  - 'lightgbm'
  
# Run BigFeat

#### To run bigfeat on the test datasets

```python
import pandas as pd
import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import bigfeat.bigfeat_base as bigfeat
import pandas as pd
import sklearn.preprocessing as preprocessing


def run_tst(df_path,target_ft, random_state):
    df = pd.read_csv(df_path)
    object_columns = df.select_dtypes(include='object')
    if len(object_columns.columns):
        df[object_columns.columns] = object_columns.apply(preprocessing.LabelEncoder().fit_transform)
    X = df.drop(columns=target_ft)
    y = df[target_ft]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test




bf = bigfeat.BigFeat()
datasets=[
        
        
    ("data/shuttle.csv","class"),
    ("data/blood-transfusion-service-center.csv","Class"),
    ("data/credit-g.csv","class"),
    ("data/kc1.csv","defects"),
    ("data/nomao.csv", "Class"),
    ("data/eeg_eye_state.csv", "Class"),
    ("data/gina.csv", "class"),
    ("data/sonar.csv","Class"),
    ("data/arcene.csv", "Class"),
    ("data/madelon.csv", "Class"),
  
    

    ]



# titles = ['Dataset','Original','Random','Smart']
for each in datasets:
    X_train, X_test, y_train, y_test = run_tst(each[0], each[1],random_state=0)
    res = bf.fit(X_train, y_train, gen_size=5,random_state=0, iterations=5,estimator='avg',feat_imps = True, split_feats = None, check_corr= False, selection = 'fAnova', combine_res = True)
    print(each[0])
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("ORG f1 score : {}".format(f1_score(y_test, y_pred)))
    clf = LogisticRegression(random_state=0).fit(bf.transform(X_train), y_train)
    y_pred_bf = clf.predict(bf.transform(X_test))
    print("BF f1 score : {}".format(f1_score(y_test, y_pred_bf)))
    #print(res)

```

  
  
  ## Paper is under submission to IEEE BigData research conference.
