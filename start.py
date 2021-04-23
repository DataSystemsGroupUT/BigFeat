import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from paths import paths
import bigfeat

def load_data(data_path,target_col): 
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def run_rf(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_probas_train = clf.predict_proba(X_train)[:, 1]
    y_probas_test = clf.predict_proba(X_test)[:, 1]
    roc_train = roc_auc_score(y_train, y_probas_train)
    roc_test = roc_auc_score(y_test, y_probas_test)
    return roc_train, roc_test

def run_test(data_path,target_col):
    print("--------- Running: {} ---------".format(data_path))
    X_train, X_test, y_train, y_test = load_data(data_path,target_col)
    print('Raw:')
    rs = run_rf(X_train, X_test, y_train, y_test)
    print(rs)
    print('BigFeat:')
    bf = bigfeat.BigFeat()
    X_train = bf.fit(X_train, y_train)
    X_test = bf.produce(X_test)
    #return 0
    rs = run_rf(X_train, X_test, y_train, y_test)
    print(rs)


if __name__ == "__main__":
    for dataset in paths:
        run_test(dataset[0],dataset[1])