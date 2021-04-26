import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from paths import paths
import bigfeat

def load_data(data_path,target_col,random_state=0): 
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

def run_rf(X_train, X_test, y_train, y_test, random_state=0):
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    y_probas_train = clf.predict_proba(X_train)[:, 1]
    y_probas_test = clf.predict_proba(X_test)[:, 1]
    roc_train = roc_auc_score(y_train, y_probas_train)
    roc_test = roc_auc_score(y_test, y_probas_test)
    return np.array((roc_train, roc_test))

def run_test(data_path,target_col,runs=5):
    print("--------- Running: {} ---------".format(data_path))
    og_rs = np.zeros(2)
    bf_rs = np.zeros(2)

    for i in range(runs):
        X_train, X_test, y_train, y_test = load_data(data_path,target_col,random_state=i)
        og_rs += run_rf(X_train, X_test, y_train, y_test,random_state=i)
        bf = bigfeat.BigFeat()
        X_train = bf.fit(X_train, y_train,random_state=i)
        X_test = bf.produce(X_test)
        bf_rs += run_rf(X_train, X_test, y_train, y_test,random_state=i)
    og_rs /= runs
    bf_rs /= runs
    print('Raw:')
    print(og_rs)
    print('BigFeat:')
    print(bf_rs)


if __name__ == "__main__":
    for dataset in paths:
        run_test(dataset[0],dataset[1],runs= 5)