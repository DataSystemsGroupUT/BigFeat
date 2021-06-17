import numpy as np
import pandas as pd
import scipy.stats

def unary_cube(arr):
    return np.power(arr,3)

def unary_multinv(arr):
    return 1/arr

def unary_sqrtabs(arr):
    return np.sqrt(np.abs(arr)) * np.sign(arr)

def unary_logabs(arr):
    return np.log(np.abs(arr)) * np.sign(arr)

def convert_with_max(arr):
    arr[arr>np.finfo(np.dtype('float32')).max] = np.finfo(np.dtype('float32')).max
    arr[arr<np.finfo(np.dtype('float32')).min] = np.finfo(np.dtype('float32')).min
    return np.float32(arr)

def mode(ar1):
    return scipy.stats.mode(ar1)[0][0]
def ar_range(ar1):
    return ar1.max()-ar1.min()
def percentile_25(ar1):
    return np.percentile(ar1, 25)
def percentile_75(ar1):
    return np.percentile(ar1, 75)


def group_by(ar1,ar2):
    group_by_ops =[np.mean,np.std,np.max,np.min,np.sum,mode,len,ar_range,np.median,percentile_25,percentile_75]
    group_by_op = np.random.choice(group_by_ops)
    temp_df=pd.DataFrame({'ar1':ar1, 'ar2':ar2})
    group_res = temp_df.groupby(['ar1'])['ar2'].apply(group_by_op).to_dict() 
    return temp_df['ar1'].map(group_res).values


def original_feat(ar1):
    return ar1