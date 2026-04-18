"""
local_utils.py
--------------
Provides utility functions and unary mathematical operators for feature transformation and grouping.
"""

import numpy as np
import pandas as pd
import scipy.stats

def unary_cube(arr):
    """
    Computes the element-wise cube of the input array.
    """
    return np.power(arr, 3)

def unary_multinv(arr):
    """
    Computes the element-wise multiplicative inverse of the input array.
    """
    return 1 / arr

def unary_sqrtabs(arr):
    """
    Computes the element-wise square root of the absolute values, preserving the original sign.
    """
    return np.sqrt(np.abs(arr)) * np.sign(arr)

def unary_logabs(arr):
    """
    Computes the element-wise natural logarithm of the absolute values, preserving the original sign.
    """
    return np.log(np.abs(arr)) * np.sign(arr)

def convert_with_max(arr):
    """
    Clips array values to the range of float32 to prevent overflow and converts type.
    """
    arr[arr > np.finfo(np.dtype('float32')).max] = np.finfo(np.dtype('float32')).max
    arr[arr < np.finfo(np.dtype('float32')).min] = np.finfo(np.dtype('float32')).min
    return np.float32(arr)

def mode(ar1):
    """
    Calculates the statistical mode of the input array.
    """
    return scipy.stats.mode(ar1, keepdims=True).mode[0]

def ar_range(ar1):
    """
    Calculates the range (max - min) of the input array.
    """
    return ar1.max() - ar1.min()

def percentile_25(ar1):
    """
    Calculates the 25th percentile of the input array.
    """
    return np.percentile(ar1, 25)

def percentile_75(ar1):
    """
    Calculates the 75th percentile of the input array.
    """
    return np.percentile(ar1, 75)

def group_by(ar1, ar2):
    """
    Groups the second array by the values of the first array and applies a randomly selected aggregation.
    """
    rng = np.random.default_rng(seed=42)
    group_by_ops = [np.mean, np.std, np.max, np.min, np.sum, mode, len, ar_range, np.median, percentile_25, percentile_75]
    group_by_op = rng.choice(group_by_ops)
    temp_df = pd.DataFrame({'ar1': ar1, 'ar2': ar2})
    group_res = temp_df.groupby(['ar1'])['ar2'].apply(group_by_op).to_dict() 
    return temp_df['ar1'].map(group_res).values

def original_feat(ar1):
    """
    Identity function that returns the input array unchanged.
    """
    return ar1