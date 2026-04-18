"""
test_utils.py
-------------
Unit tests for mathematical operators and utility functions in local_utils.py.
"""

import numpy as np
import pytest
from bigfeat.local_utils import (
    unary_cube,
    unary_multinv,
    unary_sqrtabs,
    unary_logabs,
    convert_with_max,
    original_feat
)

def test_unary_cube():
    """Test standard cubing operation."""
    arr = np.array([-2, 0, 3])
    expected = np.array([-8, 0, 27])
    np.testing.assert_array_equal(unary_cube(arr), expected)

def test_unary_multinv():
    """Test multiplicative inverse, including safe handling of non-zero values."""
    arr = np.array([2.0, -0.5, 4.0])
    expected = np.array([0.5, -2.0, 0.25])
    np.testing.assert_array_equal(unary_multinv(arr), expected)

def test_unary_multinv_divide_by_zero():
    """Test how numpy handles divide by zero (should warn/inf, but not crash)."""
    arr = np.array([0.0])
    with np.errstate(divide='ignore'):  # We expect a division by zero warning
        res = unary_multinv(arr)
    assert np.isinf(res[0])

def test_unary_sqrtabs():
    """Test square root of absolute values, keeping the original sign."""
    arr = np.array([-4.0, 0.0, 9.0])
    expected = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_array_equal(unary_sqrtabs(arr), expected)

def test_unary_logabs():
    """Test log of absolute values, keeping the original sign."""
    # Using np.exp(1) ~ 2.718 to get a clean log value of 1.0
    arr = np.array([-np.exp(1), np.exp(2)])
    expected = np.array([-1.0, 2.0])
    np.testing.assert_allclose(unary_logabs(arr), expected, rtol=1e-5)

def test_unary_logabs_zero():
    """Test log of zero (should return -inf but keep running)."""
    arr = np.array([0.0])
    with np.errstate(divide='ignore'):
        res = unary_logabs(arr)
    # The sign of 0 is 0, so 0 * -inf = nan. 
    # We just ensure it doesn't crash the script.
    assert np.isnan(res[0])

def test_convert_with_max():
    """Test clipping of excessively large/small numbers to float32 limits."""
    # Create numbers larger than float32 max
    huge_positive = np.array([1e40])
    huge_negative = np.array([-1e40])
    
    clipped_pos = convert_with_max(huge_positive)
    clipped_neg = convert_with_max(huge_negative)
    
    assert clipped_pos[0] == np.finfo(np.float32).max
    assert clipped_neg[0] == np.finfo(np.float32).min
    assert clipped_pos.dtype == np.float32

def test_original_feat():
    """Test that original_feat returns the array unchanged."""
    arr = np.array([1, 2, 3])
    res = original_feat(arr)
    np.testing.assert_array_equal(arr, res)