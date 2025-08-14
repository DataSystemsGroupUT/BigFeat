# BigFeat Time Series Enhancement Documentation

## Overview

This document outlines the enhancements made to the BigFeat library to support time series feature engineering while maintaining 100% backward compatibility with the original implementation.

## Summary of Changes

The enhanced BigFeat adds **optional** time series capabilities without modifying any existing functionality. When time series features are disabled (default behavior), the library behaves identically to the original implementation.

## Core Enhancements

### 1. Time Series Initialization Parameters

The `__init__` method now accepts additional parameters for time series functionality:

```python
def __init__(self, task_type='classification', enable_time_series=False,
             window_sizes=None, lag_periods=None, verbose=True,
             datetime_col=None, groupby_cols=None, time_step='D'):
```

**New Parameters:**
- `enable_time_series` (bool): Enables/disables time series operators (default: False)
- `window_sizes` (list): Time-based window sizes for rolling operations
- `lag_periods` (list): Time-based lag periods for time series operations  
- `verbose` (bool): Progress reporting (default: True)
- `datetime_col` (str): Name of datetime column for time series operations
- `groupby_cols` (list): Columns to group by when applying time series operations
- `time_step` (str): Time step for resampling operations (default: 'D')

### 2. Time Series Operators

When `enable_time_series=True`, 15 new operators are added to the existing operator set:

#### Rolling Window Operations
- `_safe_rolling_mean()`: Time-aware rolling averages
- `_safe_rolling_std()`: Time-aware rolling standard deviation
- `_safe_rolling_min()`: Time-aware rolling minimum
- `_safe_rolling_max()`: Time-aware rolling maximum
- `_safe_rolling_median()`: Time-aware rolling median
- `_safe_rolling_sum()`: Time-aware rolling sum

#### Temporal Transformation Operations
- `_safe_lag_feature()`: Time-aware lagged features
- `_safe_diff_feature()`: Time-aware differencing
- `_safe_pct_change()`: Time-aware percentage changes
- `_safe_ewm()`: Exponential weighted moving averages
- `_safe_momentum()`: Momentum calculations

#### Advanced Time Series Operations
- `_safe_seasonal_decompose()`: Seasonal pattern extraction
- `_safe_trend_feature()`: Trend analysis
- `_safe_weekday_mean()`: Day-of-week patterns
- `_safe_month_mean()`: Monthly patterns

### 3. Intelligent Data Handling

#### DataFrame Processing
```python
# Automatic detection and separation of datetime vs feature columns
if isinstance(X, pd.DataFrame):
    self.original_data = X.copy()
    # Exclude datetime and groupby columns from features
    exclude_cols = [self.datetime_col] + self.groupby_cols
    # Filter for numeric columns only
    numeric_feature_cols = [col for col in X.columns 
                           if col not in exclude_cols and is_numeric(col)]
```

#### Time Series Data Preparation
```python
def _prepare_time_series_data(self, X, y=None):
    """Organize data with datetime and groupby columns for time-based operations"""
    # Ensures proper datetime indexing and grouping for time series operations
```

### 4. Time-Based Operations Engine

#### Group-Aware Processing
```python
def _apply_time_based_operation(self, data, feature_col, operation, 
                               window_size=None, lag_period=None):
    """Apply operations with proper grouping and time-awareness"""
    if self.groupby_cols:
        # Process each group separately
        for name, group in data.groupby(self.groupby_cols):
            # Apply time-based operations within group
    else:
        # Single group operation
```

#### Flexible Time Period Parsing
```python
def _parse_time_periods(self, periods):
    """Parse time periods from various formats ('7D', '30D', '3M', etc.)"""
    # Supports string formats: '7D', '30D', '3M', '1Y'
    # Supports pandas Timedelta objects
    # Intelligent fallbacks for different formats
```

### 5. Enhanced Feature Generation

#### Time-Aware Feature Creation
The core `feat_with_depth()` method now supports time series context:

```python
def feat_with_depth(self, X, depth, op_ls, feat_ls):
    """Enhanced to handle datetime-aware time series operators"""
    # Original logic preserved
    if depth == 0:
        feat_ind = self.rng.choice(np.arange(len(self.ig_vector)), p=self.ig_vector)
        # NEW: Set context for time series operations
        if self.enable_time_series:
            self._current_feature_index = feat_ind
        return X[:, feat_ind]
    # Rest of method unchanged...
```

#### Fallback Mechanisms
Each time series operator includes intelligent fallbacks:

```python
def _safe_rolling_mean(self, feature_data):
    if self.enable_time_series and hasattr(self, '_current_data'):
        # Use time-based operations with datetime awareness
        return self._apply_time_based_operation(...)
    else:
        # Fallback to pandas rolling (original behavior)
        return pd.Series(feature_data).rolling(...).mean()
```

## Backward Compatibility

### Original Behavior Preserved
- **Default Settings**: `enable_time_series=False` maintains exact original behavior
- **Operator Set**: Original operators unchanged, time series operators only added when enabled
- **Method Signatures**: All original methods maintain identical signatures
- **Output Format**: Same output structure and data types

### Migration Path
```python
# Original usage (unchanged)
bf = BigFeat(task_type='classification')
features = bf.fit(X, y)

# Enhanced usage (new capabilities)
bf = BigFeat(task_type='classification', 
             enable_time_series=True,
             datetime_col='timestamp',
             window_sizes=['7D', '30D', '90D'])
features = bf.fit(X_with_datetime, y)
```

## New Capabilities

### 1. Time-Aware Feature Engineering
```python
# Supports DataFrames with datetime columns
df = pd.DataFrame({
    'timestamp': pd.date_range('2020-01-01', periods=1000, freq='D'),
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'group_id': np.random.choice(['A', 'B', 'C'], 1000)
})

bf = BigFeat(enable_time_series=True,
             datetime_col='timestamp',
             groupby_cols=['group_id'],
             window_sizes=['7D', '14D', '30D'],
             lag_periods=['1D', '7D', '14D'])

features = bf.fit(df, target)
```

### 2. Flexible Time Window Definitions
```python
# String formats
window_sizes = ['7D', '14D', '30D', '3M', '6M', '1Y']

# Pandas Timedelta objects
window_sizes = [pd.Timedelta(days=7), pd.Timedelta(days=30)]

# Mixed formats supported
```

### 3. Grouped Time Series Operations
- Automatically handles multiple time series within the same dataset
- Respects group boundaries when applying temporal operations
- Maintains proper temporal ordering within groups

### 4. Robust Error Handling
```python
def _clean_feature(self, feature_data):
    """Clean feature data to ensure stability"""
    # Replace inf with large finite values
    # Replace nan with zeros  
    # Clip extreme values
    # Type safety checks
```

## Implementation Details

### Memory Efficiency
- Time series data is processed incrementally where possible
- Original data is stored only when needed for datetime operations
- Efficient groupby operations using pandas native methods

### Performance Optimizations
- Lazy evaluation of time series operations
- Caching of group structures
- Vectorized operations where possible
- Intelligent fallbacks to avoid computation overhead

### Error Resilience
- All time series operations wrapped in try-catch blocks
- Graceful fallbacks to non-time-aware operations
- Data validation and cleaning at multiple stages
- Informative warning messages when operations fail

## Testing and Validation

### Backward Compatibility Tests
- All original test cases should pass unchanged
- Same random seed produces identical results when time series disabled
- Performance benchmarks maintained for non-time-series usage

### New Functionality Tests
- Time series operations with various window sizes
- Grouped time series handling
- DateTime column detection and processing
- Edge cases (missing data, irregular time series, etc.)

## Usage Examples

### Basic Time Series Enhancement
```python
import pandas as pd
from enhanced_bigfeat import BigFeat

# Prepare time series data
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=365),
    'sales': np.random.randn(365).cumsum(),
    'price': np.random.randn(365) + 100,
    'store_id': np.random.choice(['A', 'B', 'C'], 365)
})

# Create BigFeat with time series support
bf = BigFeat(
    task_type='regression',
    enable_time_series=True,
    datetime_col='date',
    groupby_cols=['store_id'],
    window_sizes=['7D', '30D', '90D'],
    verbose=True
)

# Generate features
features = bf.fit(df, target)
```

### Advanced Configuration
```python
# Custom time periods and operations
bf = BigFeat(
    enable_time_series=True,
    datetime_col='timestamp',
    window_sizes=[
        pd.Timedelta(days=7),
        pd.Timedelta(weeks=2),
        pd.Timedelta(days=90)
    ],
    lag_periods=['1D', '3D', '7D', '14D'],
    time_step='H'  # Hourly resampling
)
```

## Migration Guide

### For Existing Users
1. **No Changes Required**: Existing code continues to work unchanged
2. **Gradual Migration**: Add time series parameters incrementally
3. **Testing**: Verify results match expectations before deploying

### For New Time Series Projects
1. **DataFrame Input**: Use pandas DataFrames with datetime columns
2. **Column Specification**: Clearly specify datetime and groupby columns  
3. **Window Selection**: Choose appropriate time windows for your domain
4. **Validation**: Test with known time series patterns

## Conclusion

The enhanced BigFeat successfully extends the original library's capabilities while maintaining perfect backward compatibility. The time series enhancements provide powerful new feature engineering capabilities for temporal data while preserving all existing functionality for users who don't need time series features.

Key benefits:
- ✅ **100% Backward Compatible**: Existing code unchanged
- ✅ **Powerful Time Series Support**: 15 new temporal operators
- ✅ **Flexible Configuration**: Customizable windows and periods
- ✅ **Robust Implementation**: Error handling and fallbacks
- ✅ **Performance Optimized**: Efficient time series processing
- ✅ **Well Documented**: Clear usage patterns and examples