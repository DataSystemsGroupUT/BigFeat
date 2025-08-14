# BigFeat Time Series Operators & Rolling Analysis

## Overview
BigFeat's time series enhancement provides automated feature generation for temporal data through sophisticated time-based rolling window operations, lag features, and time-aware transformations. This document explains the implementation, lifecycle, and usage of these time series operators.

## Architecture Overview
```
textInput Data (DataFrame with DateTime)
                ↓
   Data Preparation & Validation
                ↓
   Time Series Operators Applied
                ↓
   Feature Generation & Selection
                ↓
   Enhanced Feature Set Output
```

## Core Components
## 1. Time Series Operators
The BigFeat implementation includes 15 specialized time series operators:

### Rolling Window Operations

- `_safe_rolling_mean`: Time-based moving average calculations
- `_safe_rolling_std`: Time-based rolling standard deviation for volatility measures
- `_safe_rolling_min`: Time-based rolling minimum values
- `_safe_rolling_max`: Time-based rolling maximum values
- `_safe_rolling_median`: Time-based rolling median for robust central tendency
- `_safe_rolling_sum`: Time-based rolling sum aggregations

### Temporal Shift Operations

- `_safe_lag_feature`: Time-based lagged versions of features
- `_safe_diff_feature`: Time-based first and higher-order differences
- `_safe_pct_change`: Time-based percentage change calculations

### Advanced Time Series Features

- `_safe_ewm`: Time-based exponentially weighted moving averages
- `_safe_momentum`: Time-based momentum indicators (price - lagged_price)
- `_safe_seasonal_decompose`: Simple seasonal pattern extraction
- `_safe_trend_feature`: Simple trend as rolling slope
- `_safe_weekday_mean`: Mean value by weekday
- `_safe_month_mean`: Mean value by month

## 2. Data Preparation Infrastructure

```python
def _prepare_time_series_data(self, X, y=None):
    """
    Organizes data with datetime and groupby columns for proper time series operations
    """
    # Convert to DataFrame if needed
    # Add datetime column from stored original data
    # Add groupby columns for multi-series data
    # Sort by datetime and groupby columns
    return processed_dataframe
```

### Key Features:

- Automatic datetime column integration
- Support for multi-series data via groupby columns
- Proper temporal ordering ensures time series integrity
- Fallback handling for missing datetime information

## 3. Safe Operation Framework
Each time series operator follows a consistent safety pattern:
```python
def _safe_rolling_mean(self, feature_data):
    # Check if datetime-aware processing is available
    if self.enable_time_series and hasattr(self, '_current_data'):
        # Use datetime-aware operations with proper grouping
        return self._apply_time_based_operation(...)
    else:
        # Fallback to basic rolling operations
        try:
            window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
            window_size = min(window_size, len(feature_data))
            result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).mean().bfill().values
            return self._clean_feature(result)
        except Exception:
            return feature_data
```

## Implementation Lifecycle
### Phase 1: Initialization
```python
bigfeat = BigFeat(
    task_type='classification',
    enable_time_series=True,
    datetime_col='Date',
    groupby_cols=['Symbol', 'Store'],
    window_sizes=['7D', '14D', '30D', '3M', '6M', '1Y'],
    lag_periods=['1D', '7D', '30D'],
    verbose=True,
    time_step='D'
)
```
### Configuration Parameters:

- `enable_time_series`: Activates time series functionality
- `datetime_col`: Specifies the datetime column name
- `groupby_cols`: Columns for grouping multi-series data
- `window_sizes`: Time-based rolling window sizes (str or pd.Timedelta)
- `lag_periods`: Time-based lag periods (str or pd.Timedelta)
- `time_step`: Time step for resampling (e.g., 'D' for daily, 'H' for hourly)

### Phase 2: Data Processing
```python
# Data ingestion and preparation
self.original_data = X.copy()  # Store original DataFrame
self.feature_columns = [numeric_columns_only]  # Extract feature columns
self._current_data = self._prepare_time_series_data(X)  # Prepare for TS ops
```
### Data Flow:

1. **Original Data Storage**: Full DataFrame preserved for datetime/groupby access
2. **Feature Column Identification**: Automatic detection of numeric feature columns
3. **Data Preparation**: Sorting and organizing for time series operations
4. **Validation**: Ensure datetime columns and groupby integrity

### Phase 3: Feature Generation
```python
def feat_with_depth(self, X, depth, op_ls, feat_ls):
    """ Recursively generate a new features - Enhanced to handle datetime-aware time series operators """
    if depth == 0:
        feat_ind = self.rng.choice(np.arange(len(self.ig_vector)), p=self.ig_vector)
        feat_ls.append(feat_ind)
        # Set current feature index for time series operations
        if self.enable_time_series:
            self._current_feature_index = feat_ind
        return X[:, feat_ind]

    depth -= 1
    op = self.rng.choice(self.operators, p=self.operator_weights)

    if op in self.binary_operators:
        feat_1 = self.feat_with_depth(X, depth, op_ls, feat_ls)
        feat_2 = self.feat_with_depth(X, depth, op_ls, feat_ls)
        op_ls.append((op, depth))
        result = op(feat_1, feat_2)
        return self._clean_feature(result) if self.enable_time_series else result

    elif op in self.unary_operators:
        feat_1 = self.feat_with_depth(X, depth, op_ls, feat_ls)
        op_ls.append((op, depth))
        result = op(feat_1)
        return self._clean_feature(result) if self.enable_time_series else result
```

### Generation Process:

1. **Operator Selection**: Weighted random selection including TS operators
2. **Context Setting**: Current feature index set for TS operations
3. **Safe Application**: Each operator includes error handling and fallbacks
4. **Feature Cleaning**: Automatic handling of NaN, infinity, and extreme values

### Phase 4: Time Series Operation Execution
```python
def _apply_time_based_operation(self, data, feature_col, operation, window_size=None, lag_period=None):
    """
    Apply time-based series operation to a specific feature column with proper grouping
    """
    try:
        if feature_col not in data.columns or self.datetime_col not in data.columns:
            return np.zeros(len(data))

        # Set datetime as index for time-based operations
        if self.groupby_cols and any(col in data.columns for col in self.groupby_cols):
            # Group data by groupby columns
            groupby_cols = [col for col in self.groupby_cols if col in data.columns]

            results = []
            for name, group in data.groupby(groupby_cols):
                group_sorted = group.set_index(self.datetime_col).sort_index()
                group_result = self._apply_single_group_operation(
                    group_sorted, feature_col, operation, window_size, lag_period
                )
                # Restore original order
                group_result = group_result.reindex(group[self.datetime_col]).values
                results.extend(group_result)

            return np.array(results)
        else:
            # Single group operation
            data_sorted = data.set_index(self.datetime_col).sort_index()
            result = self._apply_single_group_operation(
                data_sorted, feature_col, operation, window_size, lag_period
            )
            # Restore original order
            return result.reindex(data[self.datetime_col]).values

    except Exception as e:
        if self.verbose:
            print(f"Warning: Time-based operation {operation} failed for {feature_col}: {str(e)}")
        return np.zeros(len(data))
```

### Operation Features:

- **Groupby Support**: Proper handling of multi-series data
- **Parameter Management**: Dynamic window sizes and lag periods
- **Robust Execution**: Comprehensive error handling
- **Data Cleaning**: Automatic post-processing of results

## Rolling Analysis Implementation
### Window Size Strategy
```python
# Default window sizes optimized for different data types
financial_windows = ['3D', '7D', '14D', '30D']    # Short-term trading patterns
retail_windows = ['7D', '14D', '21D', '30D']         # Weekly/monthly cycles
general_windows = ['3D', '7D', '14D', '21D', '30D', '60D', '90D']  # Comprehensive coverage
```

### Selection Logic:

- **Dynamic Selection**: Random selection from configured ranges
- **Minimum Periods**: Always set to 1 to avoid NaN proliferation
- **Adaptive Sizing**: Window size limited by data length

### Groupby Mechanics
```python
# Multi-series handling example
# Data: [Date, Symbol, Price, Volume, ...]
# Groupby: ['Symbol']
# Result: Rolling operations applied separately per symbol

grouped = data.groupby(['Symbol'])['Price']
rolling_mean = grouped.rolling(window=pd.Timedelta(days=10), min_periods=1).mean()
```

### Benefits:

- **Series Isolation**: Each time series processed independently
- **Temporal Integrity**: No cross-contamination between different series
- **Scalability**: Efficient processing of large multi-series datasets

### Data Cleaning Pipeline
```python
def _clean_feature(self, feature_data):
    """Clean feature data to ensure stability"""
    try:
        feature_data = np.asarray(feature_data, dtype=float)
        # Replace inf with large finite values
        feature_data = np.where(np.isinf(feature_data), np.sign(feature_data) * 1e8, feature_data)
        # Replace nan with zeros
        feature_data = np.where(np.isnan(feature_data), 0, feature_data)
        # Clip extreme values
        feature_data = np.clip(feature_data, -1e8, 1e8)
        return feature_data
    except Exception:
        return np.zeros_like(feature_data, dtype=float)
```

### Cleaning Steps:

- **Type Conversion**: Ensure numeric data types
- **Infinity Handling**: Replace with large finite values
- **NaN Replacement**: Fill with zeros (preserves array shape)
- **Extreme Value Clipping**: Prevent numerical instability

## Feature Validation
### Quality Checks
```python
def _validate_feature(self, feature_data):
    """Validate features for stability and usefulness"""
    try:
        if len(feature_data) == 0:
            return False
        feature_data = np.asarray(feature_data, dtype=float)
        if not np.isfinite(feature_data).all():
            return False
        if np.std(feature_data) < 1e-10:
            return False
        if np.max(np.abs(feature_data)) > 1e8:
            return False
        return True
    except Exception:
        return False
```

 ### Validation Criteria:

- **Non-empty**: Features must contain data
- **Finite Values**: No NaN or infinite values after cleaning
- **Sufficient Variance**: Avoid constant or near-constant features
- **Reasonable Magnitude**: Prevent numerical overflow issues

## Usage Examples
### Basic Time Series Setup
```python
# Initialize with time series support
bigfeat = BigFeat(
    task_type='classification',
    enable_time_series=True,
    datetime_col='Date',
    groupby_cols=['Symbol'],
    window_sizes=['7D', '14D', '30D', '3M', '6M', '1Y'],
    lag_periods=['1D', '7D', '30D'],
    time_step='D'
)

# Fit with DataFrame including datetime column
X_enhanced = bigfeat.fit(df, target_column)
```

### Multi-Series Configuration
```python
# Multiple time series (e.g., stock data)
bigfeat = BigFeat(
    enable_time_series=True,
    datetime_col='Date',
    groupby_cols=['Symbol'],  # Separate processing per stock
    window_sizes=['3D', '7D', '14D', '21D'],
    lag_periods=['1D', '3D', '7D'],
    time_step='D'
)
```
### Transform New Data

```python
# Apply same transformations to new data
X_test_enhanced = bigfeat.transform(test_df)
```

## Performance Considerations
### Computational Complexity

- **Rolling Operations**: O(n × w) where n is data length, w is window size (time-based)
- **Groupby Operations**: O(n × g) where g is number of groups
- **Feature Generation**: O(f × d × o) where f is features, d is depth, o is operations

### Memory Management

- **Original Data Storage**: Full DataFrame kept for datetime access
- **Processed Data Caching**: Temporary storage during generation
- **Result Cleaning**: Immediate cleanup of intermediate results

### Optimization Strategies

- **Lazy Evaluation**: Operations only computed when needed
- **Vectorized Operations**: Pandas/NumPy optimizations utilized
- **Memory Cleanup**: Intermediate results freed promptly
- **Efficient Grouping**: Optimized groupby operations

## Error Handling
### Robust Fallbacks
```python
try:
    # Attempt datetime-aware operation
    result = self._apply_time_based_operation(...)
except Exception as e:
    if self.verbose:
        print(f"Warning: Time-based operation failed: {str(e)}")
    # Fallback to basic operation or zeros
    return self._safe_fallback_operation(feature_data)
```

### Common Error Scenarios

- **Missing DateTime Column**: Graceful degradation to basic operations
- **Insufficient Data**: Minimum periods handling prevents NaN proliferation
- **Type Mismatches**: Automatic type conversion and validation
- **Memory Issues**: Chunked processing for large datasets

## Best Practices
### Data Preparation

- **Sort Data**: Ensure proper temporal ordering before processing
- **Handle Missing Values**: Clean data before applying time series operations
- **Validate DateTime**: Ensure datetime column is properly formatted
- **Check Groupby Columns**: Verify grouping variables are meaningful

### Configuration Tuning

- **Window Sizes**: Match to data frequency and patterns (e.g., '7D' for weekly cycles)
- **Lag Periods**: Consider the prediction horizon (e.g., '1D' for next-day predictions)
- **Operator Selection**: Balance comprehensive coverage with computational cost
- **Groupby Strategy**: Group by meaningful time series identifiers

### Performance Optimization

- **Feature Selection**: Use stability selection to choose best time series features
- **Correlation Checking**: Remove highly correlated rolling features
- **Memory Monitoring**: Consider chunked processing for very large datasets
- **Validation Strategy**: Use time series cross-validation techniques

This comprehensive time series framework enables BigFeat to automatically discover and generate meaningful temporal features while maintaining robustness and performance across diverse datasets.