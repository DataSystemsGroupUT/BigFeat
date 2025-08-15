# BigFeat Rolling Mean: Complete Walkthrough

## Step-by-Step Example

Let's trace through exactly how BigFeat's rolling mean works from initialization to feature creation.

## Sample Dataset

```python
import pandas as pd
import numpy as np

# Create sample time series data
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
    'store_id': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
    'sales': [100, 120, 110, 130, 125, 80, 85, 90, 88, 92],
    'inventory': [500, 480, 490, 470, 475, 300, 285, 275, 277, 273]
})

print("Original Data:")
print(df)
```

**Output:**
```
    timestamp store_id  sales  inventory
0  2024-01-01        A    100        500
1  2024-01-02        A    120        480
2  2024-01-03        A    110        490
3  2024-01-04        A    130        470
4  2024-01-05        A    125        475
5  2024-01-06        B     80        300
6  2024-01-07        B     85        285
7  2024-01-08        B     90        275
8  2024-01-09        B     88        277
9  2024-01-10        B     92        273
```

## BigFeat Initialization

```python
from enhanced_bigfeat import BigFeat

bf = BigFeat(
    task_type='regression',
    enable_time_series=True,
    datetime_col='timestamp',
    groupby_cols=['store_id'],
    window_sizes=['2D', '3D'],  # 2-day and 3-day windows
    verbose=True
)
```

## What Happens During `fit()`

### **Step 1: Data Preparation**
```python
# BigFeat internally processes the DataFrame
self.original_data = df.copy()
self.feature_columns = ['sales', 'inventory']  # Excludes timestamp, store_id

# Feature matrix extracted:
X_features = [[100, 500], [120, 480], [110, 490], ...]  # Only numeric features
```

### **Step 2: Time Series Data Organization**
```python
# _prepare_time_series_data() organizes data by datetime and groups:
sorted_data = df.sort_values(['store_id', 'timestamp'])
```

### **Step 3: Feature Generation Loop**
During feature generation, BigFeat randomly selects operators. Let's say it selects `_safe_rolling_mean`:

```python
# Randomly selected: rolling mean operation on 'sales' feature (index 0)
self._current_feature_index = 0  # 'sales'
self._current_data = prepared_time_series_data

# Call rolling mean operator
result = self._safe_rolling_mean(X_scaled[:, 0])  # sales column
```

## Deep Dive: `_safe_rolling_mean` Execution

### **Step 1: Time Series Check**
```python
def _safe_rolling_mean(self, feature_data):
    if self.enable_time_series and hasattr(self, '_current_data'):
        # Use time-aware operations
        feature_col = 'sales'  # self.feature_columns[0]
        return self._apply_time_based_operation(
            self._current_data, 
            feature_col, 
            'rolling_mean'
        )
```

### **Step 2: Group-Based Processing**
```python
def _apply_time_based_operation(self, data, feature_col, operation):
    # Group by store_id
    results = []
    for store_id, group in data.groupby(['store_id']):
        group_result = self._apply_single_group_operation(
            group, feature_col, operation
        )
        results.extend(group_result)
    return np.array(results)
```

### **Step 3: Single Group Rolling Mean**

**For Store A:**
```python
# Store A data (sorted by timestamp):
store_a_data = {
    'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
    'sales': [100, 120, 110, 130, 125]
}

# Randomly selected window: '3D' (3-day window)
window_size = pd.Timedelta('3D')

# Rolling mean calculation with time-based window:
series = pd.Series([100, 120, 110, 130, 125])
series.index = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])

rolling_result = series.rolling(window='3D', min_periods=1).mean()
```

**Store A Rolling Mean Results:**
```
2024-01-01: 100.0      # Only 1 day available: (100)/1 = 100.0
2024-01-02: 110.0      # 2 days available: (100+120)/2 = 110.0  
2024-01-03: 110.0      # 3 days available: (100+120+110)/3 = 110.0
2024-01-04: 120.0      # 3-day window: (120+110+130)/3 = 120.0
2024-01-05: 121.67     # 3-day window: (110+130+125)/3 = 121.67
```

**For Store B:**
```python
# Store B data:
store_b_data = {
    'timestamp': ['2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10'],
    'sales': [80, 85, 90, 88, 92]
}

# Same 3D window applied:
series = pd.Series([80, 85, 90, 88, 92])
series.index = pd.to_datetime(['2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10'])

rolling_result = series.rolling(window='3D', min_periods=1).mean()
```

**Store B Rolling Mean Results:**
```
2024-01-06: 80.0       # Only 1 day: (80)/1 = 80.0
2024-01-07: 82.5       # 2 days: (80+85)/2 = 82.5
2024-01-08: 85.0       # 3 days: (80+85+90)/3 = 85.0  
2024-01-09: 87.67      # 3-day window: (85+90+88)/3 = 87.67
2024-01-10: 90.0       # 3-day window: (90+88+92)/3 = 90.0
```

## Final Feature Combination

### **Step 4: Combine Group Results**
```python
# Combined rolling mean feature for all rows:
combined_rolling_mean = [
    100.0,    # Store A, Day 1
    110.0,    # Store A, Day 2  
    110.0,    # Store A, Day 3
    120.0,    # Store A, Day 4
    121.67,   # Store A, Day 5
    80.0,     # Store B, Day 1
    82.5,     # Store B, Day 2
    85.0,     # Store B, Day 3
    87.67,    # Store B, Day 4
    90.0      # Store B, Day 5
]
```

### **Step 5: Feature Matrix Assembly**
```python
# This rolling mean becomes one column in the generated feature matrix:
gen_feats = np.array([
    [100.0, other_feature_1, other_feature_2, ...],  # Row 0
    [110.0, other_feature_1, other_feature_2, ...],  # Row 1
    [110.0, other_feature_1, other_feature_2, ...],  # Row 2
    # ... etc
])
```

## Key Points

### **1. Time-Aware Windows**
- Uses **actual time differences**, not just row positions
- `'3D'` means 3 calendar days, regardless of data frequency
- Handles irregular time series gracefully

### **2. Group Isolation**
- Store A's rolling mean **never uses Store B's data**
- Each entity maintains its own temporal patterns
- Prevents data leakage between groups

### **3. Window Boundaries**
```python
# For 3-day window on 2024-01-04:
# Looks back 3 days: 2024-01-01 to 2024-01-04
# Includes: [2024-01-02, 2024-01-03, 2024-01-04] values
# Rolling mean = (120 + 110 + 130) / 3 = 120.0
```

### **4. Fallback Behavior**
If time series fails or is disabled:
```python
else:
    # Fallback to simple pandas rolling
    window_size = self.rng.choice([3, 5, 7, 10])  # Row-based window
    result = pd.Series(feature_data).rolling(window=window_size).mean()
```

## Complete Example Output

**Original Data:**
```
sales:     [100, 120, 110, 130, 125,  80,  85,  90,  88,  92]
store_id:  [ A,   A,   A,   A,   A,   B,   B,   B,   B,   B ]
```

**Rolling Mean Feature (3D window):**
```
rolling_mean: [100, 110, 110, 120, 121.67, 80, 82.5, 85, 87.67, 90]
```

**Why This Works:**
- ✅ **Temporal accuracy**: Uses actual dates, not just positions
- ✅ **Group isolation**: Store A and B calculated separately  
- ✅ **Pattern preservation**: Each store's trend captured independently
- ✅ **No data leakage**: Future data never influences past calculations

This is how BigFeat transforms raw time series data into powerful temporal features while respecting entity boundaries and temporal order! 