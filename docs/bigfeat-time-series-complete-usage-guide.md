# BigFeat Time Series Operations: Complete Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Basic Setup](#basic-setup)
3. [Data Preparation](#data-preparation)
4. [Configuration Examples](#configuration-examples)
5. [Single vs Multi-Series Data](#single-vs-multi-series-data)
6. [Advanced Usage Patterns](#advanced-usage-patterns)
7. [Troubleshooting](#troubleshooting)
8. [Real-World Examples](#real-world-examples)
9. [Performance Optimization](#performance-optimization)
10. [Best Practices](#best-practices)

## Quick Start

### Minimal Example

```python
from bigfeat.bigfeat import BigFeat  # Updated import
import pandas as pd

# Your time series data
df = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=100),
    'value1': np.random.randn(100).cumsum(),
    'value2': np.random.randn(100),
    'target': np.random.randint(0, 2, 100)
})

# Initialize BigFeat with time series support
bigfeat = BigFeat(
    task_type='classification',
    enable_time_series=True,
    datetime_col='Date',
    time_step='D'
)

# Generate enhanced features
X_enhanced = bigfeat.fit(df, df['target'])

# Apply to new data
X_new_enhanced = bigfeat.transform(new_df)
```

## Basic Setup

### 1. Import Required Libraries

```python
import pandas as pd
import numpy as np
from bigfeat.bigfeat import BigFeat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
```

### 2. Initialize BigFeat with Time Series

```python
# Basic time series configuration
bigfeat = BigFeat(
    task_type='classification',          # or 'regression'
    enable_time_series=True,             # Enable TS features
    datetime_col='Date',                 # Name of datetime column
    window_sizes=['3D', '7D', '14D', '30D'],        # Rolling window options
    lag_periods=['1D', '3D', '7D', '14D'],          # Lag options
    time_step='D',                       # Time step for resampling
    verbose=True                         # Show progress
)
```

### 3. Key Parameters Explained

| Parameter | Purpose | Example Values |
|-----------|---------|----------------|
| `enable_time_series` | Activates time series operators | `True`/`False` |
| `datetime_col` | Name of datetime column | `'Date'`, `'timestamp'`, `'time'` |
| `groupby_cols` | Columns for multi-series grouping | `['Symbol']`, `['store_id', 'product']` |
| `window_sizes` | Time-based rolling window options (str or pd.Timedelta) | `['3D', '7D', '14D', '30D']` |
| `lag_periods` | Time-based lag period options (str or pd.Timedelta) | `['1D', '3D', '7D', '14D']` |
| `time_step` | Time step for resampling | `'D'`, `'H'`, `'W'`, `'M'` |
| `verbose` | Print progress | `True`/`False` |

## Data Preparation

### Required Data Format
Your DataFrame must include:

- **DateTime column**: Properly formatted datetime data
- **Feature columns**: Numeric columns for feature generation
- **Target column**: What you want to predict (separate from DataFrame)

### Example Data Structures

#### Stock Market Data

```python
df = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=252),
    'Symbol': ['AAPL'] * 252,
    'Open': np.random.uniform(150, 200, 252),
    'High': np.random.uniform(150, 200, 252),
    'Low': np.random.uniform(150, 200, 252),
    'Close': np.random.uniform(150, 200, 252),
    'Volume': np.random.uniform(1e6, 1e8, 252),
    'target': np.random.randint(0, 2, 252)  # Price up/down
})

# Ensure datetime column is properly formatted
df['Date'] = pd.to_datetime(df['Date'])

# Sort by time (crucial for time series)
df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
```

#### Retail Sales Data

```python
df = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=365),
    'Store': np.random.choice(['A', 'B', 'C'], 365),
    'Sales': np.random.uniform(1000, 5000, 365),
    'Customers': np.random.uniform(50, 200, 365),
    'Temperature': np.random.uniform(0, 35, 365),
    'IsWeekend': np.random.choice([0, 1], 365),
    'target': np.random.uniform(2000, 6000, 365)  # Next day sales
})

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
```

#### IoT Sensor Data

```python
df = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=8760, freq='h'),
    'sensor_id': np.random.choice(['S1', 'S2', 'S3'], 8760),
    'temperature': np.random.uniform(20, 80, 8760),
    'humidity': np.random.uniform(30, 90, 8760),
    'pressure': np.random.uniform(980, 1020, 8760),
    'target': np.random.randint(0, 2, 8760)  # Anomaly detection
})

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['sensor_id', 'timestamp']).reset_index(drop=True)
```

## Configuration Examples

### 1. Financial Markets

```python
# High-frequency trading
bigfeat = BigFeat(
    task_type='classification',
    enable_time_series=True,
    datetime_col='timestamp',
    groupby_cols=['symbol'],
    window_sizes=['3D', '7D', '14D', '30D'],        # 3 days to 1 month
    lag_periods=['1D', '3D', '7D', '14D'],           # 1 day to 2 weeks
    time_step='D',
    verbose=True
)

# Daily stock analysis
bigfeat = BigFeat(
    task_type='regression',
    enable_time_series=True,
    datetime_col='Date',
    groupby_cols=['Symbol'],
    window_sizes=['7D', '14D', '30D', '90D'],        # 1 week to 3 months
    lag_periods=['1D', '7D', '14D', '30D'],          # 1 day to 1 month
    time_step='D',
    verbose=True
)
```

### 2. Business Analytics

```python
# Retail sales forecasting
bigfeat = BigFeat(
    task_type='regression',
    enable_time_series=True,
    datetime_col='Date',
    groupby_cols=['Store', 'Product'],
    window_sizes=['7D', '14D', '30D', '90D'],        # 1 week to 3 months
    lag_periods=['1D', '7D', '14D', '30D'],          # 1 day to 1 month
    time_step='D',
    verbose=True
)

# Website analytics
bigfeat = BigFeat(
    task_type='classification',
    enable_time_series=True,
    datetime_col='date',
    window_sizes=['7D', '14D', '28D'],            # 1-4 weeks
    lag_periods=['1D', '7D', '14D'],              # Recent history
    time_step='D',
    verbose=True
)
```

### 3. Industrial IoT

```python
# Manufacturing equipment monitoring
bigfeat = BigFeat(
    task_type='classification',
    enable_time_series=True,
    datetime_col='timestamp',
    groupby_cols=['machine_id', 'line'],
    window_sizes=['1H', '6H', '12H', '24H'],          # 1 hour to 1 day
    lag_periods=['1H', '6H', '12H'],              # 1 hour to 12 hours
    time_step='H',
    verbose=True
)

# Environmental monitoring
bigfeat = BigFeat(
    task_type='regression',
    enable_time_series=True,
    datetime_col='datetime',
    groupby_cols=['station_id'],
    window_sizes=['24H', '72H', '7D'],          # 1 day to 1 week (hourly data)
    lag_periods=['1H', '12H', '24H', '72H'],         # 1 hour to 3 days
    time_step='H',
    verbose=True
)
```

## Single vs Multi-Series Data

### Single Time Series (No Grouping)

```python
# Simple temperature forecasting
df = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=365),
    'temperature': 20 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365),
    'humidity': np.random.uniform(40, 80, 365),
    'pressure': np.random.uniform(990, 1010, 365)
})

# No groupby columns needed
bigfeat = BigFeat(
    task_type='regression',
    enable_time_series=True,
    datetime_col='Date',
    # groupby_cols=[]  # Empty or omit entirely
    window_sizes=['7D', '14D', '30D'],
    lag_periods=['1D', '7D', '14D'],
    time_step='D'
)

# Target: predict next day temperature
target = df['temperature'].shift(-1).fillna(method='ffill')
X_enhanced = bigfeat.fit(df, target)
```

### Multi-Series Data (With Grouping)

```python
# Multiple stock symbols
df = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=500).repeat(3),
    'Symbol': ['AAPL', 'GOOGL', 'MSFT'] * 500,
    'Price': np.random.uniform(100, 300, 1500),
    'Volume': np.random.uniform(1e6, 1e8, 1500)
})

# Group by Symbol to prevent cross-contamination
bigfeat = BigFeat(
    task_type='classification',
    enable_time_series=True,
    datetime_col='Date',
    groupby_cols=['Symbol'],  # Critical for multi-series
    window_sizes=['3D', '7D', '14D', '21D'],
    lag_periods=['1D', '3D', '7D'],
    time_step='D'
)

target = (df['Price'].shift(-1) > df['Price']).astype(int)
X_enhanced = bigfeat.fit(df, target)
```

## Advanced Usage Patterns

### 1. Custom Window and Lag Configurations

```python
# Adaptive configuration based on data frequency
def get_adaptive_config(data_freq, data_length):
    if data_freq == 'D':  # Daily
        windows = ['3D', '7D', '14D', '30D']
        lags = ['1D', '3D', '7D', '14D']
    elif data_freq == 'H':  # Hourly
        windows = ['1H', '3H', '6H', '12H', '1D']
        lags = ['1H', '3H', '6H', '12H']
    elif data_freq == 'W':  # Weekly
        windows = ['1W', '2W', '4W', '12W']
        lags = ['1W', '2W', '4W']
    
    # Adjust for data length
    max_window = data_length // 10
    windows = [w for w in windows if w <= max_window]
    
    return windows, lags

# Apply adaptive configuration
windows, lags = get_adaptive_config('D', len(df))
bigfeat = BigFeat(
    enable_time_series=True,
    datetime_col='Date',
    window_sizes=windows,
    lag_periods=lags,
    time_step='D'
)
```

### 2. Feature Generation Control

```python
# More aggressive feature generation
bigfeat = BigFeat(
    task_type='regression',
    enable_time_series=True,
    datetime_col='Date',
    groupby_cols=['ID'],
    window_sizes=['3D', '7D', '14D', '30D', '60D'],
    lag_periods=['1D', '3D', '7D', '14D', '30D'],
    time_step='D',
    verbose=True
)

# Generate more features with more iterations
X_enhanced = bigfeat.fit(
    df, target,
    gen_size=10,           # Generate 10 features per iteration
    iterations=5,          # Run 5 iterations
    estimator='avg',       # Use ensemble estimator
    selection='stability'   # Use stability selection
)
```

### 3. Incremental Processing for Large Datasets

```python
# Process data in chunks for memory efficiency
def process_large_dataset(df, target, chunk_size=10000):
    # Initial fit on first chunk
    first_chunk = df.iloc[:chunk_size]
    first_target = target.iloc[:chunk_size]
    
    bigfeat = BigFeat(
        enable_time_series=True,
        datetime_col='Date',
        groupby_cols=['ID'],
        verbose=False
    )
    
    X_enhanced = bigfeat.fit(first_chunk, first_target)
    
    # Transform remaining chunks
    results = [X_enhanced]
    for i in range(chunk_size, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        X_chunk = bigfeat.transform(chunk)
        results.append(X_chunk)
    
    return np.vstack(results)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "No numeric feature columns found"

```python
# Problem: DataFrame contains only datetime/categorical columns
df = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=100),
    'Category': ['A', 'B'] * 50,
    'Status': ['Active', 'Inactive'] * 50
})

# Solution: Create numeric features first
df['Category_encoded'] = pd.factorize(df['Category'])[0]
df['Status_encoded'] = pd.factorize(df['Status'])[0]
# Now BigFeat can work with Category_encoded and Status_encoded
```

#### 2. "Shape mismatch" errors

```python
# Problem: Inconsistent data shapes during transform
# Solution: Ensure consistent column structure

# During fit
train_df = df[['Date', 'feature1', 'feature2', 'feature3']]
bigfeat.fit(train_df, target)

# During transform - use same columns
test_df = test_df[['Date', 'feature1', 'feature2', 'feature3']]
X_enhanced = bigfeat.transform(test_df)
```

#### 3. Poor time series performance

```python
# Problem: Data not properly sorted by time
df = df.sample(frac=1)  # Random shuffle - BAD!

# Solution: Always sort by datetime (and groupby columns)
df = df.sort_values(['GroupCol', 'Date']).reset_index(drop=True)

# Problem: Window sizes too large for dataset
bigfeat = BigFeat(window_sizes=['100D', '200D'])  # Bad for 300-row dataset

# Solution: Scale windows to data size
max_window = len(df) // 10
window_sizes = [w for w in ['3D', '7D', '14D', '30D'] if w <= max_window]
```

#### 4. Memory issues with large datasets

```python
# Problem: Out of memory with large time series
# Solution: Reduce parameters or process in chunks

# Lighter configuration
bigfeat = BigFeat(
    enable_time_series=True,
    datetime_col='Date',
    window_sizes=['3D', '7D'],        # Fewer options
    lag_periods=['1D', '3D'],         # Fewer options
    gen_size=3,                 # Fewer generated features
    iterations=2                # Fewer iterations
)
```

## Real-World Examples

### Example 1: Stock Price Direction Prediction

```python
import yfinance as yf

# Download stock data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="2y")
df = df.reset_index()

# Add technical features
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(20).std()
df['Volume_MA'] = df['Volume'].rolling(20).mean()

# Create target: next day price direction
df['Next_Return'] = df['Returns'].shift(-1)
df['Price_Up'] = (df['Next_Return'] > 0).astype(int)

# Remove missing values
df = df.dropna()

# Configure BigFeat for stock analysis
bigfeat = BigFeat(
    task_type='classification',
    enable_time_series=True,
    datetime_col='Date',
    window_sizes=['3D', '7D', '14D', '30D'],     # 3 days to 1 month
    lag_periods=['1D', '3D', '7D'],           # 1-7 days back
    time_step='D',
    verbose=True
)

# Features to use (exclude target and date)
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Volatility', 'Volume_MA']
X = df[['Date'] + feature_cols]
y = df['Price_Up']

# Time series split
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Generate enhanced features
X_train_enhanced = bigfeat.fit(X_train, y_train, 
                              gen_size=5, 
                              iterations=3,
                              estimator='rf')

X_test_enhanced = bigfeat.transform(X_test)

# Train final model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_enhanced, y_train)

# Evaluate
predictions = model.predict(X_test_enhanced)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy with time series features: {accuracy:.4f}")
```

### Example 2: Retail Sales Forecasting

```python
# Create synthetic retail data
np.random.seed(42)
dates = pd.date_range('2022-01-01', periods=730, freq='D')
stores = ['Store_A', 'Store_B', 'Store_C']

data = []
for store in stores:
    for i, date in enumerate(dates):
        # Base sales with trends and seasonality
        base_sales = 1000 + i * 2  # Growing trend
        seasonal = 500 * np.sin(2 * np.pi * date.dayofyear / 365)  # Yearly
        weekly = 200 * np.sin(2 * np.pi * date.weekday / 7)  # Weekly
        weekend_boost = 300 if date.weekday >= 5 else 0
        noise = np.random.normal(0, 100)
        
        sales = base_sales + seasonal + weekly + weekend_boost + noise
        
        data.append({
            'Date': date,
            'Store': store,
            'Sales': sales,
            'DayOfWeek': date.weekday,
            'Month': date.month,
            'IsWeekend': int(date.weekday >= 5)
        })

df = pd.DataFrame(data)

# Create target: next day sales
df['NextDaySales'] = df.groupby('Store')['Sales'].shift(-1)
df = df.dropna()

# Configure for retail forecasting
bigfeat = BigFeat(
    task_type='regression',
    enable_time_series=True,
    datetime_col='Date',
    groupby_cols=['Store'],       # Separate analysis per store
    window_sizes=['7D', '14D', '30D'],     # 1 week to 1 month
    lag_periods=['1D', '7D', '14D'],       # 1 day, 1 week, 2 weeks
    time_step='D',
    verbose=True
)

# Feature columns
feature_cols = ['Sales', 'DayOfWeek', 'Month', 'IsWeekend']
X = df[['Date', 'Store'] + feature_cols]
y = df['NextDaySales']

# Time split (80% train, 20% test)
split_date = df['Date'].quantile(0.8)
train_mask = df['Date'] <= split_date
test_mask = df['Date'] > split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# Generate features
X_train_enhanced = bigfeat.fit(X_train, y_train,
                              gen_size=6,
                              iterations=4,
                              estimator='avg')

X_test_enhanced = bigfeat.transform(X_test)

# Train and evaluate
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_enhanced, y_train)

predictions = model.predict(X_test_enhanced)
r2 = r2_score(y_test, predictions)
mae = np.mean(np.abs(y_test - predictions))

print(f"R² Score with time series features: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")
```

### Example 3: IoT Anomaly Detection

```python
# Create synthetic sensor data
np.random.seed(42)
timestamps = pd.date_range('2023-01-01', periods=8760, freq='H')
sensors = ['Sensor_1', 'Sensor_2', 'Sensor_3']

data = []
for sensor in sensors:
    for i, timestamp in enumerate(timestamps):
        # Normal operating patterns
        temp = 25 + 10 * np.sin(2 * np.pi * timestamp.hour / 24)  # Daily cycle
        temp += 5 * np.sin(2 * np.pi * timestamp.dayofyear / 365)  # Seasonal
        temp += np.random.normal(0, 1)  # Noise
        
        # Occasional anomalies
        is_anomaly = np.random.random() < 0.02  # 2% anomaly rate
        if is_anomaly:
            temp += np.random.normal(0, 10)  # Large deviation
        
        humidity = 50 + 20 * np.sin(2 * np.pi * timestamp.hour / 24 + np.pi/4)
        humidity += np.random.normal(0, 2)
        
        data.append({
            'timestamp': timestamp,
            'sensor_id': sensor,
            'temperature': temp,
            'humidity': humidity,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'anomaly': int(is_anomaly)
        })

df = pd.DataFrame(data)

# Configure for anomaly detection
bigfeat = BigFeat(
    task_type='classification',
    enable_time_series=True,
    datetime_col='timestamp',
    groupby_cols=['sensor_id'],
    window_sizes=['1H', '6H', '12H', '24H'],     # 1h to 1 day
    lag_periods=['1H', '6H', '12H'],       # 1h to 12h
    time_step='H',
    verbose=True
)

# Features (exclude target)
feature_cols = ['temperature', 'humidity', 'hour', 'day_of_week']
X = df[['timestamp', 'sensor_id'] + feature_cols]
y = df['anomaly']

# Split data
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Generate features
X_train_enhanced = bigfeat.fit(X_train, y_train,
                              gen_size=8,
                              iterations=3)

X_test_enhanced = bigfeat.transform(X_test)

# Train anomaly detector
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Use enhanced features for better anomaly detection
detector = IsolationForest(contamination=0.02, random_state=42)
detector.fit(X_train_enhanced[y_train == 0])  # Train on normal data only

# Predict anomalies
anomaly_predictions = detector.predict(X_test_enhanced)
anomaly_predictions = (anomaly_predictions == -1).astype(int)

print("Anomaly Detection Results:")
print(classification_report(y_test, anomaly_predictions))
```

## Performance Optimization

### 1. Configuration Tuning

```python
# Light configuration for development/testing
config_light = {
    'window_sizes': ['3D', '7D'],
    'lag_periods': ['1D', '3D'],
    'gen_size': 3,
    'iterations': 2
}

# Heavy configuration for production
config_heavy = {
    'window_sizes': ['3D', '7D', '14D', '30D', '60D'],
    'lag_periods': ['1D', '3D', '7D', '14D', '30D'],
    'gen_size': 10,
    'iterations': 5
}

# Auto-scaling based on data size
def get_scaled_config(data_length):
    if data_length < 500:
        return config_light
    elif data_length < 5000:
        return {
            'window_sizes': ['3D', '7D', '14D'],
            'lag_periods': ['1D', '3D', '7D'],
            'gen_size': 5,
            'iterations': 3
        }
    else:
        return config_heavy

# Auto-scaling based on data size
max_window = len(df) // 10
window_sizes = [w for w in ['3D', '7D', '14D', '30D'] if w <= max_window]
```

### 2. Memory Management

```python
# Monitor memory usage
import psutil
import gc

def check_memory():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory before: {check_memory():.1f} MB")

# Process with memory cleanup
X_enhanced = bigfeat.fit(X_train, y_train)
gc.collect()  # Force garbage collection

print(f"Memory after: {check_memory():.1f} MB")
```

### 3. Parallel Processing

```python
# Use multiple cores for feature generation
bigfeat = BigFeat(
    enable_time_series=True,
    datetime_col='Date',
    n_jobs=-1,  # Use all available cores
    verbose=True
)
```

## Best Practices

### 1. Data Quality

```python
# Always validate your data before processing
def validate_time_series_data(df, datetime_col, feature_cols):
    """Validate data quality for time series processing"""
    
    # Check datetime column
    assert datetime_col in df.columns, f"Datetime column '{datetime_col}' not found"
    assert pd.api.types.is_datetime64_any_dtype(df[datetime_col]), "Datetime column must be datetime type"
    
    # Check for missing values
    missing = df[feature_cols].isnull().sum()
    if missing.any():
        print(f"Warning: Missing values found:\n{missing[missing > 0]}")
    
    # Check for infinite values
    infinite = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum()
    if infinite.any():
        print(f"Warning: Infinite values found:\n{infinite[infinite > 0]}")
    
    # Check temporal ordering
    if not df[datetime_col].is_monotonic_increasing:
        print("Warning: Data is not sorted by datetime")
    
    print("Data validation complete")

# Use validation
validate_time_series_data(df, 'Date', ['feature1', 'feature2'])
```

### 2. Feature Engineering Pipeline

```python
def create_time_series_pipeline(df, datetime_col, feature_cols, target_col, 
                               groupby_cols=None, test_size=0.2):
    """Complete pipeline for time series feature engineering"""
    
    # 1. Data validation
    validate_time_series_data(df, datetime_col, feature_cols)
    
    # 2. Sort data
    sort_cols = (groupby_cols or []) + [datetime_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    
    # 3. Create time-based split
    split_date = df[datetime_col].quantile(0.8)
    train_mask = df[datetime_col] <= split_date
    test_mask = df[datetime_col] > split_date
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    # 4. Configure BigFeat
    bigfeat = BigFeat(
        task_type='classification' if df[target_col].dtype == 'int' else 'regression',
        enable_time_series=True,
        datetime_col=datetime_col,
        groupby_cols=groupby_cols,
        verbose=True
    )
    
    # 5. Prepare feature data
    X_train = train_df[[datetime_col] + (groupby_cols or []) + feature_cols]
    X_test = test_df[[datetime_col] + (groupby_cols or []) + feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    # 6. Generate features
    X_train_enhanced = bigfeat.fit(X_train, y_train)
    X_test_enhanced = bigfeat.transform(X_test)
    
    return X_train_enhanced, X_test_enhanced, y_train, y_test, bigfeat

# Use pipeline
X_train, X_test, y_train, y_test, bigfeat = create_time_series_pipeline(
    df, 'Date', ['feature1', 'feature2'], 'target', 
    groupby_cols=['group']
)
```

### 3. Model Selection and Validation

```python
def evaluate_time_series_features(df, datetime_col, feature_cols, target_col, 
                                 groupby_cols=None):
    """Compare performance with and without time series features"""
    
    # Prepare data
    X_train, X_test, y_train, y_test, bigfeat = create_time_series_pipeline(
        df, datetime_col, feature_cols, target_col, groupby_cols
    )
    
    # Baseline model (no time series)
    bigfeat_baseline = BigFeat(
        task_type='classification' if df[target_col].dtype == 'int' else 'regression',
        enable_time_series=False
    )
    
    # Time-based split for baseline
    split_date = df[datetime_col].quantile(0.8)
    train_mask = df[datetime_col] <= split_date
    test_mask = df[datetime_col] > split_date
    
    X_train_baseline = bigfeat_baseline.fit(
        df[train_mask][feature_cols], y_train
    )
    X_test_baseline = bigfeat_baseline.transform(
        df[test_mask][feature_cols]
    )
    
    # Compare models
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    if df[target_col].dtype == 'int':  # Classification
        model = RandomForestClassifier(random_state=42)
        metric = accuracy_score
        metric_name = "Accuracy"
    else:  # Regression
        model = RandomForestRegressor(random_state=42)
        metric = r2_score
        metric_name = "R² Score"
    
    # Baseline performance
    model.fit(X_train_baseline, y_train)
    baseline_pred = model.predict(X_test_baseline)
    baseline_score = metric(y_test, baseline_pred)
    
    # Time series enhanced performance
    model.fit(X_train, y_train)
    enhanced_pred = model.predict(X_test)
    enhanced_score = metric(y_test, enhanced_pred)
    
    # Results
    improvement = enhanced_score - baseline_score
    improvement_pct = (improvement / abs(baseline_score)) * 100 if baseline_score != 0 else 0
    
    print(f"\n=== Time Series Feature Evaluation ===")
    print(f"Baseline {metric_name}: {baseline_score:.4f}")
    print(f"Enhanced {metric_name}: {enhanced_score:.4f}")
    print(f"Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
    print(f"Features: {X_train_baseline.shape[1]} → {X_train.shape[1]}")
    
    return {
        'baseline_score': baseline_score,
        'enhanced_score': enhanced_score,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'baseline_features': X_train_baseline.shape[1],
        'enhanced_features': X_train.shape[1]
    }

# Use evaluation
results = evaluate_time_series_features(df, 'Date', ['feature1', 'feature2'], 'target')
```

### 4. Production Deployment

```python
def deploy_time_series_model(bigfeat, model, feature_cols, datetime_col, 
                           groupby_cols=None):
    """Create production-ready prediction function"""
    
    def predict_new_data(new_df):
        """Predict on new time series data"""
        
        # Validate input
        required_cols = [datetime_col] + (groupby_cols or []) + feature_cols
        missing_cols = set(required_cols) - set(new_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Prepare data
        X_new = new_df[required_cols].copy()
        X_new[datetime_col] = pd.to_datetime(X_new[datetime_col])
        
        # Sort data
        sort_cols = (groupby_cols or []) + [datetime_col]
        X_new = X_new.sort_values(sort_cols).reset_index(drop=True)
        
        # Generate features
        try:
            X_enhanced = bigfeat.transform(X_new)
            predictions = model.predict(X_enhanced)
            
            # Add predictions to original dataframe
            result_df = new_df.copy()
            result_df['prediction'] = predictions
            result_df['prediction_timestamp'] = pd.Timestamp.now()
            
            return result_df
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise
    
    return predict_new_data

# Create production predictor
predictor = deploy_time_series_model(bigfeat, model, feature_cols, 'Date', ['group'])

# Use for new predictions
new_predictions = predictor(new_data)
```

## Advanced Techniques

### 1. Custom Time Series Operators

```python
# You can extend BigFeat with custom time series operations
class CustomBigFeat(BigFeat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add custom operators
        if self.enable_time_series:
            self.custom_operators = [
                self._custom_rolling_quantile,
                self._custom_seasonal_decompose,
                self._custom_autocorr_feature
            ]
            self.operators.extend(self.custom_operators)
            self.unary_operators.extend(self.custom_operators)
    
    def _custom_rolling_quantile(self, feature_data):
        """Custom time-based rolling quantile operator"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'rolling_quantile')
        else:
            try:
                window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                window_size = min(window_size, len(feature_data))
                quantile = self.rng.choice([0.25, 0.5, 0.75])
                result = pd.Series(feature_data).rolling(
                    window=window_size, min_periods=1
                ).quantile(quantile)
                return self._clean_feature(result)
            except Exception:
                return feature_data
    
    def _custom_seasonal_decompose(self, feature_data):
        """Simple seasonal decomposition"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'seasonal_decompose')
        else:
            try:
                # Simple seasonal pattern extraction
                series = pd.Series(feature_data)
                # Create a simple seasonal pattern based on position in series
                season_length = min(365, len(series) // 4) if len(series) > 365 else len(series) // 4
                if season_length < 2:
                    return feature_data
                seasonal = series.rolling(window=season_length, center=True, min_periods=1).mean()
                return self._clean_feature(seasonal.fillna(series.mean()).values)
            except Exception:
                return feature_data
    
    def _custom_autocorr_feature(self, feature_data):
        """Autocorrelation-based feature"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'autocorr')
        else:
            try:
                lag = self.rng.choice([1, 2, 3, 5, 7, 10])
                lag = min(lag, len(feature_data) - 1)
                series = pd.Series(feature_data)
                lagged = series.shift(lag)
                correlation = series.corr(lagged)
                
                # Create feature based on correlation strength
                corr_feature = np.full_like(feature_data, correlation if not np.isnan(correlation) else 0)
                return self._clean_feature(corr_feature)
            except Exception:
                return feature_data

# Use custom BigFeat
custom_bigfeat = CustomBigFeat(
    enable_time_series=True,
    datetime_col='Date',
    verbose=True
)
```

### 2. Multi-Step Ahead Forecasting

```python
def create_multi_step_targets(df, target_col, steps=[1, 3, 7], groupby_cols=None):
    """Create multiple forecasting horizons"""
    
    target_cols = {}
    
    for step in steps:
        col_name = f"{target_col}_t+{step}"
        
        if groupby_cols:
            df[col_name] = df.groupby(groupby_cols)[target_col].shift(-step)
        else:
            df[col_name] = df[target_col].shift(-step)
        
        target_cols[f"step_{step}"] = col_name
    
    return df, target_cols

# Create multi-step targets
df_multi, target_mapping = create_multi_step_targets(
    df, 'sales', steps=[1, 3, 7], groupby_cols=['store']
)

# Train separate models for each horizon
models = {}
for step_name, target_col in target_mapping.items():
    print(f"\nTraining model for {step_name}...")
    
    # Remove rows with missing targets
    df_clean = df_multi.dropna(subset=[target_col])
    
    # Configure BigFeat for this horizon
    bigfeat = BigFeat(
        task_type='regression',
        enable_time_series=True,
        datetime_col='Date',
        groupby_cols=['store'],
        verbose=False
    )
    
    # Time-based split
    split_date = df_clean['Date'].quantile(0.8)
    train_mask = df_clean['Date'] <= split_date
    test_mask = df_clean['Date'] > split_date
    
    X_train = df_clean[train_mask][['Date', 'store'] + feature_cols]
    X_test = df_clean[test_mask][['Date', 'store'] + feature_cols]
    y_train = df_clean[train_mask][target_col]
    y_test = df_clean[test_mask][target_col]
    
    # Generate features and train
    X_train_enhanced = bigfeat.fit(X_train, y_train)
    X_test_enhanced = bigfeat.transform(X_test)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_enhanced, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_enhanced)
    r2 = r2_score(y_test, predictions)
    
    models[step_name] = {
        'bigfeat': bigfeat,
        'model': model,
        'r2_score': r2
    }
    
    print(f"{step_name} R² Score: {r2:.4f}")

print("\nMulti-step forecasting models trained successfully!")
```

### 3. Feature Importance Analysis

```python
def analyze_time_series_features(bigfeat, model, X_enhanced, feature_cols):
    """Analyze which time series features are most important"""
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        perm_imp = permutation_importance(model, X_enhanced, y_test)
        importances = perm_imp.importances_mean
    
    # Create feature descriptions
    n_generated = len(bigfeat.tracking_ops) if hasattr(bigfeat, 'tracking_ops') else 0
    n_original = len(feature_cols)
    
    feature_names = []
    feature_types = []
    
    # Generated features
    if hasattr(bigfeat, 'tracking_ops'):
        for i, (ops, ids) in enumerate(zip(bigfeat.tracking_ops, bigfeat.tracking_ids)):
            if not ops:
                # Original feature
                feat_name = f"Original_{feature_cols[ids[0]] if ids and ids[0] < len(feature_cols) else 'Unknown'}"
                feat_type = "Original"
            else:
                # Generated feature
                op_names = []
                for op_info in ops:
                    if len(op_info) > 0:
                        op_name = getattr(op_info[0], '__name__', 'Unknown')
                        op_name = op_name.replace('_safe_', '')
                        op_names.append(op_name)
                
                feat_name = f"Generated_{i}_{'_'.join(op_names[:2])}"
                feat_type = "Time Series" if any('rolling' in op or 'lag' in op or 'ewm' in op 
                                               for op in op_names) else "Generated"
            
            feature_names.append(feat_name)
            feature_types.append(feat_type)
    
    # Add remaining original features
    remaining = len(importances) - len(feature_names)
    for i in range(remaining):
        if i < len(feature_cols):
            feature_names.append(f"Original_{feature_cols[i]}")
            feature_types.append("Original")
        else:
            feature_names.append(f"Feature_{i}")
            feature_types.append("Unknown")
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances,
        'type': feature_types[:len(importances)]
    }).sort_values('importance', ascending=False)
    
    # Analyze by type
    type_analysis = importance_df.groupby('type').agg({
        'importance': ['sum', 'mean', 'count']
    }).round(4)
    
    print("\n=== Feature Importance Analysis ===")
    print(f"\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    print(f"\nImportance by Feature Type:")
    print(type_analysis.to_string())
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        # Feature type distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Top features
        importance_df.head(15).plot(x='feature', y='importance', kind='barh', ax=ax1)
        ax1.set_title('Top 15 Features by Importance')
        ax1.set_xlabel('Importance')
        
        # Importance by type
        type_summary = importance_df.groupby('type')['importance'].sum()
        type_summary.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Importance by Feature Type')
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    
    return importance_df

# Use feature importance analysis
importance_df = analyze_time_series_features(bigfeat, model, X_test_enhanced, feature_cols)
```

## Summary

This comprehensive guide covers everything you need to know about using BigFeat's time series operations:

### Key Takeaways:

1. **Enable time series with proper configuration**: Set `enable_time_series=True` and specify your datetime column
2. **Prepare data correctly**: Always ensure datetime columns are properly formatted and data is sorted chronologically
3. **Configure parameters thoughtfully**: Choose time-based window sizes and lag periods that make sense for your domain and data frequency
4. **Use groupby for multi-series data**: Essential for preventing cross-contamination between different time series
5. **Validate and monitor**: Always check data quality and feature generation results
6. **Start simple, then optimize**: Begin with basic configurations and gradually add complexity based on performance

### Quick Reference:

```python
# Basic setup
bigfeat = BigFeat(
    enable_time_series=True,
    datetime_col='Date',
    groupby_cols=['ID'],  # For multi-series
    window_sizes=['3D', '7D', '14D'],
    lag_periods=['1D', '3D', '7D'],
    time_step='D'
)

# Generate features
X_enhanced = bigfeat.fit(df_with_datetime, target)
X_new_enhanced = bigfeat.transform(new_df_with_datetime)
```

With this guide, you should be able to effectively leverage BigFeat's time series capabilities to enhance your temporal machine learning projects!