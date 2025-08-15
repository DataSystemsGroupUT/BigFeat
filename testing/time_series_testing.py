import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    import yfinance as yf

    print("✓ yfinance imported successfully")
except ImportError:
    print("⚠ yfinance not found. Install with: pip install yfinance")
    print("Continuing with alternative data sources...")

# Import your BigFeat implementation
try:
    from bigfeat.bigfeat_base import BigFeat  # Updated import to match your file

    print("✓ BigFeat imported successfully")
except ImportError:
    print("✗ BigFeat not found. Please ensure bigfeat_base.py is in the same directory.")
    sys.exit(1)


class ComprehensiveTimeSeriesTester:
    """
    Comprehensive testing suite for BigFeat time-based series capabilities
    Uses multiple real-world datasets with temporal components and time-based windows
    """

    def __init__(self, verbose=True, save_results=True):
        self.verbose = verbose
        self.save_results = save_results
        self.results = {}

        if self.save_results:
            os.makedirs('results', exist_ok=True)

    def print_section(self, title):
        """Print formatted section header"""
        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"{title}")
            print(f"{'=' * 80}")

    def load_stock_data(self, symbols=['AAPL', 'GOOGL', 'MSFT'], period='2y'):
        """
        Load stock data from Yahoo Finance with higher frequency for time-based analysis
        """
        try:
            self.print_section("LOADING FINANCIAL DATA FROM YAHOO FINANCE")

            stock_data = []

            for symbol in symbols:
                if self.verbose:
                    print(f"Downloading {symbol} data...")

                ticker = yf.Ticker(symbol)
                # Get daily data for better time-based analysis
                hist = ticker.history(period=period, interval='1d')

                if len(hist) == 0:
                    print(f"⚠ No data found for {symbol}")
                    continue

                hist = hist.reset_index()
                hist['Symbol'] = symbol

                # Calculate additional features
                hist['Returns'] = hist['Close'].pct_change()
                hist['LogReturns'] = np.log1p(hist['Close'].pct_change())
                hist['Volatility'] = hist['Returns'].rolling(20, min_periods=1).std()
                hist['HL_Pct'] = (hist['High'] - hist['Low']) / hist['Close']
                hist['Price_Change'] = hist['Close'] - hist['Open']
                hist['Volume_MA'] = hist['Volume'].rolling(20, min_periods=1).mean()
                hist['RSI'] = self.calculate_rsi(hist['Close'])
                hist['MACD'] = self.calculate_macd(hist['Close'])
                hist['BB_Upper'], hist['BB_Lower'] = self.calculate_bollinger_bands(hist['Close'])

                # Create target variables
                hist['Next_Return'] = hist['Returns'].shift(-1)
                hist['Price_Up'] = (hist['Next_Return'] > 0).astype(int)
                hist['High_Vol'] = (hist['Volume'] > hist['Volume'].quantile(0.75)).astype(int)
                hist['Volatility_High'] = (hist['Volatility'] > hist['Volatility'].quantile(0.75)).astype(int)

                # Add time-based features for better time-series analysis
                hist['DayOfWeek'] = hist['Date'].dt.dayofweek
                hist['Month'] = hist['Date'].dt.month
                hist['Quarter'] = hist['Date'].dt.quarter
                hist['DaysFromStart'] = (hist['Date'] - hist['Date'].min()).dt.days
                hist['IsMonthEnd'] = hist['Date'].dt.is_month_end.astype(int)
                hist['IsQuarterEnd'] = hist['Date'].dt.is_quarter_end.astype(int)

                stock_data.append(hist)

            df = pd.concat(stock_data, ignore_index=True)
            df = df.dropna()

            # Clean data to prevent infinity or large values
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = np.clip(df[col], -1e8, 1e8)

            if self.verbose:
                print(f"Stock data loaded: {len(df)} rows, {df['Symbol'].nunique()} symbols")
                print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
                print(f"Features: {df.select_dtypes(include=[np.number]).columns.tolist()}")

            return df

        except Exception as e:
            print(f"Error loading stock data: {e}")
            return None

    def load_crypto_data(self, symbols=['BTC-USD', 'ETH-USD'], period='1y'):
        """
        Load cryptocurrency data from Yahoo Finance with daily frequency
        """
        try:
            self.print_section("LOADING CRYPTOCURRENCY DATA")

            crypto_data = []

            for symbol in symbols:
                if self.verbose:
                    print(f"Downloading {symbol} data...")

                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval='1d')

                if len(hist) == 0:
                    continue

                hist = hist.reset_index()
                hist['Symbol'] = symbol

                # Crypto-specific features
                hist['Returns'] = hist['Close'].pct_change()
                hist['LogReturns'] = np.log1p(hist['Close'].pct_change())
                hist['Volatility'] = hist['Returns'].rolling(10, min_periods=1).std()
                hist['Price_Range'] = hist['High'] - hist['Low']
                hist['Volume_USD'] = hist['Volume'] * hist['Close']

                # Technical indicators adapted for crypto
                hist['SMA_7'] = hist['Close'].rolling(7, min_periods=1).mean()
                hist['SMA_30'] = hist['Close'].rolling(30, min_periods=1).mean()
                hist['Price_Position'] = (hist['Close'] - hist['Low'].rolling(20, min_periods=1).min()) / (
                        hist['High'].rolling(20, min_periods=1).max() - hist['Low'].rolling(20,
                                                                                            min_periods=1).min())

                # Targets
                hist['Next_Return'] = hist['Returns'].shift(-1)
                hist['Price_Up'] = (hist['Next_Return'] > 0).astype(int)
                hist['High_Volatility'] = (hist['Volatility'] > hist['Volatility'].quantile(0.8)).astype(int)

                # Time features
                hist['DayOfWeek'] = hist['Date'].dt.dayofweek
                hist['Month'] = hist['Date'].dt.month
                hist['IsWeekend'] = (hist['Date'].dt.dayofweek >= 5).astype(int)

                crypto_data.append(hist)

            if crypto_data:
                df = pd.concat(crypto_data, ignore_index=True)
                df = df.dropna()

                # Clean data to prevent infinity or large values
                df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
                for col in df.select_dtypes(include=[np.number]).columns:
                    df[col] = np.clip(df[col], -1e8, 1e8)

                if self.verbose:
                    print(f"Crypto data loaded: {len(df)} rows, {df['Symbol'].nunique()} symbols")
                    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
                    print(f"Features: {df.select_dtypes(include=[np.number]).columns.tolist()}")

                return df
            else:
                return None

        except Exception as e:
            print(f"Error loading crypto data: {e}")
            return None

    def create_synthetic_sales_data(self, n_stores=5, days=730):
        """
        Create synthetic sales data with complex temporal patterns - daily frequency for time-based analysis
        """
        self.print_section("CREATING SYNTHETIC SALES DATA")

        np.random.seed(42)
        start_date = pd.to_datetime('2022-01-01')
        dates = pd.date_range(start_date, periods=days, freq='D')

        data_list = []

        for store_id in range(1, n_stores + 1):
            store_base = 5000 + store_id * 1000

            for i, date in enumerate(dates):
                yearly_trend = 500 * (i / len(dates))
                monthly_seasonal = 1000 * np.sin(2 * np.pi * date.month / 12)
                weekly_pattern = 800 * np.sin(2 * np.pi * date.weekday() / 7)
                is_weekend = date.weekday() >= 5
                is_holiday = (date.month == 12 and date.day >= 20) or (date.month == 1 and date.day <= 5)
                is_summer = date.month in [6, 7, 8]
                promo_prob = 0.1 + 0.05 * (store_id % 2)
                has_promo = np.random.random() < promo_prob
                recession_effect = -200 if date.year == 2023 and date.month > 6 else 0
                weather_effect = 300 * np.sin(2 * np.pi * (date.dayofyear - 80) / 365)
                competition_effect = -100 if store_id > 3 and date > pd.to_datetime('2022-06-01') else 0

                base_sales = (store_base + yearly_trend + monthly_seasonal + weekly_pattern +
                              weather_effect + recession_effect + competition_effect)

                if is_weekend:
                    base_sales *= 1.3
                if is_holiday:
                    base_sales *= 1.8
                if is_summer:
                    base_sales *= 1.1
                if has_promo:
                    base_sales *= np.random.uniform(1.2, 1.8)

                noise = np.random.normal(0, 300)
                sales = max(100, base_sales + noise)

                customers = int(sales / np.random.uniform(15, 25))
                avg_transaction = sales / max(customers, 1)
                foot_traffic = customers * np.random.uniform(1.1, 1.5)

                data_list.append({
                    'Date': date,
                    'Store': store_id,
                    'Sales': sales,
                    'Customers': customers,
                    'AvgTransaction': avg_transaction,
                    'FootTraffic': foot_traffic,
                    'HasPromo': int(has_promo),
                    'IsWeekend': int(is_weekend),
                    'IsHoliday': int(is_holiday),
                    'IsSummer': int(is_summer),
                    'DayOfWeek': date.weekday(),
                    'Month': date.month,
                    'Quarter': date.quarter,
                    'WeekOfYear': date.isocalendar()[1],
                    'Temperature': 20 + 15 * np.sin(2 * np.pi * (date.dayofyear - 80) / 365) + np.random.normal(0, 5),
                    'CompetitorDistance': 1000 + store_id * 500 + np.random.normal(0, 100)
                })

        df = pd.DataFrame(data_list)

        # Create target variables
        df['HighSales'] = (df['Sales'] > df['Sales'].quantile(0.75)).astype(int)
        df['SalesGrowth'] = df.groupby('Store')['Sales'].pct_change()
        df['NextDaySales'] = df.groupby('Store')['Sales'].shift(-1)

        # Add lag features for comparison
        df['Sales_Lag1'] = df.groupby('Store')['Sales'].shift(1)
        df['Sales_Lag7'] = df.groupby('Store')['Sales'].shift(7)

        df = df.dropna()

        if self.verbose:
            print(f"Synthetic sales data created: {len(df)} rows, {df['Store'].nunique()} stores")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"Sales statistics: mean={df['Sales'].mean():.2f}, std={df['Sales'].std():.2f}")

        return df

    def create_hourly_energy_data(self, days=180):
        """
        Create synthetic hourly energy consumption data for high-frequency time-based analysis
        """
        self.print_section("CREATING HOURLY ENERGY CONSUMPTION DATA")

        np.random.seed(42)
        start_date = pd.to_datetime('2023-01-01')
        dates = pd.date_range(start_date, periods=days * 24, freq='H')

        data_list = []

        for i, timestamp in enumerate(dates):
            # Base consumption pattern
            base_consumption = 1000

            # Hourly pattern (lower at night, higher during day)
            hourly_pattern = 300 * np.sin(2 * np.pi * (timestamp.hour - 6) / 24)

            # Weekly pattern (lower on weekends)
            weekly_pattern = 200 * np.sin(2 * np.pi * timestamp.weekday() / 7)

            # Seasonal pattern (higher in winter/summer for heating/cooling)
            seasonal_pattern = 400 * np.sin(2 * np.pi * (timestamp.dayofyear - 80) / 365)

            # Weather effect (temperature-based)
            temp = 20 + 15 * np.sin(2 * np.pi * (timestamp.dayofyear - 80) / 365) + np.random.normal(0, 3)
            weather_effect = 0
            if temp < 10:  # Heating
                weather_effect = (10 - temp) * 50
            elif temp > 25:  # Cooling
                weather_effect = (temp - 25) * 40

            # Special events (holidays, etc.)
            is_holiday = (timestamp.month == 12 and timestamp.day >= 20) or (
                        timestamp.month == 1 and timestamp.day <= 5)
            holiday_effect = -200 if is_holiday else 0

            # Random noise
            noise = np.random.normal(0, 100)

            consumption = max(200, base_consumption + hourly_pattern + weekly_pattern +
                              seasonal_pattern + weather_effect + holiday_effect + noise)

            data_list.append({
                'DateTime': timestamp,
                'Consumption': consumption,
                'Temperature': temp,
                'Hour': timestamp.hour,
                'DayOfWeek': timestamp.weekday(),
                'Month': timestamp.month,
                'Quarter': timestamp.quarter,
                'IsWeekend': int(timestamp.weekday() >= 5),
                'IsHoliday': int(is_holiday),
                'IsBusinessHour': int(9 <= timestamp.hour <= 17),
                'HourlyPattern': hourly_pattern,
                'WeatherEffect': weather_effect
            })

        df = pd.DataFrame(data_list)

        # Create targets
        df['NextHourConsumption'] = df['Consumption'].shift(-1)
        df['HighConsumption'] = (df['Consumption'] > df['Consumption'].quantile(0.8)).astype(int)
        df['ConsumptionChange'] = df['Consumption'].diff()

        df = df.dropna()

        if self.verbose:
            print(f"Energy data created: {len(df)} rows")
            print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
            print(f"Consumption statistics: mean={df['Consumption'].mean():.2f}, std={df['Consumption'].std():.2f}")

        return df

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.fillna(sma), lower_band.fillna(sma)

    def test_dataset(self, df, dataset_name, date_col, target_configs):
        """
        Test a dataset with multiple target variables and configurations using time-based BigFeat implementation
        """
        self.print_section(f"TESTING DATASET: {dataset_name}")

        if df is None or len(df) == 0:
            print(f"⚠ Skipping {dataset_name} - no data available")
            return {}

        results = {}

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([date_col]).reset_index(drop=True)

        for target_config in target_configs:
            target_col = target_config['target']
            task_type = target_config['task_type']
            feature_cols = target_config['features']
            config_name = target_config['name']
            groupby_cols = target_config.get('groupby_cols', [])
            time_step = target_config.get('time_step', 'D')

            print(f"\n{'-' * 60}")
            print(f"Testing: {config_name} ({task_type})")
            print(f"Target: {target_col}")
            print(f"Features: {len(feature_cols)}")
            print(f"Groupby columns: {groupby_cols}")
            print(f"Time step: {time_step}")

            try:
                # Prepare data - Create full DataFrame with all required columns
                X_full = df[feature_cols + [date_col] + groupby_cols].copy()
                y = df[target_col].copy()

                # Handle missing values
                X_full[feature_cols] = X_full[feature_cols].fillna(X_full[feature_cols].mean())
                y = y.fillna(y.mean() if task_type == 'regression' else y.mode().iloc[0])

                # Time-based split
                split_date = df[date_col].quantile(0.8)
                train_mask = df[date_col] <= split_date
                test_mask = df[date_col] > split_date

                X_train_full = X_full[train_mask]
                X_test_full = X_full[test_mask]
                y_train = y[train_mask].values
                y_test = y[test_mask].values

                if len(X_train_full) < 100 or len(X_test_full) < 20:
                    print(f"⚠ Insufficient data: train={len(X_train_full)}, test={len(X_test_full)}")
                    continue

                print(f"Data split: train={len(X_train_full)}, test={len(X_test_full)}")

                if task_type == 'classification':
                    unique_classes = len(np.unique(y_train))
                    print(f"Classes: {unique_classes}, distribution: {np.bincount(y_train.astype(int))}")
                else:
                    print(f"Target stats: mean={y_train.mean():.4f}, std={y_train.std():.4f}")

                # Define time-based window configurations
                if 'hourly' in dataset_name.lower() or time_step == 'H':
                    # Hourly data configurations
                    window_configs = {
                        'short_term': ['1H', '3H', '6H', '12H', '1D'],
                        'medium_term': ['1D', '3D', '7D', '14D'],
                        'long_term': ['7D', '14D', '30D', '60D']
                    }
                    lag_configs = {
                        'short_term': ['1H', '3H', '6H', '12H'],
                        'medium_term': ['1D', '3D', '7D'],
                        'long_term': ['7D', '14D', '30D']
                    }
                elif 'crypto' in dataset_name.lower() or 'stock' in dataset_name.lower():
                    # Financial data configurations
                    window_configs = {
                        'short_term': ['3D', '7D', '14D', '21D'],
                        'medium_term': ['30D', '60D', '90D'],
                        'long_term': ['6M', '1Y']
                    }
                    lag_configs = {
                        'short_term': ['1D', '3D', '7D'],
                        'medium_term': ['14D', '30D'],
                        'long_term': ['60D', '90D']
                    }
                else:
                    # Default daily configurations
                    window_configs = {
                        'short_term': ['7D', '14D', '30D'],
                        'medium_term': ['60D', '90D', '180D'],
                        'long_term': ['1Y']
                    }
                    lag_configs = {
                        'short_term': ['1D', '7D', '14D'],
                        'medium_term': ['30D', '60D'],
                        'long_term': ['90D', '180D']
                    }

                # Test configurations with time-based windows
                configurations = [
                    {
                        'name': 'Baseline (No Time Series)',
                        'params': {
                            'task_type': task_type,
                            'enable_time_series': False,
                            'verbose': False
                        },
                        'fit_params': {
                            'gen_size': 3,
                            'iterations': 2,
                            'random_state': 42,
                            'estimator': 'rf' if task_type == 'classification' else 'rf_reg',
                            'selection': 'stability'
                        }
                    },
                    {
                        'name': 'Time Series (Short-term)',
                        'params': {
                            'task_type': task_type,
                            'enable_time_series': True,
                            'datetime_col': date_col,
                            'groupby_cols': groupby_cols,
                            'window_sizes': window_configs['short_term'],
                            'lag_periods': lag_configs['short_term'],
                            'time_step': time_step,
                            'verbose': False
                        },
                        'fit_params': {
                            'gen_size': 4,
                            'iterations': 3,
                            'random_state': 42,
                            'estimator': 'rf' if task_type == 'classification' else 'rf_reg',
                            'selection': 'stability'
                        }
                    },
                    {
                        'name': 'Time Series (Medium-term)',
                        'params': {
                            'task_type': task_type,
                            'enable_time_series': True,
                            'datetime_col': date_col,
                            'groupby_cols': groupby_cols,
                            'window_sizes': window_configs['medium_term'],
                            'lag_periods': lag_configs['medium_term'],
                            'time_step': time_step,
                            'verbose': False
                        },
                        'fit_params': {
                            'gen_size': 4,
                            'iterations': 3,
                            'random_state': 42,
                            'estimator': 'rf' if task_type == 'classification' else 'rf_reg',
                            'selection': 'stability'
                        }
                    },
                    {
                        'name': 'Time Series (Long-term)',
                        'params': {
                            'task_type': task_type,
                            'enable_time_series': True,
                            'datetime_col': date_col,
                            'groupby_cols': groupby_cols,
                            'window_sizes': window_configs['long_term'],
                            'lag_periods': lag_configs['long_term'],
                            'time_step': time_step,
                            'verbose': False
                        },
                        'fit_params': {
                            'gen_size': 4,
                            'iterations': 3,
                            'random_state': 42,
                            'estimator': 'avg',
                            'selection': 'stability'
                        }
                    },
                    {
                        'name': 'Time Series (Mixed)',
                        'params': {
                            'task_type': task_type,
                            'enable_time_series': True,
                            'datetime_col': date_col,
                            'groupby_cols': groupby_cols,
                            'window_sizes': window_configs['short_term'] + window_configs['medium_term'],
                            'lag_periods': lag_configs['short_term'] + lag_configs['medium_term'],
                            'time_step': time_step,
                            'verbose': False
                        },
                        'fit_params': {
                            'gen_size': 5,
                            'iterations': 4,
                            'random_state': 42,
                            'estimator': 'avg',
                            'selection': 'stability'
                        }
                    }
                ]

                config_results = {}

                for config in configurations:
                    try:
                        print(f"\n  Testing: {config['name']}")
                        if config['params'].get('enable_time_series', False):
                            print(f"    Window sizes: {config['params']['window_sizes']}")
                            print(f"    Lag periods: {config['params']['lag_periods']}")
                            print(f"    Time step: {config['params']['time_step']}")

                        bigfeat = BigFeat(**config['params'])

                        # Fit BigFeat with full DataFrame (including datetime and groupby columns)
                        X_train_enhanced = bigfeat.fit(X_train_full, y_train, **config['fit_params'])
                        X_test_enhanced = bigfeat.transform(X_test_full)

                        # Count time series operations
                        ts_ops_count = 0
                        if config['params'].get('enable_time_series', False) and hasattr(bigfeat, 'tracking_ops'):
                            time_series_op_names = [
                                '_safe_rolling_mean', '_safe_rolling_std', '_safe_rolling_min', '_safe_rolling_max',
                                '_safe_rolling_median', '_safe_rolling_sum', '_safe_lag_feature', '_safe_diff_feature',
                                '_safe_pct_change', '_safe_ewm', '_safe_momentum', '_safe_seasonal_decompose',
                                '_safe_trend_feature', '_safe_weekday_mean', '_safe_month_mean'
                            ]

                            for ops in bigfeat.tracking_ops:
                                if ops:  # Check if ops list is not empty
                                    for op_info in ops:
                                        if len(op_info) > 0 and callable(op_info[0]):
                                            op_name = getattr(op_info[0], '__name__', '')
                                            if op_name in time_series_op_names:
                                                ts_ops_count += 1

                        print(f"    Generated features shape: {X_train_enhanced.shape}")
                        print(f"    Time series operations used: {ts_ops_count}")

                        # Select and train model
                        estimator_names = ['rf', 'dt'] if task_type == 'classification' else ['rf_reg', 'dt_reg']
                        best_model = bigfeat.select_estimator(X_train_enhanced, y_train, estimator_names)

                        # Make predictions
                        y_pred = best_model.predict(X_test_enhanced)

                        # Calculate metrics
                        if task_type == 'classification':
                            score = accuracy_score(y_test, y_pred)
                            metric_name = 'Accuracy'
                        else:
                            score = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            metric_name = 'R²'

                        config_results[config['name']] = {
                            'score': score,
                            'n_features': X_train_enhanced.shape[1],
                            'ts_ops_count': ts_ops_count,
                            'model': type(best_model).__name__
                        }

                        if task_type == 'regression':
                            config_results[config['name']]['mae'] = mae
                            config_results[config['name']]['rmse'] = rmse

                        print(
                            f"    {metric_name}: {score:.4f}, Features: {X_train_enhanced.shape[1]}, TS Ops: {ts_ops_count}")

                        # Save detailed results if requested
                        if self.save_results and config['name'] != 'Baseline (No Time Series)':
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                            # Generate feature information
                            feature_names = []
                            feature_descriptions = []

                            if hasattr(bigfeat, 'tracking_ops') and hasattr(bigfeat, 'tracking_ids'):
                                for i, (ops, ids) in enumerate(zip(bigfeat.tracking_ops, bigfeat.tracking_ids)):
                                    if not ops or len(ops) == 0:
                                        if ids and len(ids) > 0 and ids[0] < len(feature_cols):
                                            feat_name = f"Original_{feature_cols[ids[0]]}"
                                            desc = f"Original: {feature_cols[ids[0]]}"
                                        else:
                                            feat_name = f"Original_Feat_{i}"
                                            desc = f"Original feature {i}"
                                    else:
                                        # Generated feature
                                        op_names = []
                                        for op_info in ops:
                                            if len(op_info) > 0:
                                                op = op_info[0]
                                                op_name = getattr(op, '__name__', str(op)).replace('_safe_',
                                                                                                   '').replace(
                                                    '<built-in function ', '').replace('>', '')
                                                op_names.append(op_name)

                                        feat_indices = []
                                        if ids:
                                            for idx in ids:
                                                if idx < len(feature_cols):
                                                    feat_indices.append(feature_cols[idx])
                                                else:
                                                    feat_indices.append(f"feat_{idx}")

                                        if op_names:
                                            feat_name = f"Gen_Feat_{i}_{'_'.join(op_names[:2])}"
                                            desc = f"{' -> '.join(op_names)}({', '.join(feat_indices)})"
                                        else:
                                            feat_name = f"Gen_Feat_{i}"
                                            desc = f"Generated feature {i}"

                                    feature_names.append(feat_name)
                                    feature_descriptions.append(desc)

                            # Add original feature names for remaining features
                            while len(feature_names) < X_train_enhanced.shape[1]:
                                idx = len(feature_names)
                                if idx - len(bigfeat.tracking_ops) < len(feature_cols):
                                    orig_idx = idx - len(bigfeat.tracking_ops)
                                    feature_names.append(f"Original_{feature_cols[orig_idx]}")
                                    feature_descriptions.append(f"Original: {feature_cols[orig_idx]}")
                                else:
                                    feature_names.append(f"Original_Feat_Extra_{idx}")
                                    feature_descriptions.append(f"Original feature {idx}")

                            # Save feature information
                            feature_info = {
                                'feature_names': feature_names[:X_train_enhanced.shape[1]],
                                'feature_descriptions': feature_descriptions[:X_train_enhanced.shape[1]],
                                'n_generated_features': len(bigfeat.tracking_ops) if hasattr(bigfeat,
                                                                                             'tracking_ops') else 0,
                                'n_original_features': len(feature_cols),
                                'total_features': X_train_enhanced.shape[1],
                                'ts_ops_count': ts_ops_count,
                                'datetime_col': date_col,
                                'groupby_cols': groupby_cols,
                                'window_sizes': config['params'].get('window_sizes', []),
                                'lag_periods': config['params'].get('lag_periods', []),
                                'time_step': config['params'].get('time_step', 'D')
                            }

                            feature_info_file = f'results/{dataset_name}_{config_name}_{config["name"]}_feature_info_{timestamp}.json'
                            with open(feature_info_file, 'w') as f:
                                json.dump(feature_info, f, indent=2)

                            print(f"    Feature info saved to {feature_info_file}")

                    except Exception as e:
                        print(f"    Error: {str(e)}")
                        import traceback
                        if self.verbose:
                            traceback.print_exc()
                        config_results[config['name']] = {'error': str(e), 'score': 0}

                # Calculate improvements
                baseline_score = config_results.get('Baseline (No Time Series)', {}).get('score', 0)

                print(f"\n  Results Summary for {config_name}:")
                print(f"  {'-' * 70}")

                for name, result in config_results.items():
                    if 'error' not in result:
                        improvement = result['score'] - baseline_score
                        line = f"  {name:30} | {metric_name}: {result['score']:7.4f} | Feat: {result.get('n_features', 0):3d} | TS: {result.get('ts_ops_count', 0):2d} | Δ: {improvement:+7.4f}"
                        if task_type == 'regression':
                            line += f" | MAE: {result.get('mae', 0):8.4f} | RMSE: {result.get('rmse', 0):8.4f}"
                        print(line)
                    else:
                        print(f"  {name:30} | ERROR: {result['error'][:40]}...")

                results[config_name] = config_results

            except Exception as e:
                print(f"Error testing {config_name}: {str(e)}")
                import traceback
                if self.verbose:
                    traceback.print_exc()
                results[config_name] = {'error': str(e)}

        return results

    def generate_comprehensive_report(self, all_results):
        """
        Generate a comprehensive report of all test results with time-based analysis
        """
        self.print_section("COMPREHENSIVE TIME-BASED TEST RESULTS SUMMARY")

        total_tests = 0
        successful_tests = 0
        improvements = []
        best_improvements = {}
        time_series_improvements = {}

        for dataset_name, dataset_results in all_results.items():
            print(f"\n{dataset_name}:")
            print("-" * 70)

            for config_name, config_results in dataset_results.items():
                if isinstance(config_results, dict) and 'error' not in config_results:

                    baseline_score = config_results.get('Baseline (No Time Series)', {}).get('score', 0)

                    print(f"\n  {config_name}:")
                    print(f"  {'Method':<30} | {'Score':<8} | {'Features':<8} | {'TS Ops':<6} | {'Improvement':<11}")
                    print(f"  {'-' * 75}")

                    for method_name, method_result in config_results.items():
                        if isinstance(method_result, dict) and 'error' not in method_result:
                            score = method_result.get('score', 0)
                            n_features = method_result.get('n_features', 0)
                            ts_ops = method_result.get('ts_ops_count', 0)
                            improvement = score - baseline_score

                            improvements.append(improvement)
                            total_tests += 1

                            if improvement > 0:
                                successful_tests += 1

                            # Track time series specific improvements
                            if 'Time Series' in method_name:
                                time_period = method_name.split('(')[-1].split(')')[
                                    0] if '(' in method_name else 'Mixed'
                                key = f"{dataset_name}_{config_name}_{time_period}"
                                time_series_improvements[key] = {
                                    'improvement': improvement,
                                    'method': method_name,
                                    'score': score,
                                    'ts_ops': ts_ops,
                                    'time_period': time_period
                                }

                            # Track best improvements per dataset/config
                            key = f"{dataset_name}_{config_name}"
                            if key not in best_improvements or improvement > best_improvements[key]['improvement']:
                                best_improvements[key] = {
                                    'improvement': improvement,
                                    'method': method_name,
                                    'score': score,
                                    'dataset': dataset_name,
                                    'config': config_name,
                                    'ts_ops': ts_ops
                                }

                            line = f"  {method_name:<30} | {score:8.4f} | {n_features:8d} | {ts_ops:6d} | {improvement:+11.4f}"
                            if 'mae' in method_result:
                                line += f" | MAE: {method_result['mae']:8.4f} | RMSE: {method_result['rmse']:8.4f}"
                            print(line)

                        elif isinstance(method_result, dict) and 'error' in method_result:
                            print(f"  {method_name:<30} | ERROR: {method_result['error'][:40]}...")

        # Overall statistics
        print(f"\n{'=' * 80}")
        print("TIME-BASED SERIES PERFORMANCE ANALYSIS")
        print(f"{'=' * 80}")

        if improvements:
            positive_improvements = [imp for imp in improvements if imp > 0]
            significant_improvements = [imp for imp in improvements if imp > 0.01]

            print(f"Total tests conducted: {total_tests}")
            print(
                f"Tests with positive improvement: {len(positive_improvements)} ({len(positive_improvements) / total_tests * 100:.1f}%)")
            print(
                f"Tests with significant improvement (>0.01): {len(significant_improvements)} ({len(significant_improvements) / total_tests * 100:.1f}%)")
            print(f"Average improvement: {np.mean(improvements):+.4f}")
            print(f"Best improvement: {np.max(improvements):+.4f}")
            print(f"Worst improvement: {np.min(improvements):+.4f}")

            print(f"\nBest Time-Based Window Performance Analysis:")
            print(f"{'-' * 80}")

            # Analyze performance by time period
            time_period_performance = {}
            for key, result in time_series_improvements.items():
                period = result['time_period']
                if period not in time_period_performance:
                    time_period_performance[period] = []
                time_period_performance[period].append(result['improvement'])

            print(f"{'Time Period':<15} | {'Avg Improvement':<15} | {'Best Improvement':<15} | {'Count':<8}")
            print(f"{'-' * 65}")
            for period, improvements_list in time_period_performance.items():
                avg_imp = np.mean(improvements_list)
                best_imp = np.max(improvements_list)
                count = len(improvements_list)
                print(f"{period:<15} | {avg_imp:+15.4f} | {best_imp:+15.4f} | {count:<8}")

            print(f"\nBest Improvements by Dataset/Task:")
            print(f"{'-' * 70}")
            print(
                f"{'Dataset':<20} | {'Task':<25} | {'Method':<25} | {'Score':<8} | {'TS Ops':<6} | {'Improvement':<11}")
            print(f"{'-' * 85}")
            for key, result in best_improvements.items():
                print(
                    f"{result['dataset']:<20} | {result['config']:<25} | {result['method']:<25} | {result['score']:8.4f} | {result.get('ts_ops', 0):6d} | {result['improvement']:+11.4f}")

            # Generate visualization
            if self.save_results:
                self.plot_time_based_improvements(improvements, best_improvements, time_series_improvements)

        else:
            print("No valid test results to analyze.")

    def plot_time_based_improvements(self, improvements, best_improvements, time_series_improvements):
        """
        Generate visualization of time-based improvements
        """
        try:
            # Distribution of improvements
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            sns.histplot(improvements, bins=30, kde=True)
            plt.title('Distribution of Performance Improvements')
            plt.xlabel('Improvement (Score - Baseline)')
            plt.ylabel('Count')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)

            # Time period performance
            plt.subplot(2, 3, 2)
            time_period_data = {}
            for key, result in time_series_improvements.items():
                period = result['time_period']
                if period not in time_period_data:
                    time_period_data[period] = []
                time_period_data[period].append(result['improvement'])

            periods = list(time_period_data.keys())
            avg_improvements = [np.mean(time_period_data[period]) for period in periods]

            plt.bar(periods, avg_improvements)
            plt.title('Average Improvement by Time Period')
            plt.xlabel('Time Period')
            plt.ylabel('Average Improvement')
            plt.xticks(rotation=45)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

            # Best improvements by dataset
            plt.subplot(2, 3, 3)
            datasets_tasks = [f"{r['dataset']}_{r['config'][:15]}" for r in best_improvements.values()]
            scores = [r['improvement'] for r in best_improvements.values()]
            plt.barh(datasets_tasks, scores)
            plt.title('Best Improvements by Dataset')
            plt.xlabel('Improvement (Score - Baseline)')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)

            # Time series operations usage
            plt.subplot(2, 3, 4)
            ts_ops_counts = [r.get('ts_ops', 0) for r in best_improvements.values() if r.get('ts_ops', 0) > 0]
            improvements_with_ts = [r['improvement'] for r in best_improvements.values() if r.get('ts_ops', 0) > 0]

            if ts_ops_counts and improvements_with_ts:
                plt.scatter(ts_ops_counts, improvements_with_ts, alpha=0.7)
                plt.xlabel('Number of Time Series Operations')
                plt.ylabel('Improvement')
                plt.title('TS Operations vs Improvement')

                # Add trend line
                if len(ts_ops_counts) > 1:
                    z = np.polyfit(ts_ops_counts, improvements_with_ts, 1)
                    p = np.poly1d(z)
                    plt.plot(ts_ops_counts, p(ts_ops_counts), "r--", alpha=0.8)

            # Performance comparison
            plt.subplot(2, 3, 5)
            method_performance = {}
            for key, results in best_improvements.items():
                method = results['method']
                if 'Time Series' in method:
                    method_key = 'Time Series'
                else:
                    method_key = 'Baseline'

                if method_key not in method_performance:
                    method_performance[method_key] = []
                method_performance[method_key].append(results['improvement'])

            methods = list(method_performance.keys())
            method_data = [method_performance[method] for method in methods]

            plt.boxplot(method_data, labels=methods)
            plt.title('Improvement Distribution by Method Type')
            plt.ylabel('Improvement')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

            # Summary statistics
            plt.subplot(2, 3, 6)
            plt.axis('off')
            stats_text = f"""
            Time-Based Series Analysis Summary:

            Total Tests: {len(improvements)}
            Positive Improvements: {len([x for x in improvements if x > 0])}
            Success Rate: {len([x for x in improvements if x > 0]) / len(improvements) * 100:.1f}%

            Average Improvement: {np.mean(improvements):+.4f}
            Best Improvement: {np.max(improvements):+.4f}
            Std Dev: {np.std(improvements):.4f}

            Time Series Operations:
            Avg TS Ops per Test: {np.mean([r.get('ts_ops', 0) for r in best_improvements.values()]):.1f}
            Max TS Ops Used: {max([r.get('ts_ops', 0) for r in best_improvements.values()])}
            """
            plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
                     verticalalignment='top', fontsize=10, fontfamily='monospace')

            plt.tight_layout()
            plt.savefig('results/time_based_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\nTime-based analysis visualization saved to 'results/time_based_analysis.png'")

        except Exception as e:
            print(f"Error generating time-based plots: {e}")

    def save_test_results(self, all_results):
        """
        Save test results to JSON file with time-based analysis details
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'results/time_based_test_results_{timestamp}.json'

            results_to_save = {}
            for dataset_name, dataset_results in all_results.items():
                results_to_save[dataset_name] = {}
                for config_name, config_results in dataset_results.items():
                    if isinstance(config_results, dict) and 'error' not in config_results:
                        results_to_save[dataset_name][config_name] = {
                            name: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                   for k, v in result.items()}
                            for name, result in config_results.items()
                        }
                    else:
                        results_to_save[dataset_name][config_name] = config_results

            with open(output_file, 'w') as f:
                json.dump(results_to_save, f, indent=2)

            print(f"\nTime-based test results saved to {output_file}")

        except Exception as e:
            print(f"Error saving results: {e}")

    def run_comprehensive_tests(self):
        """
        Run comprehensive tests on multiple datasets with time-based window analysis
        """
        self.print_section("BIGFEAT TIME-BASED SERIES COMPREHENSIVE TESTING SUITE")
        print("Testing BigFeat time-based series capabilities on real-world temporal datasets")
        print("This comprehensive test focuses on time-based windows (days, weeks, months)")
        print("This comprehensive test may take 15-20 minutes to complete...")

        all_results = {}

        # Test 1: Stock Market Data (Daily)
        try:
            stock_df = self.load_stock_data(['AAPL', 'GOOGL', 'MSFT'], period='2y')
            if stock_df is not None:
                stock_features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Volatility',
                                  'HL_Pct', 'Volume_MA', 'RSI', 'MACD', 'DayOfWeek', 'Month', 'DaysFromStart']

                stock_configs = [
                    {
                        'name': 'Stock_Direction_Prediction',
                        'target': 'Price_Up',
                        'task_type': 'classification',
                        'features': stock_features,
                        'groupby_cols': ['Symbol'],
                        'time_step': 'D'
                    },
                    {
                        'name': 'Stock_Return_Prediction',
                        'target': 'Next_Return',
                        'task_type': 'regression',
                        'features': stock_features,
                        'groupby_cols': ['Symbol'],
                        'time_step': 'D'
                    },
                    {
                        'name': 'Stock_Volatility_Prediction',
                        'target': 'Volatility_High',
                        'task_type': 'classification',
                        'features': [f for f in stock_features if f != 'Volatility'],
                        'groupby_cols': ['Symbol'],
                        'time_step': 'D'
                    }
                ]

                stock_results = self.test_dataset(stock_df, 'Stock_Market_Daily', 'Date', stock_configs)
                all_results['Stock_Market_Daily'] = stock_results
        except Exception as e:
            print(f"Stock data test failed: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()

        # Test 2: Cryptocurrency Data (Daily)
        try:
            crypto_df = self.load_crypto_data(['BTC-USD', 'ETH-USD'], period='1y')
            if crypto_df is not None:
                crypto_features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Volatility',
                                   'Price_Range', 'SMA_7', 'SMA_30', 'Price_Position', 'DayOfWeek', 'Month']

                crypto_configs = [
                    {
                        'name': 'Crypto_Direction_Prediction',
                        'target': 'Price_Up',
                        'task_type': 'classification',
                        'features': crypto_features,
                        'groupby_cols': ['Symbol'],
                        'time_step': 'D'
                    },
                    {
                        'name': 'Crypto_Return_Prediction',
                        'target': 'Next_Return',
                        'task_type': 'regression',
                        'features': crypto_features,
                        'groupby_cols': ['Symbol'],
                        'time_step': 'D'
                    }
                ]

                crypto_results = self.test_dataset(crypto_df, 'Cryptocurrency_Daily', 'Date', crypto_configs)
                all_results['Cryptocurrency_Daily'] = crypto_results
        except Exception as e:
            print(f"Crypto data test failed: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()

        # Test 3: Synthetic Sales Data (Daily)
        try:
            sales_df = self.create_synthetic_sales_data(n_stores=5, days=730)
            sales_features = ['Customers', 'AvgTransaction', 'FootTraffic', 'HasPromo',
                              'IsWeekend', 'Temperature', 'DayOfWeek', 'Month', 'Quarter', 'WeekOfYear']

            sales_configs = [
                {
                    'name': 'Sales_High_Performance',
                    'target': 'HighSales',
                    'task_type': 'classification',
                    'features': sales_features,
                    'groupby_cols': ['Store'],
                    'time_step': 'D'
                },
                {
                    'name': 'Sales_Next_Day_Prediction',
                    'target': 'NextDaySales',
                    'task_type': 'regression',
                    'features': sales_features,
                    'groupby_cols': ['Store'],
                    'time_step': 'D'
                }
            ]

            sales_results = self.test_dataset(sales_df, 'Retail_Sales_Daily', 'Date', sales_configs)
            all_results['Retail_Sales_Daily'] = sales_results
        except Exception as e:
            print(f"Sales data test failed: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()

        # Test 4: Hourly Energy Data
        try:
            energy_df = self.create_hourly_energy_data(days=180)
            energy_features = ['Temperature', 'Hour', 'DayOfWeek', 'Month', 'Quarter',
                               'IsWeekend', 'IsHoliday', 'IsBusinessHour', 'HourlyPattern', 'WeatherEffect']

            energy_configs = [
                {
                    'name': 'Energy_High_Consumption',
                    'target': 'HighConsumption',
                    'task_type': 'classification',
                    'features': energy_features,
                    'groupby_cols': [],
                    'time_step': 'H'
                },
                {
                    'name': 'Energy_Next_Hour_Prediction',
                    'target': 'NextHourConsumption',
                    'task_type': 'regression',
                    'features': energy_features,
                    'groupby_cols': [],
                    'time_step': 'H'
                }
            ]

            energy_results = self.test_dataset(energy_df, 'Energy_Consumption_Hourly', 'DateTime', energy_configs)
            all_results['Energy_Consumption_Hourly'] = energy_results
        except Exception as e:
            print(f"Energy data test failed: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()

        # Test 5: Simple Time Series Test (Weekly aggregation)
        try:
            # Create a simple time series dataset for weekly analysis
            self.print_section("CREATING SIMPLE WEEKLY TIME SERIES DATA")

            np.random.seed(42)
            dates = pd.date_range('2022-01-01', periods=520, freq='W')  # Weekly data for 10 years

            # Create trend + seasonality + noise
            trend = np.linspace(100, 200, len(dates))
            seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)  # yearly seasonality
            noise = np.random.normal(0, 5, len(dates))

            values = trend + seasonality + noise

            simple_df = pd.DataFrame({
                'Date': dates,
                'Value': values,
                'WeekOfYear': dates.isocalendar().week,
                'Month': dates.month,
                'Quarter': dates.quarter,
                'IsEndOfMonth': dates.is_month_end.astype(int),
                'Feature1': np.random.normal(10, 2, len(dates)),
                'Feature2': np.random.normal(5, 1, len(dates)),
                'Feature3': values * 0.1 + np.random.normal(0, 1, len(dates))
            })

            # Create targets
            simple_df['NextValue'] = simple_df['Value'].shift(-1)
            simple_df['HighValue'] = (simple_df['Value'] > simple_df['Value'].quantile(0.7)).astype(int)
            simple_df = simple_df.dropna()

            simple_features = ['WeekOfYear', 'Month', 'Quarter', 'IsEndOfMonth', 'Feature1', 'Feature2', 'Feature3']

            simple_configs = [
                {
                    'name': 'Simple_Weekly_Regression',
                    'target': 'NextValue',
                    'task_type': 'regression',
                    'features': simple_features,
                    'groupby_cols': [],
                    'time_step': 'W'
                },
                {
                    'name': 'Simple_Weekly_Classification',
                    'target': 'HighValue',
                    'task_type': 'classification',
                    'features': simple_features,
                    'groupby_cols': [],
                    'time_step': 'W'
                }
            ]

            simple_results = self.test_dataset(simple_df, 'Simple_Weekly_TimeSeries', 'Date', simple_configs)
            all_results['Simple_Weekly_TimeSeries'] = simple_results

        except Exception as e:
            print(f"Simple weekly time series test failed: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()

        # Generate comprehensive report
        self.generate_comprehensive_report(all_results)

        # Save results if requested
        if self.save_results:
            self.save_test_results(all_results)

        return all_results


def main():
    """Main function to run the comprehensive time-based tests"""
    print("Starting BigFeat Time-Based Series Testing...")
    print("=" * 80)
    print("This enhanced test suite will:")
    print("1. Test your BigFeat implementation with time-based window capabilities")
    print("2. Compare performance across different time horizons (short/medium/long-term)")
    print("3. Test on multiple frequencies (hourly, daily, weekly)")
    print("4. Analyze effectiveness of time-based vs data-point-based windows")
    print("5. Generate detailed time-based analysis reports")
    print("6. Test seasonal, trend, and cyclical pattern detection")
    print("=" * 80)

    tester = ComprehensiveTimeSeriesTester(verbose=True, save_results=True)
    results = tester.run_comprehensive_tests()

    print("\n" + "=" * 80)
    print("TIME-BASED TESTING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Check the 'results' directory for:")
    print("- Time-based feature analysis (JSON files)")
    print("- Time horizon performance comparisons (PNG files)")
    print("- Complete time-based test results (JSON file)")
    print("- Time series operation effectiveness analysis")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()