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

warnings.filterwarnings('ignore')

try:
    import yfinance as yf

    print("✓ yfinance imported successfully")
except ImportError:
    print("⚠ yfinance not found. Install with: pip install yfinance")

try:
    from bigfeat.bigfeat_base import BigFeat

    print("✓ BigFeat imported successfully")
except ImportError:
    print("✗ BigFeat not found. Please ensure bigfeat_base.py is in the correct location.")
    sys.exit(1)


class ComprehensiveTimeSeriesTester:
    """
    Comprehensive testing suite for BigFeat time-based series capabilities
    """

    def __init__(self, verbose=True, save_results=True):
        self.verbose = verbose
        self.save_results = save_results
        self.results = {}

        if self.save_results:
            os.makedirs('results', exist_ok=True)

    def print_section(self, title):
        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"{title}")
            print(f"{'=' * 80}")

    def load_stock_data(self, symbols=['AAPL', 'GOOGL', 'MSFT'], period='2y'):
        try:
            self.print_section("LOADING FINANCIAL DATA FROM YAHOO FINANCE")
            stock_data = []

            for symbol in symbols:
                if self.verbose:
                    print(f"Downloading {symbol} data...")

                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval='1d')

                if len(hist) == 0:
                    print(f"⚠ No data found for {symbol}")
                    continue

                hist = hist.reset_index()
                hist['Symbol'] = symbol

                hist['Returns'] = hist['Close'].pct_change()
                hist['LogReturns'] = np.log1p(hist['Close'].pct_change())
                hist['Volatility'] = hist['Returns'].rolling(20, min_periods=1).std()
                hist['HL_Pct'] = (hist['High'] - hist['Low']) / hist['Close']
                hist['Price_Change'] = hist['Close'] - hist['Open']
                hist['Volume_MA'] = hist['Volume'].rolling(20, min_periods=1).mean()
                hist['RSI'] = self.calculate_rsi(hist['Close'])
                hist['MACD'] = self.calculate_macd(hist['Close'])
                hist['BB_Upper'], hist['BB_Lower'] = self.calculate_bollinger_bands(hist['Close'])

                hist['Next_Return'] = hist['Returns'].shift(-1)
                hist['Price_Up'] = (hist['Next_Return'] > 0).astype(int)
                hist['High_Vol'] = (hist['Volume'] > hist['Volume'].quantile(0.75)).astype(int)
                hist['Volatility_High'] = (hist['Volatility'] > hist['Volatility'].quantile(0.75)).astype(int)

                stock_data.append(hist)

            df = pd.concat(stock_data, ignore_index=True)
            df = df.dropna()

            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = np.clip(df[col], -1e8, 1e8)

            if self.verbose:
                print(f"Stock data loaded: {len(df)} rows, {df['Symbol'].nunique()} symbols")
                print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

            return df

        except Exception as e:
            print(f"Error loading stock data: {e}")
            return None

    def load_crypto_data(self, symbols=['BTC-USD', 'ETH-USD'], period='1y'):
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

                hist['Returns'] = hist['Close'].pct_change()
                hist['LogReturns'] = np.log1p(hist['Close'].pct_change())
                hist['Volatility'] = hist['Returns'].rolling(10, min_periods=1).std()
                hist['Price_Range'] = hist['High'] - hist['Low']
                hist['Volume_USD'] = hist['Volume'] * hist['Close']

                hist['SMA_7'] = hist['Close'].rolling(7, min_periods=1).mean()
                hist['SMA_30'] = hist['Close'].rolling(30, min_periods=1).mean()
                hist['Price_Position'] = (hist['Close'] - hist['Low'].rolling(20, min_periods=1).min()) / (
                        hist['High'].rolling(20, min_periods=1).max() - hist['Low'].rolling(20, min_periods=1).min())

                hist['Next_Return'] = hist['Returns'].shift(-1)
                hist['Price_Up'] = (hist['Next_Return'] > 0).astype(int)
                hist['High_Volatility'] = (hist['Volatility'] > hist['Volatility'].quantile(0.8)).astype(int)

                crypto_data.append(hist)

            if crypto_data:
                df = pd.concat(crypto_data, ignore_index=True)
                df = df.dropna()

                df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
                for col in df.select_dtypes(include=[np.number]).columns:
                    df[col] = np.clip(df[col], -1e8, 1e8)

                if self.verbose:
                    print(f"Crypto data loaded: {len(df)} rows, {df['Symbol'].nunique()} symbols")
                    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

                return df
            else:
                return None

        except Exception as e:
            print(f"Error loading crypto data: {e}")
            return None

    def create_synthetic_sales_data(self, n_stores=5, days=730):
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
                has_promo = np.random.random() < 0.15

                base_sales = (store_base + yearly_trend + monthly_seasonal + weekly_pattern)

                if is_weekend:
                    base_sales *= 1.3
                if is_holiday:
                    base_sales *= 1.8
                if has_promo:
                    base_sales *= np.random.uniform(1.2, 1.8)

                noise = np.random.normal(0, 300)
                sales = max(100, base_sales + noise)

                customers = int(sales / np.random.uniform(15, 25))
                avg_transaction = sales / max(customers, 1)

                data_list.append({
                    'Date': date,
                    'Store': store_id,
                    'Sales': sales,
                    'Customers': customers,
                    'AvgTransaction': avg_transaction,
                    'HasPromo': int(has_promo),
                    'IsWeekend': int(is_weekend),
                    'IsHoliday': int(is_holiday),
                    'DayOfWeek': date.weekday(),
                    'Month': date.month,
                    'Quarter': date.quarter,
                })

        df = pd.DataFrame(data_list)

        df['HighSales'] = (df['Sales'] > df['Sales'].quantile(0.75)).astype(int)
        df['NextDaySales'] = df.groupby('Store')['Sales'].shift(-1)

        df = df.dropna()

        if self.verbose:
            print(f"Synthetic sales data created: {len(df)} rows, {df['Store'].nunique()} stores")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

        return df

    def create_hourly_energy_data(self, days=180):
        self.print_section("CREATING HOURLY ENERGY CONSUMPTION DATA")

        np.random.seed(42)
        start_date = pd.to_datetime('2023-01-01')
        dates = pd.date_range(start_date, periods=days * 24, freq='H')

        data_list = []

        for i, timestamp in enumerate(dates):
            base_consumption = 1000

            hourly_pattern = 300 * np.sin(2 * np.pi * (timestamp.hour - 6) / 24)
            weekly_pattern = 200 * np.sin(2 * np.pi * timestamp.weekday() / 7)
            seasonal_pattern = 400 * np.sin(2 * np.pi * (timestamp.dayofyear - 80) / 365)

            temp = 20 + 15 * np.sin(2 * np.pi * (timestamp.dayofyear - 80) / 365) + np.random.normal(0, 3)
            weather_effect = 0
            if temp < 10:
                weather_effect = (10 - temp) * 50
            elif temp > 25:
                weather_effect = (temp - 25) * 40

            is_holiday = (timestamp.month == 12 and timestamp.day >= 20) or (
                    timestamp.month == 1 and timestamp.day <= 5)
            holiday_effect = -200 if is_holiday else 0

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

        df['NextHourConsumption'] = df['Consumption'].shift(-1)
        df['HighConsumption'] = (df['Consumption'] > df['Consumption'].quantile(0.8)).astype(int)
        df['ConsumptionChange'] = df['Consumption'].diff()

        df = df.dropna()

        if self.verbose:
            print(f"Energy data created: {len(df)} rows")
            print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")

        return df

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_macd(self, prices, fast=12, slow=26):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        sma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.fillna(sma), lower_band.fillna(sma)

    def parse_operation_chain(self, ops_list, ids_list, feature_cols):
        """Parse operation chains and create clear descriptions"""
        if not ops_list or len(ops_list) == 0:
            if ids_list and len(ids_list) > 0:
                if isinstance(ids_list[0], (int, np.integer)) and ids_list[0] < len(feature_cols):
                    return f"Original[{feature_cols[ids_list[0]]}]", "original"
                else:
                    return f"Original[feature_{ids_list[0]}]", "original"
            else:
                return "Original[unknown]", "original"

        operation_chain = []
        feature_references = []

        if ids_list:
            for idx in ids_list:
                if isinstance(idx, (int, np.integer)) and idx < len(feature_cols):
                    feature_references.append(feature_cols[idx])
                else:
                    feature_references.append(f"feature_{idx}")

        for i, op_info in enumerate(ops_list):
            if len(op_info) >= 2:
                op_func = op_info[0]
                depth = op_info[1]
                op_name = self.get_readable_operation_name(op_func)
                operation_chain.append({
                    'operation': op_name,
                    'depth': depth,
                    'index': i
                })

        if len(operation_chain) == 0:
            if feature_references:
                return f"Original[{feature_references[0]}]", "original"
            else:
                return "Original[unknown]", "original"

        description = self.build_operation_description(operation_chain, feature_references)

        ts_ops = ['rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max', 'rolling_median',
                  'rolling_sum', 'lag', 'diff', 'pct_change', 'ewm', 'momentum',
                  'seasonal_decompose', 'trend', 'weekday_mean', 'month_mean']

        is_time_series = any(op['operation'] in ts_ops for op in operation_chain)

        return description, "time_series" if is_time_series else "standard"

    def get_readable_operation_name(self, op_func):
        """Convert operation function to readable name"""
        if hasattr(op_func, '__name__'):
            name = op_func.__name__
        else:
            name = str(op_func)

        name_mapping = {
            'multiply': 'multiply',
            'add': 'add',
            'subtract': 'subtract',
            'abs': 'abs',
            'square': 'square',
            'original_feat': 'identity',
            '_safe_rolling_mean': 'rolling_mean',
            '_safe_rolling_std': 'rolling_std',
            '_safe_rolling_min': 'rolling_min',
            '_safe_rolling_max': 'rolling_max',
            '_safe_rolling_median': 'rolling_median',
            '_safe_rolling_sum': 'rolling_sum',
            '_safe_lag_feature': 'lag',
            '_safe_diff_feature': 'diff',
            '_safe_pct_change': 'pct_change',
            '_safe_ewm': 'ewm',
            '_safe_momentum': 'momentum',
            '_safe_seasonal_decompose': 'seasonal_decompose',
            '_safe_trend_feature': 'trend',
            '_safe_weekday_mean': 'weekday_mean',
            '_safe_month_mean': 'month_mean'
        }

        clean_name = name.replace('<built-in function ', '').replace('>', '').replace('<ufunc ', '').replace("'", "")

        return name_mapping.get(clean_name, clean_name)

    def build_operation_description(self, operation_chain, feature_references):
        """Build a clear nested description of the operation chain"""
        if len(operation_chain) == 0:
            return f"Original[{feature_references[0] if feature_references else 'unknown'}]"

        sorted_ops = sorted(operation_chain, key=lambda x: x['depth'], reverse=True)
        base_features = feature_references.copy() if feature_references else ['unknown']

        if len(sorted_ops) == 1:
            op = sorted_ops[0]
            if op['operation'] in ['multiply', 'add', 'subtract'] and len(base_features) >= 2:
                return f"{op['operation']}({base_features[0]}, {base_features[1]})"
            else:
                return f"{op['operation']}({base_features[0] if base_features else 'unknown'})"

        current_expr = base_features[0] if base_features else 'unknown'

        for i, op in enumerate(sorted_ops):
            op_name = op['operation']

            if op_name in ['multiply', 'add', 'subtract']:
                if i == 0 and len(base_features) >= 2:
                    current_expr = f"{op_name}({base_features[0]}, {base_features[1]})"
                else:
                    if len(base_features) > i + 1:
                        current_expr = f"{op_name}({current_expr}, {base_features[i + 1]})"
                    else:
                        current_expr = f"{op_name}({current_expr}, {current_expr})"
            else:
                current_expr = f"{op_name}({current_expr})"

        return current_expr

    def generate_feature_descriptions(self, bigfeat, feature_cols, X_enhanced_shape):
        """Generate detailed feature descriptions"""
        feature_info = {
            'feature_names': [],
            'feature_descriptions': [],
            'feature_types': [],
            'complexity_scores': []
        }

        if hasattr(bigfeat, 'tracking_ops') and hasattr(bigfeat, 'tracking_ids'):
            for i, (ops, ids) in enumerate(zip(bigfeat.tracking_ops, bigfeat.tracking_ids)):
                description, feature_type = self.parse_operation_chain(ops, ids, feature_cols)
                complexity = len(ops) if ops else 0
                feat_name = description if feature_type == "original" else f"Generated_F{i:03d}"

                feature_info['feature_names'].append(feat_name)
                feature_info['feature_descriptions'].append(description)
                feature_info['feature_types'].append(feature_type)
                feature_info['complexity_scores'].append(complexity)

        n_generated = len(feature_info['feature_names'])
        for i in range(n_generated, X_enhanced_shape):
            orig_idx = i - n_generated
            if orig_idx < len(feature_cols):
                feat_name = f"Original[{feature_cols[orig_idx]}]"
                description = f"Original feature: {feature_cols[orig_idx]}"
            else:
                feat_name = f"Original[feature_{orig_idx}]"
                description = f"Original feature {orig_idx}"

            feature_info['feature_names'].append(feat_name)
            feature_info['feature_descriptions'].append(description)
            feature_info['feature_types'].append("original")
            feature_info['complexity_scores'].append(0)

        return feature_info

    def print_feature_analysis(self, feature_info, top_n=10):
        """Print formatted feature analysis"""
        print(f"\n  FEATURE GENERATION ANALYSIS")
        print(f"  {'=' * 60}")

        original_count = sum(1 for t in feature_info['feature_types'] if t == 'original')
        standard_count = sum(1 for t in feature_info['feature_types'] if t == 'standard')
        ts_count = sum(1 for t in feature_info['feature_types'] if t == 'time_series')

        print(f"  Total Features: {len(feature_info['feature_names'])}")
        print(f"  Original features: {original_count}")
        print(f"  Standard generated: {standard_count}")
        print(f"  Time-series generated: {ts_count}")

        complexities = feature_info['complexity_scores']
        if complexities and max(complexities) > 0:
            print(f"  Average complexity: {np.mean([c for c in complexities if c > 0]):.1f} operations")
            print(f"  Max complexity: {max(complexities)} operations")

        if standard_count > 0 or ts_count > 0:
            print(f"\n  MOST COMPLEX GENERATED FEATURES (Top {min(top_n, standard_count + ts_count)}):")
            print(f"  {'-' * 60}")

            generated_features = [
                (i, desc, ftype, comp)
                for i, (desc, ftype, comp) in enumerate(zip(
                    feature_info['feature_descriptions'],
                    feature_info['feature_types'],
                    feature_info['complexity_scores']
                ))
                if ftype != 'original' and comp > 0
            ]

            generated_features.sort(key=lambda x: x[3], reverse=True)

            for idx, (i, desc, ftype, comp) in enumerate(generated_features[:top_n]):
                ts_marker = "TS" if ftype == 'time_series' else "ST"
                print(f"  {idx + 1:2d}. [{ts_marker}] {desc}")
                print(f"      Complexity: {comp} operations, Type: {ftype}")
                if idx < len(generated_features) - 1:
                    print()

    def test_dataset(self, df, dataset_name, date_col, target_configs):
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

            print(f"\n{'-' * 60}")
            print(f"Testing: {config_name} ({task_type})")
            print(f"Target: {target_col}")
            print(f"Features: {len(feature_cols)}")

            try:
                all_cols = feature_cols + [date_col] + groupby_cols
                X_full_df = df[all_cols].copy()
                y = df[target_col].copy()

                X_full_df[feature_cols] = X_full_df[feature_cols].fillna(X_full_df[feature_cols].mean())
                y = y.fillna(y.mean() if task_type == 'regression' else y.mode().iloc[0])

                split_date = df[date_col].quantile(0.8)
                train_mask = df[date_col] <= split_date
                test_mask = df[date_col] > split_date

                X_train_full_df = X_full_df[train_mask].reset_index(drop=True)
                X_test_full_df = X_full_df[test_mask].reset_index(drop=True)
                y_train = y[train_mask].values
                y_test = y[test_mask].values

                if len(X_train_full_df) < 100 or len(X_test_full_df) < 20:
                    print(f"⚠ Insufficient data: train={len(X_train_full_df)}, test={len(X_test_full_df)}")
                    continue

                print(f"Data split: train={len(X_train_full_df)}, test={len(X_test_full_df)}")

                if task_type == 'classification':
                    unique_classes = len(np.unique(y_train))
                    print(f"Classes: {unique_classes}")
                else:
                    print(f"Target stats: mean={y_train.mean():.4f}, std={y_train.std():.4f}")

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
                            'estimator': 'avg',
                            'selection': 'stability'
                        }
                    },
                    {
                        'name': 'Auto-Detection',
                        'params': {
                            'task_type': task_type,
                            'enable_time_series': True,
                            'datetime_col': date_col,
                            'groupby_cols': groupby_cols,
                            'verbose': False,
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
                        'name': 'Time Series (Short)',
                        'params': {
                            'task_type': task_type,
                            'enable_time_series': True,
                            'datetime_col': date_col,
                            'groupby_cols': groupby_cols,
                            'window_sizes': ['7D', '14D', '30D'],
                            'lag_periods': ['1D', '7D', '14D'],
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
                        'name': 'Time Series (Long)',
                        'params': {
                            'task_type': task_type,
                            'enable_time_series': True,
                            'datetime_col': date_col,
                            'groupby_cols': groupby_cols,
                            'window_sizes': ['30D', '60D', '90D'],
                            'lag_periods': ['7D', '30D', '60D'],
                            'verbose': False
                        },
                        'fit_params': {
                            'gen_size': 4,
                            'iterations': 3,
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

                        bigfeat = BigFeat(**config['params'])

                        X_train_enhanced = bigfeat.fit(X_train_full_df, y_train, **config['fit_params'])
                        X_test_enhanced = bigfeat.transform(X_test_full_df)

                        ts_enabled = getattr(bigfeat, 'enable_time_series', False)

                        print(f"    Time Series Enabled: {ts_enabled}")
                        print(f"    Generated features shape: {X_train_enhanced.shape}")

                        # Generate feature descriptions
                        feature_info = self.generate_feature_descriptions(
                            bigfeat, feature_cols, X_train_enhanced.shape[1]
                        )

                        ts_ops_count = sum(1 for ftype in feature_info['feature_types'] if ftype == 'time_series')
                        total_generated = sum(1 for ftype in feature_info['feature_types'] if ftype != 'original')
                        ts_percentage = (ts_ops_count / total_generated * 100) if total_generated > 0 else 0

                        print(f"    Time series features: {ts_ops_count}")
                        print(f"    TS feature percentage: {ts_percentage:.1f}%")

                        if config['params'].get('enable_time_series', False):
                            self.print_feature_analysis(feature_info, top_n=5)

                        estimator_names = ['rf', 'dt'] if task_type == 'classification' else ['rf_reg', 'dt_reg']
                        best_model = bigfeat.select_estimator(X_train_enhanced, y_train, estimator_names)

                        y_pred = best_model.predict(X_test_enhanced)

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
                            'ts_enabled': ts_enabled,
                            'model': type(best_model).__name__,
                            'feature_count': len(feature_cols),
                            'generated_features': X_train_enhanced.shape[1] - len(feature_cols),
                            'ts_ops_count': ts_ops_count,
                            'ts_percentage': ts_percentage,
                            'feature_info': feature_info
                        }

                        if task_type == 'regression':
                            config_results[config['name']]['mae'] = mae
                            config_results[config['name']]['rmse'] = rmse

                        print(f"    {metric_name}: {score:.4f}, Features: {X_train_enhanced.shape[1]}")

                        # Save feature information
                        if self.save_results and config['name'] != 'Baseline (No Time Series)':
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            feature_file = f'results/{dataset_name}_{config_name}_{config["name"].replace(" ", "_")}_features_{timestamp}.json'

                            feature_details = {
                                'dataset': dataset_name,
                                'config': config_name,
                                'method': config['name'],
                                'feature_names': feature_info['feature_names'],
                                'feature_descriptions': feature_info['feature_descriptions'],
                                'feature_types': feature_info['feature_types'],
                                'complexity_scores': feature_info['complexity_scores'],
                                'total_features': X_train_enhanced.shape[1],
                                'ts_features': ts_ops_count,
                                'ts_percentage': ts_percentage,
                                'performance': {
                                    'score': float(score),
                                    'metric': metric_name
                                }
                            }

                            if task_type == 'regression':
                                feature_details['performance']['mae'] = float(mae)
                                feature_details['performance']['rmse'] = float(rmse)

                            with open(feature_file, 'w') as f:
                                json.dump(feature_details, f, indent=2, default=str)

                    except Exception as e:
                        print(f"    Error: {str(e)}")
                        config_results[config['name']] = {'error': str(e), 'score': 0}

                baseline_score = config_results.get('Baseline (No Time Series)', {}).get('score', 0)

                print(f"\n  Results Summary for {config_name}:")
                print(f"  {'-' * 80}")
                print(f"  {'Method':<30} | {metric_name:<8} | {'Features':<8} | {'TS?':<4} | {'Improvement':<11}")
                print(f"  {'-' * 80}")

                for name, result in config_results.items():
                    if 'error' not in result:
                        improvement = result['score'] - baseline_score
                        ts_status = "Yes" if result.get('ts_enabled', False) else "No"

                        line = f"  {name:<30} | {result['score']:8.4f} | {result.get('n_features', 0):8d} | {ts_status:<4} | {improvement:+11.4f}"

                        if task_type == 'regression':
                            line += f" | MAE: {result.get('mae', 0):8.4f}"
                        print(line)
                    else:
                        print(f"  {name:<30} | ERROR: {result['error'][:50]}...")

                results[config_name] = config_results

            except Exception as e:
                print(f"Error testing {config_name}: {str(e)}")
                results[config_name] = {'error': str(e)}

        return results

    def generate_comprehensive_report(self, all_results):
        self.print_section("COMPREHENSIVE TEST RESULTS SUMMARY")

        total_tests = 0
        successful_tests = 0
        improvements = []
        best_improvements = {}

        for dataset_name, dataset_results in all_results.items():
            print(f"\n{dataset_name}:")
            print("-" * 70)

            for config_name, config_results in dataset_results.items():
                if isinstance(config_results, dict) and 'error' not in config_results:

                    baseline_score = config_results.get('Baseline (No Time Series)', {}).get('score', 0)

                    print(f"\n  {config_name}:")

                    for method_name, method_result in config_results.items():
                        if isinstance(method_result, dict) and 'error' not in method_result:
                            score = method_result.get('score', 0)
                            improvement = score - baseline_score

                            improvements.append(improvement)
                            total_tests += 1

                            if improvement > 0:
                                successful_tests += 1

                            key = f"{dataset_name}_{config_name}"
                            if key not in best_improvements or improvement > best_improvements[key]['improvement']:
                                best_improvements[key] = {
                                    'improvement': improvement,
                                    'method': method_name,
                                    'score': score,
                                    'dataset': dataset_name,
                                    'config': config_name,
                                    'n_features': method_result.get('n_features', 0)
                                }

        print(f"\n{'=' * 80}")
        print("OVERALL STATISTICS")
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
            print(f"Median improvement: {np.median(improvements):+.4f}")

            print(f"\nBest Improvements by Dataset/Task:")
            print(f"{'-' * 80}")
            print(f"{'Dataset':<20} | {'Task':<25} | {'Method':<20} | {'Improvement':<11}")
            print(f"{'-' * 80}")

            for key, result in sorted(best_improvements.items(), key=lambda x: x[1]['improvement'], reverse=True):
                print(
                    f"{result['dataset'][:20]:<20} | {result['config'][:25]:<25} | {result['method'][:20]:<20} | {result['improvement']:+11.4f}")

        else:
            print("No valid test results to analyze.")

        return improvements, best_improvements

    def plot_results(self, all_results, improvements, best_improvements):
        try:
            plt.figure(figsize=(16, 10))

            plt.subplot(2, 3, 1)
            if improvements:
                plt.hist(improvements, bins=20, edgecolor='black', alpha=0.7)
                plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
                plt.xlabel('Improvement (Score - Baseline)')
                plt.ylabel('Frequency')
                plt.title('Distribution of Performance Improvements')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.subplot(2, 3, 2)
            if best_improvements:
                datasets_tasks = [f"{r['dataset'][:12]}\n{r['config'][:12]}" for r in
                                  list(best_improvements.values())[:8]]
                scores = [r['improvement'] for r in list(best_improvements.values())[:8]]
                colors = ['green' if s > 0 else 'red' for s in scores]
                plt.barh(range(len(datasets_tasks)), scores, color=colors, alpha=0.7)
                plt.yticks(range(len(datasets_tasks)), datasets_tasks, fontsize=9)
                plt.xlabel('Improvement')
                plt.title('Best Improvements by Dataset/Task')
                plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                plt.grid(True, alpha=0.3, axis='x')

            plt.subplot(2, 3, 3)
            method_scores = {}
            for result in best_improvements.values():
                method = result['method']
                if method not in method_scores:
                    method_scores[method] = []
                method_scores[method].append(result['improvement'])

            if method_scores:
                methods = list(method_scores.keys())
                avg_improvements = [np.mean(method_scores[m]) for m in methods]
                colors = ['green' if x > 0 else 'red' for x in avg_improvements]
                plt.bar(range(len(methods)), avg_improvements, color=colors, alpha=0.7, edgecolor='black')
                plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=9)
                plt.ylabel('Average Improvement')
                plt.title('Average Improvement by Method')
                plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                plt.grid(True, alpha=0.3, axis='y')

            plt.subplot(2, 3, 4)
            if best_improvements:
                all_improvements = [r['improvement'] for r in best_improvements.values()]
                plt.boxplot([all_improvements], labels=['All Methods'])
                plt.ylabel('Improvement')
                plt.title('Improvement Statistics')
                plt.grid(True, alpha=0.3, axis='y')
                plt.axhline(y=0, color='red', linestyle='--', linewidth=1)

            plt.subplot(2, 3, 5)
            if best_improvements:
                feature_counts = [r['n_features'] for r in best_improvements.values()]
                improvements_list = [r['improvement'] for r in best_improvements.values()]
                plt.scatter(feature_counts, improvements_list, alpha=0.6, s=100)
                plt.xlabel('Number of Features')
                plt.ylabel('Improvement')
                plt.title('Feature Count vs Performance Improvement')
                plt.grid(True, alpha=0.3)

            plt.subplot(2, 3, 6)
            plt.axis('off')

            if improvements:
                stats_text = f"""
TEST SUMMARY STATISTICS

Total Tests: {len(improvements)}
Positive Improvements: {len([x for x in improvements if x > 0])}
Success Rate: {len([x for x in improvements if x > 0]) / len(improvements) * 100:.1f}%

PERFORMANCE METRICS
Average Improvement: {np.mean(improvements):+.4f}
Median Improvement: {np.median(improvements):+.4f}
Best Improvement: {np.max(improvements):+.4f}
Worst Improvement: {np.min(improvements):+.4f}
Std Dev: {np.std(improvements):.4f}

DATASETS TESTED
Number of Configurations: {len(best_improvements)}
                """
                plt.text(0.1, 0.95, stats_text, transform=plt.gca().transAxes,
                         verticalalignment='top', fontsize=10, fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = f'results/test_results_visualization_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\nVisualization saved to {plot_file}")
            return plot_file

        except Exception as e:
            print(f"Error generating plots: {e}")
            return None

    def run_comprehensive_tests(self):
        self.print_section("BIGFEAT TIME-BASED SERIES TESTING SUITE")
        print("Testing BigFeat's time series capabilities")
        print("This test may take 15-20 minutes to complete...")

        all_results = {}

        # Test 1: Stock Market Data
        try:
            stock_df = self.load_stock_data(['AAPL', 'GOOGL', 'MSFT'], period='2y')
            if stock_df is not None:
                stock_features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Volatility',
                                  'HL_Pct', 'Volume_MA', 'RSI', 'MACD']

                stock_configs = [
                    {
                        'name': 'Stock_Direction_Prediction',
                        'target': 'Price_Up',
                        'task_type': 'classification',
                        'features': stock_features,
                        'groupby_cols': ['Symbol']
                    },
                    {
                        'name': 'Stock_Return_Prediction',
                        'target': 'Next_Return',
                        'task_type': 'regression',
                        'features': stock_features,
                        'groupby_cols': ['Symbol']
                    }
                ]

                stock_results = self.test_dataset(stock_df, 'Stock_Market', 'Date', stock_configs)
                all_results['Stock_Market'] = stock_results
        except Exception as e:
            print(f"Stock data test failed: {e}")

        # Test 2: Cryptocurrency Data
        try:
            crypto_df = self.load_crypto_data(['BTC-USD', 'ETH-USD'], period='1y')
            if crypto_df is not None:
                crypto_features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Volatility',
                                   'Price_Range', 'SMA_7', 'SMA_30', 'Price_Position']

                crypto_configs = [
                    {
                        'name': 'Crypto_Direction_Prediction',
                        'target': 'Price_Up',
                        'task_type': 'classification',
                        'features': crypto_features,
                        'groupby_cols': ['Symbol']
                    },
                    {
                        'name': 'Crypto_Return_Prediction',
                        'target': 'Next_Return',
                        'task_type': 'regression',
                        'features': crypto_features,
                        'groupby_cols': ['Symbol']
                    }
                ]

                crypto_results = self.test_dataset(crypto_df, 'Cryptocurrency', 'Date', crypto_configs)
                all_results['Cryptocurrency'] = crypto_results
        except Exception as e:
            print(f"Crypto data test failed: {e}")

        # Test 3: Synthetic Sales Data
        try:
            sales_df = self.create_synthetic_sales_data(n_stores=5, days=730)
            sales_features = ['Customers', 'AvgTransaction', 'HasPromo', 'IsWeekend',
                              'DayOfWeek', 'Month', 'Quarter']

            sales_configs = [
                {
                    'name': 'Sales_High_Performance',
                    'target': 'HighSales',
                    'task_type': 'classification',
                    'features': sales_features,
                    'groupby_cols': ['Store']
                },
                {
                    'name': 'Sales_Next_Day_Prediction',
                    'target': 'NextDaySales',
                    'task_type': 'regression',
                    'features': sales_features,
                    'groupby_cols': ['Store']
                }
            ]

            sales_results = self.test_dataset(sales_df, 'Retail_Sales', 'Date', sales_configs)
            all_results['Retail_Sales'] = sales_results
        except Exception as e:
            print(f"Sales data test failed: {e}")

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
                    'groupby_cols': []
                },
                {
                    'name': 'Energy_Next_Hour_Prediction',
                    'target': 'NextHourConsumption',
                    'task_type': 'regression',
                    'features': energy_features,
                    'groupby_cols': []
                }
            ]

            energy_results = self.test_dataset(energy_df, 'Energy_Consumption', 'DateTime', energy_configs)
            all_results['Energy_Consumption'] = energy_results
        except Exception as e:
            print(f"Energy data test failed: {e}")

        # Generate comprehensive report
        improvements, best_improvements = self.generate_comprehensive_report(all_results)

        # Generate plots
        self.plot_results(all_results, improvements, best_improvements)

        # Save results
        if self.save_results:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f'results/test_results_{timestamp}.json'

                results_to_save = {}
                for dataset_name, dataset_results in all_results.items():
                    results_to_save[dataset_name] = {}
                    for config_name, config_results in dataset_results.items():
                        results_to_save[dataset_name][config_name] = {}
                        for method_name, method_result in config_results.items():
                            cleaned_result = {}
                            for k, v in method_result.items():
                                if isinstance(v, (np.floating, np.integer)):
                                    cleaned_result[k] = float(v)
                                else:
                                    cleaned_result[k] = v
                            results_to_save[dataset_name][config_name][method_name] = cleaned_result

                with open(output_file, 'w') as f:
                    json.dump(results_to_save, f, indent=2, default=str)

                print(f"\nTest results saved to {output_file}")

                # Save summary statistics
                summary_file = f'results/test_summary_{timestamp}.json'

                summary_stats = {
                    'timestamp': timestamp,
                    'total_tests': len(improvements) if improvements else 0,
                    'positive_improvements': len([x for x in improvements if x > 0]) if improvements else 0,
                    'success_rate': len([x for x in improvements if x > 0]) / len(
                        improvements) * 100 if improvements else 0,
                    'average_improvement': float(np.mean(improvements)) if improvements else 0,
                    'median_improvement': float(np.median(improvements)) if improvements else 0,
                    'best_improvement': float(np.max(improvements)) if improvements else 0,
                    'worst_improvement': float(np.min(improvements)) if improvements else 0,
                    'std_dev': float(np.std(improvements)) if improvements else 0,
                    'datasets_tested': len(all_results),
                    'configurations_tested': len(best_improvements)
                }

                with open(summary_file, 'w') as f:
                    json.dump(summary_stats, f, indent=2)

                print(f"Test summary saved to {summary_file}")

                # Save detailed results for each method
                for dataset_name, dataset_results in all_results.items():
                    for config_name, config_results in dataset_results.items():
                        if isinstance(config_results, dict) and 'error' not in config_results:
                            method_file = f'results/{dataset_name}_{config_name}_detailed_{timestamp}.json'

                            method_details = {
                                'dataset': dataset_name,
                                'config': config_name,
                                'timestamp': timestamp,
                                'results': {}
                            }

                            for method_name, method_result in config_results.items():
                                if isinstance(method_result, dict) and 'error' not in method_result:
                                    cleaned = {}
                                    for k, v in method_result.items():
                                        if k == 'feature_info':
                                            cleaned[k] = {
                                                'feature_names': v.get('feature_names', []),
                                                'feature_descriptions': v.get('feature_descriptions', []),
                                                'feature_types': v.get('feature_types', []),
                                                'complexity_scores': v.get('complexity_scores', [])
                                            }
                                        elif isinstance(v, (np.floating, np.integer)):
                                            cleaned[k] = float(v)
                                        else:
                                            cleaned[k] = v
                                    method_details['results'][method_name] = cleaned

                            with open(method_file, 'w') as f:
                                json.dump(method_details, f, indent=2, default=str)

            except Exception as e:
                print(f"Error saving results: {e}")

        return all_results


def main():
    """Main function to run the comprehensive tests"""
    print("Starting BigFeat Time-Based Series Testing...")
    print("=" * 80)
    print("This test will:")
    print("1. Load real financial data (stocks, crypto) and create synthetic datasets")
    print("2. Test BigFeat with multiple time series configurations")
    print("3. Compare baseline vs time series enabled models")
    print("4. Generate detailed JSON reports for each test")
    print("5. Create visualization plots of results")
    print("=" * 80)

    tester = ComprehensiveTimeSeriesTester(verbose=True, save_results=True)
    results = tester.run_comprehensive_tests()

    print("\n" + "=" * 80)
    print("TESTING COMPLETED!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("- test_results_<timestamp>.json: Main results file with all test data")
    print("- test_summary_<timestamp>.json: Summary statistics")
    print("- <dataset>_<config>_detailed_<timestamp>.json: Per-dataset detailed results")
    print("- test_results_visualization_<timestamp>.png: Visualization plots")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()