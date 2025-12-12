"""
BigFeat Time Series Comprehensive Benchmark
Evaluation script that tests BigFeat's automated feature engineering on the same
29 datasets used in AutoGluon-TimeSeries paper.

Compares:
1. Baseline model (without feature engineering)
2. BigFeat with all combinations of:
   - enable_time_series: 'auto', 'yes', 'no'
   - window_detector: 'dft', 'acf', 'lomb_scargle'

Features:
- Native dynamic downsampling (adapts to available RAM automatically)
- OOM protection for large datasets
- Phase separation: Learn on sample, apply to all

Measures:
- Predictive performance (MAE, RMSE, R2, MASE)
- Runtime (wall time)
- CPU usage (CPU seconds, CPU percent)
- Memory usage (peak memory, memory delta)
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import time
import psutil
import os
from datetime import datetime
from gluonts.dataset.repository import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import sys

# Add BigFeat to path if needed
# sys.path.append('/path/to/bigfeat')
from bigfeat.bigfeat_base import BigFeat

warnings.filterwarnings('ignore')


class ResourceMonitor:
    """Monitor CPU and memory usage during execution."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_cpu_times = None
        self.start_memory = None

    def start(self):
        """Start monitoring resources."""
        self.start_time = time.time()
        self.start_cpu_times = self.process.cpu_times()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        end_time = time.time()
        end_cpu_times = self.process.cpu_times()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Calculate metrics
        wall_time = end_time - self.start_time
        cpu_time = (end_cpu_times.user - self.start_cpu_times.user +
                   end_cpu_times.system - self.start_cpu_times.system)
        cpu_percent = (cpu_time / wall_time * 100) if wall_time > 0 else 0
        memory_delta = end_memory - self.start_memory

        return {
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'cpu_percent': cpu_percent,
            'memory_start_mb': self.start_memory,
            'memory_end_mb': end_memory,
            'memory_delta_mb': memory_delta,
            'memory_peak_mb': end_memory
        }


class TimeSeriesBenchmark:
    """Benchmark suite for evaluating BigFeat on time series forecasting datasets."""

    def __init__(self,
                 output_dir: str = "./benchmark_results",
                 time_limit_per_dataset: int = 3600,
                 random_seeds: List[int] = None,
                 verbose: bool = True):
        """
        Initialize benchmark suite.

        Parameters:
        -----------
        output_dir : str
            Directory to save results
        time_limit_per_dataset : int
            Time limit in seconds for each dataset
        random_seeds : list
            Random seeds for reproducibility (default: [42])
        verbose : bool
            Whether to print progress
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.time_limit = time_limit_per_dataset
        self.random_seeds = random_seeds or [42]
        self.verbose = verbose

        # Datasets from AutoGluon paper
        self.datasets = [
            "car_parts_without_missing",
            "cif_2016",
            "covid_deaths",
            #"electricity_hourly",
            "electricity_weekly",
            "fred_md",
            "hospital",
            "kdd_cup_2018_without_missing",
            "m1_monthly",
            "m1_quarterly",
            "m1_yearly",
            "m3_monthly",
            "m3_other",
            "m3_quarterly",
            "m3_yearly",
            "m4_daily",
            "m4_hourly",
            "m4_monthly",
            "m4_quarterly",
            "m4_weekly",
            "m4_yearly",
            "nn5_daily_without_missing",
            "nn5_weekly",
            "pedestrian_counts",
            "tourism_monthly",
            "tourism_quarterly",
            "tourism_yearly",
            "vehicle_trips_without_missing",
            "kaggle_web_traffic_weekly",
        ]

        # Estimators to test
        self.estimators = {
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }

        # BigFeat configurations to test
        # Note: For 'no' mode, detector doesn't matter, so we only test once
        self.bigfeat_configs = [
            ('auto', 'dft'),
            ('auto', 'acf'),
            ('auto', 'lomb_scargle'),
            ('yes', 'dft'),
            ('yes', 'acf'),
            ('yes', 'lomb_scargle'),
            ('no', 'dft'),  # Detector irrelevant for 'no' mode
        ]

    def load_and_prepare_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load dataset from GluonTS and prepare for supervised learning.

        Parameters:
        -----------
        dataset_name : str
            Name of the dataset

        Returns:
        --------
        df : DataFrame
            Prepared dataframe with item_id, timestamp, target columns
        metadata : dict
            Dataset metadata
        """
        if self.verbose:
            print(f"\nLoading {dataset_name}...")

        try:
            dataset = get_dataset(dataset_name, regenerate=False)

            # Extract metadata
            metadata = {
                'name': dataset_name,
                'prediction_length': int(dataset.metadata.prediction_length),
                'freq': dataset.metadata.freq,
                'num_train_series': len(list(dataset.train)),
                'num_test_series': len(list(dataset.test))
            }

            # Convert to long format DataFrame
            rows = []
            for i, entry in enumerate(dataset.train):
                item_id = entry.get("item_id", f"item_{i}")
                target = entry["target"]
                start = entry["start"]

                timestamps = pd.date_range(
                    start=start.to_timestamp(),
                    periods=len(target),
                    freq=dataset.metadata.freq
                )

                for timestamp, value in zip(timestamps, target):
                    rows.append({
                        "item_id": item_id,
                        "timestamp": timestamp,
                        "target": value
                    })

            df = pd.DataFrame(rows)

            # Calculate statistics
            metadata['min_length'] = df.groupby('item_id').size().min()
            metadata['max_length'] = df.groupby('item_id').size().max()
            metadata['mean_length'] = df.groupby('item_id').size().mean()
            metadata['total_timesteps'] = len(df)

            if self.verbose:
                print(f"  ✓ Loaded {len(df)} timesteps across {metadata['num_train_series']} series")
                print(f"  Frequency: {metadata['freq']}, Prediction length: {metadata['prediction_length']}")

            return df, metadata

        except Exception as e:
            if self.verbose:
                print(f"  ✗ Failed to load {dataset_name}: {str(e)}")
            raise

    def create_supervised_dataset(self,
                                  df: pd.DataFrame,
                                  prediction_length: int,
                                  test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert time series to supervised learning format.

        Creates train/test split where test set contains the last prediction_length
        timesteps for each series (simulating forecasting scenario).

        Parameters:
        -----------
        df : DataFrame
            Time series data
        prediction_length : int
            Number of steps to forecast
        test_size : float
            Proportion of series to use for testing

        Returns:
        --------
        train_df : DataFrame
            Training data
        test_df : DataFrame
            Test data (last prediction_length steps per series)
        """
        train_rows = []
        test_rows = []

        for item_id, group in df.groupby('item_id'):
            group = group.sort_values('timestamp').reset_index(drop=True)

            # Split each series: train on all but last prediction_length steps
            if len(group) > prediction_length:
                train_data = group.iloc[:-prediction_length]
                test_data = group.iloc[-prediction_length:]

                train_rows.append(train_data)
                test_rows.append(test_data)

        train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
        test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()

        return train_df, test_df

    def prepare_features_target(self,
                                df: pd.DataFrame,
                                datetime_col: str = 'timestamp',
                                target_col: str = 'target') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features and target from dataframe.

        CRITICAL FIX: Adds Lag-1 of target so BigFeat has signal history.
        This enables BigFeat to create autoregressive features like rolling means.

        Parameters:
        -----------
        df : DataFrame
            Input data
        datetime_col : str
            Name of datetime column
        target_col : str
            Name of target column

        Returns:
        --------
        X : DataFrame
            Features (excluding target but INCLUDING target_lag_1)
        y : ndarray
            Target values
        """
        df = df.copy()

        # 1. Create Lag-1 Feature (Autoregression base)
        # This allows BigFeat to create features like "Lag1 * month_sin"
        df['target_lag_1'] = df.groupby('item_id')[target_col].shift(1)

        # 2. Drop NaNs created by shifting (first row of each series)
        df = df.dropna(subset=['target_lag_1'])

        # 3. Separate target
        y = df[target_col].values

        # 4. Create features dataframe (drop target, keep lag)
        X = df.drop(columns=[target_col], errors='ignore')

        return X, y

    def calculate_series_mase(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              X_test: pd.DataFrame,
                              y_train: np.ndarray,
                              X_train: pd.DataFrame) -> float:
        """
        Calculate MASE correctly by computing naive error per series.

        CRITICAL FIX: The naive error must be calculated per series to avoid
        treating the jump from Series A to Series B as a huge error.

        Parameters:
        -----------
        y_true : np.ndarray
            True test values
        y_pred : np.ndarray
            Predicted test values
        X_test : pd.DataFrame
            Test features (contains item_id)
        y_train : np.ndarray
            Training target values
        X_train : pd.DataFrame
            Training features (contains item_id)

        Returns:
        --------
        mase : float
            Mean Absolute Scaled Error
        """
        # Calculate MAE on test set
        mae = mean_absolute_error(y_true, y_pred)

        # Reconstruct training set structure for naive error calculation
        train_df = pd.DataFrame({
            'item_id': X_train['item_id'].values,
            'target': y_train
        })

        # Calculate naive error per series
        naive_errors = []
        for item_id, group in train_df.groupby('item_id'):
            series = group['target'].values
            if len(series) > 1:
                # Naive forecast error: mean of |y_t - y_{t-1}|
                naive_error = np.mean(np.abs(np.diff(series)))
                if naive_error > 0:
                    naive_errors.append(naive_error)

        # Average naive error across all series
        global_naive_error = np.mean(naive_errors) if naive_errors else 1.0

        # MASE = MAE / naive_error
        mase = mae / global_naive_error if global_naive_error > 0 else float('inf')

        return mase

    def run_baseline(self,
                    X_train: pd.DataFrame,
                    y_train: np.ndarray,
                    X_test: pd.DataFrame,
                    y_test: np.ndarray,
                    estimator_name: str = 'rf') -> Dict[str, float]:
        """
        Run baseline model without BigFeat feature engineering.

        Uses lag features and basic temporal features as baseline.

        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
        estimator_name : str
            Which estimator to use

        Returns:
        --------
        results : dict
            Performance metrics
        """
        # Track total time
        monitor_total = ResourceMonitor()
        monitor_total.start()

        # Track data prep separately
        monitor_prep = ResourceMonitor()
        monitor_prep.start()

        # Use numeric features including Lag-1
        exclude_cols = ['timestamp', 'item_id']
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        if len(feature_cols) == 0:
            # Fallback: create basic feature
            if self.verbose:
                print("    Warning: No numeric features found, using basic fallback...")
            X_train_num = pd.DataFrame({'constant': np.ones(len(X_train))})
            X_test_num = pd.DataFrame({'constant': np.ones(len(X_test))})
        else:
            X_train_num = X_train[feature_cols].fillna(0)
            X_test_num = X_test[feature_cols].fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_num)
        X_test_scaled = scaler.transform(X_test_num)

        # Stop data prep tracking
        prep_resources = monitor_prep.stop()

        # Track model training separately
        monitor_model = ResourceMonitor()
        monitor_model.start()

        # Train model
        estimator = self.estimators[estimator_name]
        estimator.fit(X_train_scaled, y_train)

        # Predict
        y_pred = estimator.predict(X_test_scaled)

        # Stop model tracking
        model_resources = monitor_model.stop()

        # Calculate metrics using correct MASE
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mase = self.calculate_series_mase(y_test, y_pred, X_test, y_train, X_train)

        # Get total resource usage
        total_resources = monitor_total.stop()

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mase': mase,
            'n_features': X_train_scaled.shape[1],
            # Total resources
            'wall_time': total_resources['wall_time'],
            'cpu_time': total_resources['cpu_time'],
            'cpu_percent': total_resources['cpu_percent'],
            'memory_start_mb': total_resources['memory_start_mb'],
            'memory_end_mb': total_resources['memory_end_mb'],
            'memory_delta_mb': total_resources['memory_delta_mb'],
            'memory_peak_mb': total_resources['memory_peak_mb'],
            # Data prep resources
            'prep_wall_time': prep_resources['wall_time'],
            'prep_cpu_time': prep_resources['cpu_time'],
            'prep_cpu_percent': prep_resources['cpu_percent'],
            'prep_memory_delta_mb': prep_resources['memory_delta_mb'],
            # Model training resources
            'model_wall_time': model_resources['wall_time'],
            'model_cpu_time': model_resources['cpu_time'],
            'model_cpu_percent': model_resources['cpu_percent'],
            'model_memory_delta_mb': model_resources['memory_delta_mb'],
        }

    def create_basic_temporal_features(self, df: pd.DataFrame, datetime_col: str = 'timestamp') -> pd.DataFrame:
        """
        Create basic temporal features from datetime column.

        Parameters:
        -----------
        df : DataFrame
            Input dataframe with datetime column
        datetime_col : str
            Name of datetime column

        Returns:
        --------
        df_with_features : DataFrame
            DataFrame with additional temporal features
        """
        df = df.copy()

        if datetime_col in df.columns:
            dt = pd.to_datetime(df[datetime_col])

            # Basic temporal features
            df['year'] = dt.dt.year
            df['month'] = dt.dt.month
            df['day'] = dt.dt.day
            df['dayofweek'] = dt.dt.dayofweek
            df['dayofyear'] = dt.dt.dayofyear
            df['quarter'] = dt.dt.quarter
            df['weekofyear'] = dt.dt.isocalendar().week

            # Cyclical encoding for periodic features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # Add simple target statistics by item if available
        if 'item_id' in df.columns:
            df['item_count'] = df.groupby('item_id').cumcount()

        return df

    def run_bigfeat(self,
                   X_train: pd.DataFrame,
                   y_train: np.ndarray,
                   X_test: pd.DataFrame,
                   y_test: np.ndarray,
                   datetime_col: str,
                   enable_time_series: str = 'auto',
                   window_detector: str = 'dft',
                   estimator_name: str = 'rf',
                   bigfeat_params: Dict = None) -> Dict[str, Any]:
        """
        Run BigFeat feature engineering + model training.

        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
        datetime_col : str
            Name of datetime column
        enable_time_series : str
            BigFeat time series mode ('auto', 'yes', 'no')
        window_detector : str
            Window detection method ('dft', 'acf', 'lomb_scargle')
        estimator_name : str
            Which estimator to use
        bigfeat_params : dict
            Additional BigFeat parameters

        Returns:
        --------
        results : dict
            Performance metrics and BigFeat info
        """
        # Track total time
        monitor_total = ResourceMonitor()
        monitor_total.start()

        # Track BigFeat-specific time
        monitor_bigfeat = ResourceMonitor()

        # Create basic temporal features first
        X_train_with_features = self.create_basic_temporal_features(X_train, datetime_col)
        X_test_with_features = self.create_basic_temporal_features(X_test, datetime_col)

        # Determine appropriate parameters based on data frequency
        freq = X_train[datetime_col].iloc[1] - X_train[datetime_col].iloc[0] if len(X_train) > 1 else pd.Timedelta(days=1)

        # Set appropriate window detection parameters based on frequency
        if freq >= pd.Timedelta(days=300):  # Yearly data
            min_window = 365
            max_window = 365 * 10
            n_windows = 4
        elif freq >= pd.Timedelta(days=80):  # Quarterly data
            min_window = 90
            max_window = 365 * 3
            n_windows = 5
        elif freq >= pd.Timedelta(days=25):  # Monthly data
            min_window = 30
            max_window = 365 * 2
            n_windows = 6
        elif freq >= pd.Timedelta(days=5):  # Weekly data
            min_window = 7
            max_window = 365
            n_windows = 6
        else:  # Daily or hourly
            min_window = 1
            max_window = 365
            n_windows = 6

        # Default BigFeat parameters with native downsampling
        default_params = {
            'task_type': 'regression',
            'verbose': False,  # Reduce noise in benchmark
            'enable_time_series': enable_time_series,
            'window_detector': window_detector,
            'datetime_col': datetime_col,
            'groupby_cols': ['item_id'] if 'item_id' in X_train.columns else None,
            'dft_confidence_threshold': 0.3,
            'dft_min_window_days': min_window,
            'dft_max_window_days': max_window,
            'dft_n_windows': n_windows,
            # Native dynamic downsampling (NEW!)
            'enable_downsampling': True,
            'max_fit_samples': 0,  # 0 = fully dynamic (adapts to available RAM)
            'downsampling_random_state': 42,  # Reproducible sampling
        }

        if bigfeat_params:
            default_params.update(bigfeat_params)

        try:
            # Initialize BigFeat with native downsampling
            bf = BigFeat(**default_params)

            # Track BigFeat feature generation separately
            monitor_bigfeat.start()

            # BigFeat now handles downsampling internally with dynamic memory awareness!
            # No manual sampling needed - it automatically adapts to available memory.
            #
            # How it works:
            # 1. fit() detects available RAM and calculates optimal sample size
            # 2. If dataset > safe limit, samples for feature discovery
            # 3. transform() always uses full data for final features
            #
            # Result: Maximizes data usage while preventing OOM crashes

            # Fit on full training data (BigFeat will downsample if needed)
            X_train_transformed = bf.fit(
                X_train_with_features,  # Full data - BigFeat handles sampling
                y_train,
                gen_size=10,
                iterations=5,
                random_state=42,
                estimator='avg'
            )

            # Transform test set (always uses full data)
            X_test_transformed = bf.transform(X_test_with_features)

            # Stop BigFeat tracking
            bigfeat_resources = monitor_bigfeat.stop()

            # Get detector summary
            detector_info = {}
            if bf.enable_time_series:
                summary = bf.get_window_detection_summary()
                detector_info = {
                    'time_series_enabled': summary['time_series_enabled'],
                    'detection_strategy': summary['detection_strategy'],
                    'window_sizes_days': summary['window_sizes'],
                    'avg_confidence': summary['avg_confidence'],
                    'window_detector': window_detector
                }
            else:
                detector_info = {
                    'time_series_enabled': False,
                    'detection_strategy': 'disabled',
                    'window_sizes_days': None,
                    'avg_confidence': None,
                    'window_detector': window_detector
                }

            # Track model training separately
            monitor_model = ResourceMonitor()
            monitor_model.start()

            # Train model on transformed features
            estimator = self.estimators[estimator_name]
            estimator.fit(X_train_transformed, y_train)

            # Predict
            y_pred = estimator.predict(X_test_transformed)

            # Stop model tracking
            model_resources = monitor_model.stop()

            # Calculate metrics using correct MASE
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mase = self.calculate_series_mase(y_test, y_pred, X_test, y_train, X_train)

            # Get total resource usage
            total_resources = monitor_total.stop()

            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mase': mase,
                'n_features_generated': X_train_transformed.shape[1],
                'n_features_original': X_train_with_features.select_dtypes(include=[np.number]).shape[1],
                'n_features_base_temporal': len([c for c in X_train_with_features.columns
                                                 if c not in X_train.columns]),
                'detector_info': detector_info,
                'status': 'success',
                # Total resources (BigFeat + model training)
                'wall_time': total_resources['wall_time'],
                'cpu_time': total_resources['cpu_time'],
                'cpu_percent': total_resources['cpu_percent'],
                'memory_start_mb': total_resources['memory_start_mb'],
                'memory_end_mb': total_resources['memory_end_mb'],
                'memory_delta_mb': total_resources['memory_delta_mb'],
                'memory_peak_mb': total_resources['memory_peak_mb'],
                # BigFeat-specific resources
                'bigfeat_wall_time': bigfeat_resources['wall_time'],
                'bigfeat_cpu_time': bigfeat_resources['cpu_time'],
                'bigfeat_cpu_percent': bigfeat_resources['cpu_percent'],
                'bigfeat_memory_delta_mb': bigfeat_resources['memory_delta_mb'],
                # Model training resources
                'model_wall_time': model_resources['wall_time'],
                'model_cpu_time': model_resources['cpu_time'],
                'model_cpu_percent': model_resources['cpu_percent'],
                'model_memory_delta_mb': model_resources['memory_delta_mb'],
            }

        except Exception as e:
            total_resources = monitor_total.stop()

            if self.verbose:
                print(f"    ✗ BigFeat failed: {str(e)}")

            return {
                'status': 'failed',
                'error': str(e),
                'wall_time': total_resources['wall_time'],
                'cpu_time': total_resources['cpu_time'],
                'cpu_percent': total_resources['cpu_percent'],
                'memory_delta_mb': total_resources['memory_delta_mb'],
            }

    def benchmark_single_dataset(self,
                                dataset_name: str,
                                estimator_name: str = 'rf') -> Dict[str, Any]:
        """
        Run complete benchmark on a single dataset.

        CRASH RECOVERY: Saves results incrementally after each method completes.
        If benchmark crashes, it will resume from the last saved checkpoint.

        Parameters:
        -----------
        dataset_name : str
            Name of dataset to benchmark
        estimator_name : str
            Which estimator to use

        Returns:
        --------
        results : dict
            Complete results for this dataset
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Benchmarking: {dataset_name}")
            print(f"{'='*80}")

        # Check if results already exist (crash recovery)
        result_file = self.output_dir / f"{dataset_name}_results.json"
        if result_file.exists():
            if self.verbose:
                print(f"  ⚠️  Found existing results file, loading...")
            try:
                with open(result_file, 'r') as f:
                    existing_results = json.load(f)

                # Check if benchmark was completed
                if existing_results.get('status') == 'completed':
                    if self.verbose:
                        print(f"  ✓ Dataset already completed, skipping")
                    return existing_results
                else:
                    if self.verbose:
                        print(f"  ↻ Incomplete results found, resuming from checkpoint...")
                    results = existing_results
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Could not load existing results: {e}, starting fresh")
                results = {
                    'dataset': dataset_name,
                    'timestamp': datetime.now().isoformat(),
                    'estimator': estimator_name
                }
        else:
            results = {
                'dataset': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'estimator': estimator_name
            }

        try:
            # Load dataset (skip if already loaded)
            if 'metadata' not in results:
                df, metadata = self.load_and_prepare_dataset(dataset_name)
                results['metadata'] = metadata

                # Save checkpoint after loading
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                if self.verbose:
                    print(f"  💾 Checkpoint: Metadata saved")
            else:
                # Reload data from existing metadata
                df, metadata = self.load_and_prepare_dataset(dataset_name)

            # Create train/test split
            train_df, test_df = self.create_supervised_dataset(
                df,
                prediction_length=results['metadata']['prediction_length']
            )

            if len(train_df) == 0 or len(test_df) == 0:
                results['status'] = 'skipped'
                results['reason'] = 'Insufficient data for train/test split'

                # Save final result
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                return results

            # Prepare features and target
            X_train, y_train = self.prepare_features_target(train_df)
            X_test, y_test = self.prepare_features_target(test_df)

            if self.verbose:
                print(f"\nDataset split:")
                print(f"  Train: {len(X_train)} samples")
                print(f"  Test: {len(X_test)} samples")

            # Run baseline (skip if already completed)
            if 'baseline' not in results or results['baseline'].get('status') == 'failed':
                if self.verbose:
                    print(f"\n1. Running Baseline (no feature engineering)...")

                try:
                    baseline_results = self.run_baseline(
                        X_train, y_train, X_test, y_test, estimator_name
                    )
                    results['baseline'] = baseline_results

                    # Save checkpoint after baseline
                    with open(result_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)

                    if self.verbose:
                        print(f"  ✓ Baseline")
                        print(f"    MASE: {baseline_results['mase']:.4f}")
                        print(f"    Features: {baseline_results['n_features']}")
                        print(f"    Total Time: {baseline_results['wall_time']:.2f}s")
                        print(f"      - Data Prep: {baseline_results.get('prep_wall_time', 0):.2f}s")
                        print(f"      - Model Training: {baseline_results.get('model_wall_time', 0):.2f}s")
                        print(f"    CPU: {baseline_results['cpu_percent']:.1f}%")
                        print(f"    Memory: {baseline_results['memory_delta_mb']:.1f} MB")
                        print(f"  💾 Checkpoint: Baseline saved")

                except Exception as e:
                    if self.verbose:
                        print(f"  ✗ Baseline failed: {str(e)}")
                    results['baseline'] = {'status': 'failed', 'error': str(e)}

                    # Save checkpoint even on failure
                    with open(result_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
            else:
                if self.verbose:
                    print(f"\n1. Baseline (already completed)")
                    if 'mase' in results['baseline']:
                        print(f"  ✓ MASE: {results['baseline']['mase']:.4f}")

            # Run BigFeat variants
            for i, (ts_mode, detector) in enumerate(self.bigfeat_configs, 2):
                config_name = f"{ts_mode}_{detector}"
                label = f"BigFeat-{ts_mode}-{detector}"
                key = f'bigfeat_{config_name}'

                # Skip if already completed
                if key in results and results[key].get('status') == 'success':
                    if self.verbose:
                        print(f"\n{i}. {label} (already completed)")
                        if 'mase' in results[key]:
                            print(f"  ✓ MASE: {results[key]['mase']:.4f}")
                    continue

                if self.verbose:
                    print(f"\n{i}. Running {label}...")

                try:
                    bf_results = self.run_bigfeat(
                        X_train, y_train, X_test, y_test,
                        datetime_col='timestamp',
                        enable_time_series=ts_mode,
                        window_detector=detector,
                        estimator_name=estimator_name
                    )

                    results[key] = bf_results

                    # Save checkpoint after EACH BigFeat configuration
                    with open(result_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)

                    if bf_results['status'] == 'success':
                        if self.verbose:
                            print(f"  ✓ {label}")
                            print(f"    MASE: {bf_results['mase']:.4f}")
                            print(f"    Features: {bf_results['n_features_generated']}")
                            print(f"    Total Time: {bf_results['wall_time']:.2f}s")
                            print(f"      - BigFeat: {bf_results['bigfeat_wall_time']:.2f}s ({bf_results.get('bigfeat_cpu_percent', 0):.1f}% CPU)")
                            print(f"      - Model Training: {bf_results.get('model_wall_time', 0):.2f}s ({bf_results.get('model_cpu_percent', 0):.1f}% CPU)")
                            print(f"    Memory: {bf_results['memory_delta_mb']:.1f} MB (BigFeat: {bf_results['bigfeat_memory_delta_mb']:.1f} MB)")

                            if bf_results['detector_info']['time_series_enabled']:
                                det_info = bf_results['detector_info']
                                print(f"    TS Strategy: {det_info['detection_strategy']}")
                                if det_info.get('avg_confidence'):
                                    print(f"    TS Confidence: {det_info['avg_confidence']:.2f}")

                            print(f"  💾 Checkpoint: {label} saved")

                except Exception as e:
                    if self.verbose:
                        print(f"  ✗ {label} failed: {str(e)}")
                    results[key] = {'status': 'failed', 'error': str(e)}

                    # Save checkpoint even on failure
                    with open(result_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)

            results['status'] = 'completed'

            # Final save with completed status
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            if self.verbose:
                print(f"\n  ✅ Dataset completed and saved")

        except Exception as e:
            if self.verbose:
                print(f"\n✗ Dataset failed: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)

            # Save even on catastrophic failure
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

        return results

    def run_full_benchmark(self,
                          datasets: List[str] = None,
                          estimator_name: str = 'rf') -> pd.DataFrame:
        """
        Run benchmark on all datasets.

        Parameters:
        -----------
        datasets : list, optional
            List of dataset names (default: all datasets)
        estimator_name : str
            Which estimator to use

        Returns:
        --------
        summary_df : DataFrame
            Summary of all results
        """
        datasets = datasets or self.datasets

        print(f"\n{'='*80}")
        print(f"BigFeat Comprehensive Time Series Benchmark")
        print(f"{'='*80}")
        print(f"Datasets to test: {len(datasets)}")
        print(f"Estimator: {estimator_name}")
        print(f"BigFeat configurations: {len(self.bigfeat_configs)}")
        print(f"  - Baseline (no BigFeat)")
        for ts_mode, detector in self.bigfeat_configs:
            print(f"  - BigFeat: enable_time_series='{ts_mode}', window_detector='{detector}'")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"\n💾 CRASH RECOVERY ENABLED:")
        print(f"  - Results saved after each method completes")
        print(f"  - Re-running this script will resume from last checkpoint")
        print(f"  - Delete individual JSON files to re-run specific datasets")
        print(f"{'='*80}\n")

        all_results = []

        # Check for existing results to show progress
        completed_count = 0
        for dataset_name in datasets:
            result_file = self.output_dir / f"{dataset_name}_results.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        existing = json.load(f)
                    if existing.get('status') == 'completed':
                        completed_count += 1
                except:
                    pass

        if completed_count > 0:
            print(f"📊 Found {completed_count}/{len(datasets)} completed datasets")
            print(f"   Will skip completed datasets and resume incomplete ones\n")

        for i, dataset_name in enumerate(datasets, 1):
            print(f"\n[{i}/{len(datasets)}] Processing: {dataset_name}")
            print(f"    Progress: {i-1} completed, {len(datasets)-i} remaining")

            # Run benchmark (with automatic resumption)
            result = self.benchmark_single_dataset(dataset_name, estimator_name)
            all_results.append(result)

            # Note: Individual result already saved by benchmark_single_dataset

            if self.verbose:
                status_icon = "✅" if result.get('status') == 'completed' else "⚠️"
                print(f"    {status_icon} Dataset {dataset_name}: {result.get('status', 'unknown')}")

        # Create summary DataFrame
        summary_data = []

        for result in all_results:
            if result['status'] != 'completed':
                continue

            row = {
                'dataset': result['dataset'],
                'freq': result['metadata']['freq'],
                'n_series': result['metadata']['num_train_series'],
                'pred_length': result['metadata']['prediction_length'],
            }

            # Baseline metrics
            if 'baseline' in result and isinstance(result['baseline'], dict):
                baseline = result['baseline']
                if 'mase' in baseline:
                    row['baseline_mase'] = baseline['mase']
                    row['baseline_mae'] = baseline['mae']
                    row['baseline_r2'] = baseline['r2']
                    row['baseline_time'] = baseline['wall_time']
                    row['baseline_cpu_pct'] = baseline['cpu_percent']
                    row['baseline_mem_mb'] = baseline['memory_delta_mb']
                    row['baseline_prep_time'] = baseline.get('prep_wall_time', 0)
                    row['baseline_model_time'] = baseline.get('model_wall_time', 0)

            # BigFeat variants
            for ts_mode, detector in self.bigfeat_configs:
                config_name = f'{ts_mode}_{detector}'
                key = f'bigfeat_{config_name}'

                if key in result and result[key]['status'] == 'success':
                    row[f'{config_name}_mase'] = result[key]['mase']
                    row[f'{config_name}_mae'] = result[key]['mae']
                    row[f'{config_name}_r2'] = result[key]['r2']
                    row[f'{config_name}_time'] = result[key]['wall_time']
                    row[f'{config_name}_cpu_pct'] = result[key]['cpu_percent']
                    row[f'{config_name}_mem_mb'] = result[key]['memory_delta_mb']
                    row[f'{config_name}_n_features'] = result[key]['n_features_generated']

                    # BigFeat-specific resources
                    row[f'{config_name}_bf_time'] = result[key]['bigfeat_wall_time']
                    row[f'{config_name}_bf_cpu_pct'] = result[key]['bigfeat_cpu_percent']
                    row[f'{config_name}_bf_mem_mb'] = result[key]['bigfeat_memory_delta_mb']

                    # Model training resources
                    row[f'{config_name}_model_time'] = result[key].get('model_wall_time', 0)
                    row[f'{config_name}_model_cpu_pct'] = result[key].get('model_cpu_percent', 0)
                    row[f'{config_name}_model_mem_mb'] = result[key].get('model_memory_delta_mb', 0)

                    # Detector info
                    det_info = result[key]['detector_info']
                    row[f'{config_name}_ts_enabled'] = det_info['time_series_enabled']
                    if det_info['time_series_enabled']:
                        row[f'{config_name}_ts_strategy'] = det_info['detection_strategy']
                        row[f'{config_name}_ts_confidence'] = det_info.get('avg_confidence')

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)

        # Save summary
        summary_file = self.output_dir / "benchmark_summary.csv"
        summary_df.to_csv(summary_file, index=False)

        print(f"\n{'='*80}")
        print(f"Benchmark Complete!")
        print(f"{'='*80}")
        print(f"Results saved to: {self.output_dir.absolute()}")
        print(f"Summary: {summary_file}")
        print(f"Completed: {len(summary_data)}/{len(datasets)} datasets")

        # Print performance summary
        self.print_performance_summary(summary_df)

        return summary_df

    def print_performance_summary(self, summary_df: pd.DataFrame):
        """Print summary statistics of benchmark results."""

        print(f"\n{'='*80}")
        print("Performance Summary")
        print(f"{'='*80}\n")

        # All methods (baseline + all BigFeat configs)
        methods = ['baseline'] + [f'{ts_mode}_{detector}'
                                 for ts_mode, detector in self.bigfeat_configs]

        # MASE comparison
        print("Average MASE by method:")
        print(f"  {'Method':<30} {'MASE':>10}")
        print(f"  {'-'*30} {'-'*10}")
        for method in methods:
            col = f'{method}_mase'
            if col in summary_df.columns:
                avg_mase = summary_df[col].mean()
                method_label = 'Baseline' if method == 'baseline' else f'BF-{method}'
                print(f"  {method_label:<30} {avg_mase:>10.4f}")

        # Wins against baseline
        print("\n\nWins against Baseline (lower MASE):")
        print(f"  {'Method':<30} {'Wins':>10} {'Rate':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10}")
        for method in methods[1:]:  # Skip baseline
            mase_col = f'{method}_mase'
            if mase_col in summary_df.columns:
                wins = (summary_df[mase_col] < summary_df['baseline_mase']).sum()
                total = summary_df[mase_col].notna().sum()
                win_rate = (wins / total * 100) if total > 0 else 0
                method_label = f'BF-{method}'
                print(f"  {method_label:<30} {wins:>3}/{total:<6} {win_rate:>9.1f}%")

        # Runtime comparison
        print("\n\nAverage Runtime (seconds):")
        print(f"  {'Method':<30} {'Time (s)':>10} {'CPU %':>10} {'Mem (MB)':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        for method in methods:
            time_col = f'{method}_time'
            cpu_col = f'{method}_cpu_pct'
            mem_col = f'{method}_mem_mb'

            if time_col in summary_df.columns:
                avg_time = summary_df[time_col].mean()
                avg_cpu = summary_df[cpu_col].mean() if cpu_col in summary_df.columns else 0
                avg_mem = summary_df[mem_col].mean() if mem_col in summary_df.columns else 0

                method_label = 'Baseline' if method == 'baseline' else f'BF-{method}'
                print(f"  {method_label:<30} {avg_time:>10.2f} {avg_cpu:>10.1f} {avg_mem:>10.1f}")

        # Time series detection statistics (for auto modes)
        print("\n\nTime Series Detection Statistics (auto modes):")
        for detector in ['dft', 'acf', 'lomb_scargle']:
            config_name = f'auto_{detector}'
            ts_col = f'{config_name}_ts_enabled'

            if ts_col in summary_df.columns:
                enabled_count = summary_df[ts_col].sum()
                total = summary_df[ts_col].notna().sum()
                pct = (enabled_count / total * 100) if total > 0 else 0

                print(f"\n  {detector.upper()} detector:")
                print(f"    Enabled: {enabled_count}/{total} ({pct:.1f}%)")

                conf_col = f'{config_name}_ts_confidence'
                if conf_col in summary_df.columns:
                    avg_conf = summary_df[summary_df[ts_col] == True][conf_col].mean()
                    if not np.isnan(avg_conf):
                        print(f"    Avg confidence (when enabled): {avg_conf:.2f}")

        print(f"\n{'='*80}\n")


def main():
    """Main entry point for benchmark script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark of BigFeat on AutoGluon-TimeSeries datasets"
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets to test (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./benchmark_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--estimator',
        type=str,
        default='rf',
        choices=['rf', 'ridge', 'gbm'],
        help='Estimator to use for evaluation'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=3600,
        help='Time limit per dataset in seconds'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run on small subset for quick testing'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress'
    )

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = TimeSeriesBenchmark(
        output_dir=args.output_dir,
        time_limit_per_dataset=args.time_limit,
        verbose=args.verbose
    )

    # Select datasets
    if args.quick_test:
        # Small datasets for quick testing
        test_datasets = [
            'm1_yearly',
            'cif_2016',
            'tourism_yearly',
        ]
        print("\nRunning QUICK TEST on small datasets")
    elif args.datasets:
        test_datasets = args.datasets
    else:
        test_datasets = None  # All datasets

    # Run benchmark
    summary_df = benchmark.run_full_benchmark(
        datasets=test_datasets,
        estimator_name=args.estimator
    )

    print("\n✓ Benchmark completed successfully!")
    print(f"Results saved to: {benchmark.output_dir.absolute()}")


if __name__ == "__main__":
    main()