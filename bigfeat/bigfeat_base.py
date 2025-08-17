import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import bigfeat.local_utils as local_utils
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from functools import partial
import warnings
from datetime import timedelta


class BigFeat:
    """Enhanced BigFeat Class with time-based windows for time series support"""

    def __init__(self, task_type='classification', enable_time_series=False,
                 window_sizes=None, lag_periods=None, verbose=True,
                 datetime_col=None, groupby_cols=None, time_step='D'):
        """
        Initialize the BigFeat object

        Parameters:
        -----------
        task_type : str, default='classification'
            The type of machine learning task. Either 'classification' or 'regression'.
        enable_time_series : bool, default=False
            Whether to enable time series operators
        window_sizes : list of str or pd.Timedelta, optional
            List of time-based window sizes for rolling operations
            Examples: ['7D', '14D', '30D', '3M', '6M', '1Y'] or [pd.Timedelta(days=7), pd.Timedelta(days=30)]
        lag_periods : list of str or pd.Timedelta, optional
            List of time-based lag periods for time series operations
            Examples: ['1D', '7D', '30D'] or [pd.Timedelta(days=1), pd.Timedelta(days=7)]
        verbose : bool, default=True
            Whether to print progress messages
        datetime_col : str, optional
            Name of the datetime column to use for time series operations
        groupby_cols : list, optional
            List of columns to group by when applying time series operations
        time_step : str, default='D'
            Time step for resampling when using time-based windows
            Examples: 'D' (daily), 'H' (hourly), 'W' (weekly), 'M' (monthly)
        """
        # Original initialization
        self.n_jobs = -1
        self.operators = [np.multiply, np.add, np.subtract, np.abs, np.square]
        self.binary_operators = [np.multiply, np.add, np.subtract]
        self.unary_operators = [np.abs, np.square, local_utils.original_feat]
        self.task_type = task_type

        # Time series parameters
        self.enable_time_series = enable_time_series
        self.verbose = verbose
        self.time_step = time_step

        # Convert time-based windows to pandas Timedelta objects
        if window_sizes is None:
            self.window_sizes = [pd.Timedelta(days=7), pd.Timedelta(days=14), pd.Timedelta(days=30),
                                 pd.Timedelta(days=90), pd.Timedelta(days=180), pd.Timedelta(days=365)]
        else:
            self.window_sizes = self._parse_time_periods(window_sizes)

        if lag_periods is None:
            self.lag_periods = [pd.Timedelta(days=1), pd.Timedelta(days=7), pd.Timedelta(days=14),
                                pd.Timedelta(days=30), pd.Timedelta(days=90)]
        else:
            self.lag_periods = self._parse_time_periods(lag_periods)

        # Parameters for date/time column specification
        self.datetime_col = datetime_col
        self.groupby_cols = groupby_cols or []
        self.original_data = None  # Store original data with datetime info
        self.feature_columns = None  # Store feature column names

        # Add time series operators if enabled
        if enable_time_series:
            self.time_series_operators = [
                self._safe_rolling_mean,
                self._safe_rolling_std,
                self._safe_rolling_min,
                self._safe_rolling_max,
                self._safe_rolling_median,
                self._safe_rolling_sum,
                self._safe_lag_feature,
                self._safe_diff_feature,
                self._safe_pct_change,
                self._safe_ewm,
                self._safe_momentum,
                self._safe_seasonal_decompose,
                self._safe_trend_feature,
                self._safe_weekday_mean,
                self._safe_month_mean
            ]
            # Extend the original operators with time series operators
            self.operators.extend(self.time_series_operators)
            self.unary_operators.extend(self.time_series_operators)

            if self.verbose:
                print(f"Time series mode enabled with {len(self.time_series_operators)} additional operators")
                print(f"Window sizes: {[str(w) for w in self.window_sizes]}")
                print(f"Lag periods: {[str(l) for l in self.lag_periods]}")
                if self.datetime_col:
                    print(f"Date/time column specified: {self.datetime_col}")
                if self.groupby_cols:
                    print(f"Groupby columns specified: {self.groupby_cols}")
                print(f"Time step for resampling: {self.time_step}")

        # Validate task_type input
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be either 'classification' or 'regression'")

    def _parse_time_periods(self, periods):
        """
        Parse time periods from string or Timedelta format

        Parameters:
        -----------
        periods : list
            List of time periods as strings or Timedelta objects

        Returns:
        --------
        list of pd.Timedelta
            Parsed time periods
        """
        parsed_periods = []
        for period in periods:
            if isinstance(period, str):
                try:
                    parsed_periods.append(pd.Timedelta(period))
                except ValueError:
                    # Try parsing common formats
                    if period.endswith('D'):
                        days = int(period[:-1])
                        parsed_periods.append(pd.Timedelta(days=days))
                    elif period.endswith('W'):
                        weeks = int(period[:-1])
                        parsed_periods.append(pd.Timedelta(weeks=weeks))
                    elif period.endswith('M'):
                        months = int(period[:-1])
                        parsed_periods.append(pd.Timedelta(days=months * 30))  # Approximate
                    elif period.endswith('Y'):
                        years = int(period[:-1])
                        parsed_periods.append(pd.Timedelta(days=years * 365))  # Approximate
                    else:
                        # Default to days if no unit specified
                        parsed_periods.append(pd.Timedelta(days=int(period)))
            elif isinstance(period, pd.Timedelta):
                parsed_periods.append(period)
            else:
                # Try to convert to timedelta
                parsed_periods.append(pd.Timedelta(period))

        return parsed_periods

    def _prepare_time_series_data(self, X, y=None):
        """
        Prepare data for time-based series operations by organizing it with datetime and groupby columns

        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        y : array-like, optional
            Target variable

        Returns:
        --------
        X_processed : DataFrame
            Processed data ready for time-based series operations
        """
        if not self.enable_time_series or self.datetime_col is None:
            return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Convert to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # If we have stored feature columns, use them
            if self.feature_columns is not None:
                df = pd.DataFrame(X, columns=self.feature_columns)
            else:
                df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        # Add datetime column from stored original data if available
        if self.original_data is not None and self.datetime_col in self.original_data.columns:
            df[self.datetime_col] = self.original_data[self.datetime_col].values[:len(df)]

            # Add groupby columns if specified
            for col in self.groupby_cols:
                if col in self.original_data.columns:
                    df[col] = self.original_data[col].values[:len(df)]

        # Ensure datetime column is datetime type
        if self.datetime_col in df.columns:
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])

        # Sort by datetime and groupby columns for proper time series order
        sort_cols= [col for col in self.groupby_cols if col in df.columns]
        sort_cols.extend([self.datetime_col] if self.datetime_col in df.columns else [])

        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        return df

    def _apply_time_based_operation(self, data, feature_col, operation, window_size=None, lag_period=None):
        """
        Apply time-based series operation to a specific feature column with proper grouping

        Parameters:
        -----------
        data : DataFrame
            Data with datetime and groupby columns
        feature_col : str
            Name of the feature column to apply operation to
        operation : str
            Type of operation ('rolling_mean', 'lag', etc.)
        window_size : pd.Timedelta, optional
            Time-based window size for rolling operations
        lag_period : pd.Timedelta, optional
            Time-based lag period for lag operations

        Returns:
        --------
        result : array
            Result of the time-based series operation
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

    def _apply_single_group_operation(self, data, feature_col, operation, window_size=None, lag_period=None):
        """
        Apply operation to a single group with datetime index

        Parameters:
        -----------
        data : DataFrame
            Data with datetime index
        feature_col : str
            Feature column name
        operation : str
            Operation type
        window_size : pd.Timedelta, optional
            Window size
        lag_period : pd.Timedelta, optional
            Lag period

        Returns:
        --------
        pd.Series
            Result series with datetime index
        """
        series = data[feature_col]

        if operation == 'rolling_mean':
            window_size = window_size or self.rng.choice(self.window_sizes)
            result = series.rolling(window=window_size, min_periods=1).mean()

        elif operation == 'rolling_std':
            window_size = window_size or self.rng.choice(self.window_sizes)
            result = series.rolling(window=window_size, min_periods=1).std().fillna(0)

        elif operation == 'rolling_min':
            window_size = window_size or self.rng.choice(self.window_sizes)
            result = series.rolling(window=window_size, min_periods=1).min()

        elif operation == 'rolling_max':
            window_size = window_size or self.rng.choice(self.window_sizes)
            result = series.rolling(window=window_size, min_periods=1).max()

        elif operation == 'rolling_median':
            window_size = window_size or self.rng.choice(self.window_sizes)
            result = series.rolling(window=window_size, min_periods=1).median()

        elif operation == 'rolling_sum':
            window_size = window_size or self.rng.choice(self.window_sizes)
            result = series.rolling(window=window_size, min_periods=1).sum()

        elif operation == 'lag':
            lag_period = lag_period or self.rng.choice(self.lag_periods)
            # Create a shifted index
            shifted_index = series.index - lag_period
            result = pd.Series(index=series.index, dtype=float)
            for idx in series.index:
                shifted_idx = idx - lag_period
                # Find the closest timestamp within a reasonable tolerance
                tolerance = pd.Timedelta(self.time_step) if isinstance(self.time_step, str) else self.time_step
                mask = (series.index >= shifted_idx - tolerance) & (series.index <= shifted_idx + tolerance)
                if mask.any():
                    result[idx] = series[mask].iloc[0]
                else:
                    result[idx] = np.nan
            result = result.fillna(method='ffill').fillna(0)

        elif operation == 'diff':
            lag_period = lag_period or self.rng.choice(self.lag_periods)
            # Similar to lag but compute difference
            shifted_index = series.index - lag_period
            result = pd.Series(index=series.index, dtype=float)
            for idx in series.index:
                shifted_idx = idx - lag_period
                tolerance = pd.Timedelta(self.time_step) if isinstance(self.time_step, str) else self.time_step
                mask = (series.index >= shifted_idx - tolerance) & (series.index <= shifted_idx + tolerance)
                if mask.any():
                    result[idx] = series[idx] - series[mask].iloc[0]
                else:
                    result[idx] = 0
            result = result.fillna(0)

        elif operation == 'pct_change':
            lag_period = lag_period or self.rng.choice(self.lag_periods)
            # Percentage change over time period
            shifted_index = series.index - lag_period
            result = pd.Series(index=series.index, dtype=float)
            for idx in series.index:
                shifted_idx = idx - lag_period
                tolerance = pd.Timedelta(self.time_step) if isinstance(self.time_step, str) else self.time_step
                mask = (series.index >= shifted_idx - tolerance) & (series.index <= shifted_idx + tolerance)
                if mask.any():
                    old_val = series[mask].iloc[0]
                    if old_val != 0:
                        result[idx] = (series[idx] - old_val) / old_val
                    else:
                        result[idx] = 0
                else:
                    result[idx] = 0
            result = result.fillna(0)

        elif operation == 'ewm':
            # Use time-based exponential weighting
            window_size = window_size or self.rng.choice(self.window_sizes)
            halflife = window_size / 2
            result = series.ewm(halflife=halflife).mean()

        elif operation == 'momentum':
            lag_period = lag_period or self.rng.choice(self.lag_periods)
            # Momentum as difference from lag period
            result = self._apply_single_group_operation(data, feature_col, 'diff', None, lag_period)

        elif operation == 'seasonal_decompose':
            # Simple seasonal pattern extraction
            try:
                # Resample to regular frequency for seasonal decomposition
                resampled = series.resample(self.time_step).mean().fillna(method='ffill')
                if len(resampled) > 2:
                    # Simple seasonal pattern using day of year
                    seasonal_pattern = resampled.groupby(resampled.index.dayofyear).transform('mean')
                    # Map back to original index
                    result = pd.Series(index=series.index, dtype=float)
                    for idx in series.index:
                        day_of_year = idx.dayofyear
                        matching_seasonal = seasonal_pattern[seasonal_pattern.index.dayofyear == day_of_year]
                        if len(matching_seasonal) > 0:
                            result[idx] = matching_seasonal.iloc[0]
                        else:
                            result[idx] = series.mean()
                else:
                    result = pd.Series(series.mean(), index=series.index)
            except:
                result = pd.Series(series.mean(), index=series.index)

        elif operation == 'trend':
            window_size = window_size or self.rng.choice(self.window_sizes)
            # Simple trend as rolling slope
            result = series.rolling(window=window_size, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
            ).fillna(0)

        elif operation == 'weekday_mean':
            # Mean value by weekday
            weekday_means = series.groupby(series.index.dayofweek).mean()
            result = pd.Series(index=series.index, dtype=float)
            for idx in series.index:
                result[idx] = weekday_means.get(idx.dayofweek, series.mean())

        elif operation == 'month_mean':
            # Mean value by month
            month_means = series.groupby(series.index.month).mean()
            result = pd.Series(index=series.index, dtype=float)
            for idx in series.index:
                result[idx] = month_means.get(idx.month, series.mean())
        else:
            result = pd.Series(0, index=series.index)

        return result

    # Time Series Utility Methods (unchanged)
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

    # Enhanced Safe Time Series Operations that use time-based operations
    def _safe_rolling_mean(self, feature_data):
        """Safe rolling mean calculation using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'rolling_mean')
        else:
            # Fallback to original implementation
            try:
                window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).mean().bfill().values
                return self._clean_feature(result)
            except Exception:
                return feature_data

    def _safe_rolling_std(self, feature_data):
        """Safe rolling standard deviation using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'rolling_std')
        else:
            try:
                window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).std().fillna(0).values
                return self._clean_feature(result)
            except Exception:
                return np.zeros_like(feature_data)

    def _safe_rolling_min(self, feature_data):
        """Safe rolling minimum using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'rolling_min')
        else:
            try:
                window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).min().bfill().values
                return self._clean_feature(result)
            except Exception:
                return feature_data

    def _safe_rolling_max(self, feature_data):
        """Safe rolling maximum using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'rolling_max')
        else:
            try:
                window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).max().bfill().values
                return self._clean_feature(result)
            except Exception:
                return feature_data

    def _safe_rolling_median(self, feature_data):
        """Safe rolling median using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'rolling_median')
        else:
            try:
                window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).median().bfill().values
                return self._clean_feature(result)
            except Exception:
                return feature_data

    def _safe_rolling_sum(self, feature_data):
        """Safe rolling sum using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'rolling_sum')
        else:
            try:
                window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).sum().fillna(0).values
                return self._clean_feature(result)
            except Exception:
                return np.zeros_like(feature_data)

    def _safe_lag_feature(self, feature_data):
        """Safe lag feature creation using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'lag')
        else:
            try:
                lag_periods = self.rng.choice([1, 2, 3, 5, 7, 10])
                lag_periods = min(lag_periods, len(feature_data) - 1)
                result = pd.Series(feature_data).shift(lag_periods).bfill().values
                return self._clean_feature(result)
            except Exception:
                return feature_data

    def _safe_diff_feature(self, feature_data):
        """Safe difference calculation using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'diff')
        else:
            try:
                periods = self.rng.choice([1, 2, 3, 5, 7, 10])
                periods = min(periods, len(feature_data) - 1)
                result = pd.Series(feature_data).diff(periods).fillna(0).values
                return self._clean_feature(result)
            except Exception:
                return np.zeros_like(feature_data)

    def _safe_pct_change(self, feature_data):
        """Safe percentage change using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'pct_change')
        else:
            try:
                periods = self.rng.choice([1, 2, 3, 5, 7, 10])
                periods = min(periods, len(feature_data) - 1)
                result = pd.Series(feature_data).pct_change(periods).fillna(0).values
                result = np.where(np.isinf(result), 0, result)
                return self._clean_feature(result)
            except Exception:
                return np.zeros_like(feature_data)

    def _safe_ewm(self, feature_data):
        """Safe exponential moving average using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'ewm')
        else:
            try:
                alpha = self.rng.choice([0.1, 0.2, 0.3, 0.5])
                result = pd.Series(feature_data).ewm(alpha=alpha, adjust=False).mean().values
                return self._clean_feature(result)
            except Exception:
                return feature_data

    def _safe_momentum(self, feature_data):
        """Safe momentum calculation using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'momentum')
        else:
            try:
                periods = self.rng.choice([1, 2, 3, 5, 7, 10])
                periods = min(periods, len(feature_data) - 1)
                series = pd.Series(feature_data)
                momentum = series - series.shift(periods)
                result = momentum.fillna(0).values
                return self._clean_feature(result)
            except Exception:
                return np.zeros_like(feature_data)

    def _safe_seasonal_decompose(self, feature_data):
        """Safe seasonal decomposition using time-based operations"""
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

    def _safe_trend_feature(self, feature_data):
        """Safe trend feature using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'trend')
        else:
            try:
                window_size = self.rng.choice([7, 14, 30, 60])
                window_size = min(window_size, len(feature_data))
                series = pd.Series(feature_data)
                # Simple trend as rolling linear regression slope
                result = series.rolling(window=window_size, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
                )
                return self._clean_feature(result.fillna(0).values)
            except Exception:
                return np.zeros_like(feature_data)

    def _safe_weekday_mean(self, feature_data):
        """Safe weekday mean using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'weekday_mean')
        else:
            # Fallback: create simple cyclical feature
            try:
                result = np.sin(2 * np.pi * np.arange(len(feature_data)) / 7)
                return self._clean_feature(result * np.std(feature_data) + np.mean(feature_data))
            except Exception:
                return feature_data

    def _safe_month_mean(self, feature_data):
        """Safe month mean using time-based operations"""
        if self.enable_time_series and hasattr(self, '_current_data') and hasattr(self, '_current_feature_index'):
            feature_col = self.feature_columns[
                self._current_feature_index] if self.feature_columns else f'feature_{self._current_feature_index}'
            return self._apply_time_based_operation(self._current_data, feature_col, 'month_mean')
        else:
            # Fallback: create simple cyclical feature
            try:
                result = np.sin(2 * np.pi * np.arange(len(feature_data)) / 30)
                return self._clean_feature(result * np.std(feature_data) + np.mean(feature_data))
            except Exception:
                return feature_data

    # Updated fit method to handle DataFrame input with datetime column (unchanged from previous version)
    def fit(self, X, y, gen_size=5, random_state=0, iterations=5, estimator='avg',
            feat_imps=True, split_feats=None, check_corr=True, selection='stability', combine_res=True):
        """ Generated Features using test set - Enhanced for datetime-aware time series operations """

        if self.verbose and self.enable_time_series:
            print(
                f"Starting BigFeat with time-based series support. Data shape: {X.shape if hasattr(X, 'shape') else len(X)}")

        # Store original data if it's a DataFrame (for datetime column access)
        if isinstance(X, pd.DataFrame):
            self.original_data = X.copy()
            # Identify feature columns by excluding datetime and groupby columns
            exclude_cols = []
            if self.datetime_col and self.datetime_col in X.columns:
                exclude_cols.append(self.datetime_col)
            exclude_cols.extend([col for col in self.groupby_cols if col in X.columns])

            # Get all potential feature columns
            all_feature_cols = [col for col in X.columns if col not in exclude_cols]

            # Filter out any non-numeric columns more carefully
            numeric_feature_cols = []
            for col in all_feature_cols:
                col_dtype = str(X[col].dtype)
                # Skip datetime-like columns and object columns that might contain timestamps
                if (col_dtype.startswith('datetime') or
                        col_dtype.startswith('<M8') or  # numpy datetime64
                        col_dtype == 'object'):
                    # For object columns, check if they contain datetime-like objects
                    if col_dtype == 'object':
                        try:
                            sample_val = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else None
                            if sample_val is not None and hasattr(sample_val, 'year'):
                                # Likely a datetime object
                                if self.verbose:
                                    print(f"Warning: Skipping datetime-like column '{col}'")
                                continue
                        except:
                            pass
                    else:
                        if self.verbose:
                            print(f"Warning: Skipping datetime column '{col}' (type: {col_dtype})")
                        continue

                # Test if the column is actually numeric
                try:
                    # Try to convert a sample to float
                    test_val = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else 0
                    float(test_val)
                    numeric_feature_cols.append(col)
                except (ValueError, TypeError):
                    if self.verbose:
                        print(f"Warning: Skipping non-numeric column '{col}' (type: {col_dtype})")
                    continue

            self.feature_columns = numeric_feature_cols
            # Extract feature data for processing
            if len(self.feature_columns) > 0:
                X_features = X[self.feature_columns].values
                # Ensure all values are numeric - this should now be safe
                X_features = np.array(X_features, dtype=float)
            else:
                raise ValueError("No numeric feature columns found after filtering!")
        else:
            X_features = X
            # Ensure it's numeric
            X_features = np.array(X_features, dtype=float)
            if hasattr(X, 'columns'):
                self.feature_columns = list(X.columns)
            else:
                self.feature_columns = [f'feature_{i}' for i in range(X_features.shape[1])]

        # Prepare time series data
        if self.enable_time_series:
            self._current_data = self._prepare_time_series_data(X)

        # Original initialization - unchanged
        self.selection = selection
        self.imp_operators = np.ones(5)
        self.imp_operators. += np.ones(len(self.feature_columns) - 5) * 2
        self.operator_weights = self.imp_operators / self.imp_operators.sum()
        self.gen_steps = []
        self.n_feats = X_features.shape[1]
        self.n_rows = X_features.shape[0]
        self.ig_vector = np.ones(self.n_feats) / self.n_feats
        self.comb_mat = np.ones((self.n_feats, self.n_feats))
        self.split_vec = np.ones(self.n_feats)

        # Set RNG seed if provided for numpy - original
        self.rng = np.random.RandomState(seed=random_state)

        # Original variable initialization
        gen_feats = np.zeros((self.n_rows, self.n_feats * gen_size))
        iters_comb = np.zeros((self.n_rows, self.n_feats * iterations))
        depths_comb = np.zeros(self.n_feats * iterations)
        ids_comb = np.zeros(self.n_feats * iterations, dtype=object)
        ops_comb = np.zeros(self.n_feats * iterations, dtype=object)
        self.feat_depths = np.zeros(gen_feats.shape[1])
        self.depth_range = np.arange(3) + 1
        self.depth_weights = 1 / (2 ** self.depth_range)
        self.depth_weights /= self.depth_weights.sum()

        # Original scaling
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_features)
        X_scaled = self.scaler.transform(X_features)

        # Original feature importance calculation
        if feat_imps:
            self.ig_vector, estimators = self.get_feature_importances(X_scaled, y, estimator, random_state)
            self.ig_vector /= self.ig_vector.sum()
            for tree in estimators:
                paths = self.get_paths(tree, np.arange(X_scaled.shape[1]))
                self.get_split_feats(paths, self.split_vec)
            self.split_vec /= self.split_vec.sum()

            if split_feats == "comb":
                self.ig_vector = np.multiply(self.ig_vector, self.split_vec)
                self.ig_vector /= self.ig_vector.sum()
            elif split_feats == "splits":
                self.ig_vector = self.split_vec

        # Original iteration loop with enhanced time series support
        for iteration in range(iterations):
            if self.verbose:
                print(f"Feature generation iteration {iteration + 1}/{iterations}")

            self.tracking_ops = []
            self.tracking_ids = []
            gen_feats = np.zeros((self.n_rows, self.n_feats * gen_size))
            self.feat_depths = np.zeros(gen_feats.shape[1])

            for i in range(gen_feats.shape[1]):
                dpth = self.rng.choice(self.depth_range, p=self.depth_weights)
                ops = []
                ids = []
                gen_feats[:, i] = self.feat_with_depth(X_scaled, dpth, ops, ids)  # ops and ids are updated
                # Clean generated feature if time series is enabled
                if self.enable_time_series:
                    gen_feats[:, i] = self._clean_feature(gen_feats[:, i])
                self.feat_depths[i] = dpth
                self.tracking_ops.append(ops)
                self.tracking_ids.append(ids)

            self.tracking_ids = np.array(self.tracking_ids + [[]], dtype='object')[:-1]
            self.tracking_ops = np.array(self.tracking_ops + [[]], dtype='object')[:-1]

            # Original feature selection within iteration
            imps, estimators = self.get_feature_importances(gen_feats, y, estimator, random_state)
            total_feats = np.argsort(imps)
            feat_args = total_feats[-self.n_feats:]
            gen_feats = gen_feats[:, feat_args]
            self.tracking_ids = self.tracking_ids[feat_args]
            self.tracking_ops = self.tracking_ops[feat_args]
            self.feat_depths = self.feat_depths[feat_args]

            # Original combination tracking
            depths_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.feat_depths
            ids_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.tracking_ids
            ops_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.tracking_ops
            iters_comb[:, iteration * self.n_feats:(iteration + 1) * self.n_feats] = gen_feats

            # Original operator importance update - now includes time series operators
            for i, op in enumerate(self.operators):
                for feat in self.tracking_ops:
                    for feat_op in feat:
                        if op == feat_op[0]:
                            self.imp_operators[i] += 1
            self.operator_weights = self.imp_operators / self.imp_operators.sum()

        # Original final selection logic
        if selection == 'stability' and iterations > 1 and combine_res:
            imps, estimators = self.get_feature_importances(iters_comb, y, estimator, random_state)
            total_feats = np.argsort(imps)
            feat_args = total_feats[-self.n_feats:]
            gen_feats = iters_comb[:, feat_args]
            self.tracking_ids = ids_comb[feat_args]
            self.tracking_ops = ops_comb[feat_args]
            self.feat_depths = depths_comb[feat_args]

        if selection == 'stability' and check_corr:
            gen_feats, to_drop_cor = self.check_correlations(gen_feats)
            self.tracking_ids = np.delete(self.tracking_ids, to_drop_cor)
            self.tracking_ops = np.delete(self.tracking_ops, to_drop_cor)
            self.feat_depths = np.delete(self.feat_depths, to_drop_cor)

        # Original final combination
        gen_feats = np.hstack((gen_feats, X_scaled))

        if selection == 'fAnova':
            # Use the appropriate feature selection method based on task type
            if self.task_type == 'classification':
                self.fAnova_best = SelectKBest(f_classif, k=self.n_feats)
            else:  # regression
                self.fAnova_best = SelectKBest(f_regression, k=self.n_feats)
            gen_feats = self.fAnova_best.fit_transform(gen_feats, y)

        if self.verbose:
            print(f"Feature generation completed. Final shape: {gen_feats.shape}")
            if self.enable_time_series:
                ts_ops_count = sum(1 for ops in self.tracking_ops
                                   for op_info in ops
                                   if len(op_info) > 0 and callable(op_info[0]) and op_info[
                                       0] in self.time_series_operators)
                print(f"Time series operations used: {ts_ops_count}")

        return gen_feats

    def transform(self, X):
        """ Produce features from the fitted BigFeat object - Enhanced for datetime-aware operations """

        # Handle DataFrame input with datetime column
        if isinstance(X, pd.DataFrame):
            if self.feature_columns:
                # Use the stored feature columns from fit
                available_feature_cols = [col for col in self.feature_columns if col in X.columns]
                if len(available_feature_cols) != len(self.feature_columns):
                    missing_cols = set(self.feature_columns) - set(available_feature_cols)
                    if self.verbose:
                        print(f"Warning: Some feature columns missing in transform data: {missing_cols}")
                X_features = X[available_feature_cols].values
                # Ensure numeric
                X_features = np.array(X_features, dtype=float)
            else:
                # Exclude datetime and groupby columns if they exist
                exclude_cols = []
                if self.datetime_col and self.datetime_col in X.columns:
                    exclude_cols.append(self.datetime_col)
                exclude_cols.extend([col for col in self.groupby_cols if col in X.columns])

                # Get potential feature columns
                feature_cols = [col for col in X.columns if col not in exclude_cols]

                # Filter for numeric columns only using the same logic as fit
                numeric_feature_cols = []
                for col in feature_cols:
                    col_dtype = str(X[col].dtype)
                    # Skip datetime-like columns
                    if (col_dtype.startswith('datetime') or
                            col_dtype.startswith('<M8') or
                            col_dtype == 'object'):
                        if col_dtype == 'object':
                            try:
                                sample_val = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else None
                                if sample_val is not None and hasattr(sample_val, 'year'):
                                    continue
                            except:
                                pass
                        else:
                            continue

                    try:
                        test_val = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else 0
                        float(test_val)
                        numeric_feature_cols.append(col)
                    except (ValueError, TypeError):
                        continue

                X_features = X[numeric_feature_cols].values
                X_features = np.array(X_features, dtype=float)

            # Update current data for time series operations
            if self.enable_time_series:
                self._current_data = self._prepare_time_series_data(X)
        else:
            X_features = X
            X_features = np.array(X_features, dtype=float)

        X_scaled = self.scaler.transform(X_features)
        self.n_rows = X_scaled.shape[0]
        gen_feats = np.zeros((self.n_rows, len(self.tracking_ids)))

        for i in range(gen_feats.shape[1]):
            dpth = self.feat_depths[i]
            op_ls = self.tracking_ops[i].copy()
            id_ls = self.tracking_ids[i].copy()
            gen_feats[:, i] = self.feat_with_depth_gen(X_scaled, dpth, op_ls, id_ls)
            # Clean generated feature if time series is enabled
            if self.enable_time_series:
                gen_feats[:, i] = self._clean_feature(gen_feats[:, i])

        gen_feats = np.hstack((gen_feats, X_scaled))

        if self.selection == 'fAnova':
            gen_feats = self.fAnova_best.transform(gen_feats)

        return gen_feats

    # Enhanced feat_with_depth method to support datetime-aware operations
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

    def feat_with_depth_gen(self, X, depth, op_ls, feat_ls):
        """ Reproduce generated features with new data - Enhanced to handle datetime-aware time series operators """
        if depth == 0:
            feat_ind = feat_ls.pop()
            # Set current feature index for time series operations
            if self.enable_time_series:
                self._current_feature_index = feat_ind
            return X[:, feat_ind]

        depth -= 1
        op = op_ls.pop()[0]

        if op in self.binary_operators:
            feat_1 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls)
            feat_2 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls)
            result = op(feat_2, feat_1)
            return self._clean_feature(result) if self.enable_time_series else result

        elif op in self.unary_operators:
            feat_1 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls)
            result = op(feat_1)
            return self._clean_feature(result) if self.enable_time_series else result

    # Original methods - completely unchanged from previous version
    def select_estimator(self, X, y, estimators_names=None):
        """
        Select the best estimator based on cross-validation - Original method
        """
        if estimators_names is None:
            if self.task_type == 'classification':
                estimators_names = ['dt', 'lr']
            else:  # regression
                estimators_names = ['dt_reg', 'lr_reg']

        estimators_dic = {
            'dt': DecisionTreeClassifier(),
            'lr': LogisticRegression(),
            'rf': RandomForestClassifier(n_jobs=self.n_jobs),
            'lgb': LGBMClassifier(),
            'dt_reg': DecisionTreeRegressor(),
            'lr_reg': LinearRegression(),
            'rf_reg': RandomForestRegressor(n_jobs=self.n_jobs),
            'lgb_reg': LGBMRegressor()
        }

        models_score = {}

        for estimator in estimators_names:
            model = estimators_dic[estimator]

            if self.task_type == 'classification':
                scorer = make_scorer(f1_score)
            else:  # regression
                scorer = make_scorer(r2_score)

            models_score[estimator] = cross_val_score(model, X, y, cv=3, scoring=scorer).mean()

        best_estimator = max(models_score, key=models_score.get)
        best_model = estimators_dic[best_estimator]
        best_model.fit(X, y)
        return best_model

    def get_feature_importances(self, X, y, estimator, random_state, sample_count=1, sample_size=3, n_jobs=1):
        """Return feature importances by specified method - Original method"""
        importance_sum = np.zeros(X.shape[1])
        total_estimators = []

        for sampled in range(sample_count):
            sampled_ind = np.random.choice(np.arange(self.n_rows), size=self.n_rows // sample_size, replace=False)
            sampled_X = X[sampled_ind]
            sampled_y = np.take(y, sampled_ind)

            if estimator in ["rf", "rf_reg"]:
                if self.task_type == 'classification' or estimator == "rf":
                    estm = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
                else:
                    estm = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)

                estm.fit(sampled_X, sampled_y)
                total_importances = estm.feature_importances_
                estimators = estm.estimators_
                total_estimators += estimators

            elif estimator == "avg":
                if self.task_type == 'classification':
                    clf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
                    clf.fit(sampled_X, sampled_y)
                    rf_importances = clf.feature_importances_
                    estimators = clf.estimators_
                    total_estimators += estimators

                    train_data = lgb.Dataset(sampled_X, label=sampled_y)
                    param = {'num_leaves': 31, 'objective': 'binary', 'verbose': -1}
                    param['metric'] = 'auc'

                else:
                    clf = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
                    clf.fit(sampled_X, sampled_y)
                    rf_importances = clf.feature_importances_
                    estimators = clf.estimators_
                    total_estimators += estimators

                    train_data = lgb.Dataset(sampled_X, label=sampled_y)
                    param = {'num_leaves': 31, 'objective': 'regression', 'verbose': -1}
                    param['metric'] = 'rmse'

                num_round = 2
                bst = lgb.train(param, train_data, num_round)
                lgb_imps = bst.feature_importance(importance_type='gain')
                lgb_imps /= lgb_imps.sum()
                total_importances = (rf_importances + lgb_imps) / 2

            else:
                raise ValueError(f"Unsupported estimator: {estimator}")

            importance_sum += total_importances
        return importance_sum, total_estimators

    def get_weighted_feature_importances(self, X, y, estimator, random_state):
        """Return feature importances weighted by model performance - Original method"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state)

        if self.task_type == 'classification':
            estm = RandomForestClassifier(random_state=random_state, n_jobs=self.n_jobs)
        else:
            estm = RandomForestRegressor(random_state=random_state, n_jobs=self.n_jobs)

        estm.fit(X_train, y_train)
        ests = estm.estimators_
        model = estm
        imps = np.zeros((len(model.estimators_), X.shape[1]))
        scores = np.zeros(len(model.estimators_))

        for i, each in enumerate(model.estimators_):
            if self.task_type == 'classification':
                y_probas_train = each.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_probas_train)
            else:
                y_pred_train = each.predict(X_test)
                score = r2_score(y_test, y_pred_train)

            imps[i] = each.feature_importances_
            scores[i] = score

        weights = scores / scores.sum()
        return np.average(imps, axis=0, weights=weights)

    def check_correlations(self, feats):
        """ Check correlations among the selected features - Original method """
        cor_thresh = 0.8
        corr_matrix = pd.DataFrame(feats).corr().abs()
        mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
        tri_df = corr_matrix.mask(mask)
        to_drop = [c for c in tri_df.columns if any(tri_df[c] > cor_thresh)]
        feats = pd.DataFrame(feats).drop(to_drop, axis=1)
        return feats.values, to_drop

    def get_paths(self, clf, feature_names):
        """ Returns every path in the decision tree - Original method """
        tree_ = clf.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        path = []
        path_list = []

        def recurse(node, depth, path_list):
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                path_list.append(path.copy())
            else:
                name = feature_name[node]
                path.append(name)
                recurse(tree_.children_left[node], depth + 1, path_list)
                recurse(tree_.children_right[node], depth + 1, path_list)
                path.pop()

        recurse(0, 1, path_list)

        new_list = []
        for i in range(len(path_list)):
            if path_list[i] != path_list[i - 1]:
                new_list.append(path_list[i])
        return new_list

    def get_combos(self, paths, comb_mat):
        """ Fills Combination matrix with values - Original method """
        for i in range(len(comb_mat)):
            for pt in paths:
                if i in pt:
                    comb_mat[i][pt] += 1

    def get_split_feats(self, paths, split_vec):
        """ Fills split vector with values - Original method """
        for i in range(len(split_vec)):
            for pt in paths:
                if i in pt:
                    split_vec[i] += 1