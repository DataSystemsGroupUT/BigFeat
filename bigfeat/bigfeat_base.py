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
    def __init__(self,
                 task_type='classification',
                 enable_time_series='auto',  # 'yes'/'no'/'auto'
                 window_detector='dft', # 'dft'/'acf'/'lomb_scargle'
                 window_sizes=None,
                 lag_periods=None,
                 verbose=True,
                 datetime_col=None,
                 groupby_cols=None,
                 time_step='D',
                 ts_operation_weight_multiplier=1.0,
                 window_step_options=None,

                 # DFT-related parameters
                 dft_confidence_threshold=0.3,
                 dft_min_window_days=1,
                 dft_max_window_days=365,
                 dft_n_windows=6):
        """
        Initialize the BigFeat object with configurable window detection

        Parameters:
        -----------
        task_type : str, default='classification'
            The type of machine learning task. Either 'classification' or 'regression'.

        enable_time_series : str, default='auto'
            Time series feature generation mode:
            - 'yes': Force enable time series with DFT-detected windows
            - 'no': Disable time series features entirely
            - 'auto': Automatically detect datetime column and assess periodicity

        window_detector : str, default='dft'
            Window detection method to use:
            - 'dft': DFT-based detection (default, good for regular data)
            - 'acf': ACF-based detection (good for short series, robust)
            - 'lomb_scargle': Lomb-Scargle (good for irregular/missing data)

        window_sizes : list of str or pd.Timedelta, optional
            List of time-based window sizes for rolling operations.
            If None and enable_time_series='yes', will use DFT-detected windows.
            If None and enable_time_series='auto', will use DFT if periodicity detected.
            Examples: ['7D', '14D', '30D', '3M', '6M', '1Y']

        lag_periods : list of str or pd.Timedelta, optional
            List of time-based lag periods for time series operations
            If None, will derive from detected window sizes
            Examples: ['1D', '7D', '30D']

        verbose : bool, default=True
            Whether to print progress messages

        datetime_col : str, optional
            Name of the datetime column to use for time series operations
            If None and enable_time_series='auto', will attempt auto-detection

        groupby_cols : list, optional
            List of columns to group by when applying time series operations

        time_step : str, default='D'
            Time step for resampling when using time-based windows
            Examples: 'D' (daily), 'H' (hourly), 'W' (weekly), 'M' (monthly)

        ts_operation_weight_multiplier : float, default=1.0
            Multiplier for time series operation weights

        window_step_options : list, optional
            List of possible window step options to choose from

        dft_confidence_threshold : float, default=1.3
            Minimum confidence score to consider periodicity reliable
            Used in 'auto' mode to decide whether to enable time series

        dft_min_window_days : int, default=3
            Minimum window size in days for DFT detection

        dft_max_window_days : int, default=365
            Maximum window size in days for DFT detection

        dft_n_windows : int, default=6
            Number of window sizes to generate from DFT analysis
        """
        # BigFeat initialization
        self.n_jobs = -1
        self.operators = [np.multiply, np.add, np.subtract, np.abs, np.square]
        self.binary_operators = [np.multiply, np.add, np.subtract]
        self.unary_operators = [np.abs, np.square, local_utils.original_feat]
        self.task_type = task_type
        self.verbose = verbose

        # Validate enable_time_series parameter
        if enable_time_series not in ['yes', 'no', 'auto']:
            # Handle legacy boolean values for backward compatibility
            if isinstance(enable_time_series, bool):
                enable_time_series = 'yes' if enable_time_series else 'no'
                if self.verbose:
                    print(f"Warning: enable_time_series should be 'yes'/'no'/'auto', not boolean. "
                          f"Converting to '{enable_time_series}'")
            else:
                raise ValueError("enable_time_series must be 'yes', 'no', or 'auto'")

        self.enable_time_series_mode = enable_time_series
        self.enable_time_series = False  # Will be set during fit based on mode

        # DFT parameters
        self.dft_confidence_threshold = dft_confidence_threshold
        self.dft_min_window_days = dft_min_window_days
        self.dft_max_window_days = dft_max_window_days
        self.dft_n_windows = dft_n_windows

        # Initialize the appropriate window detector
        self.window_detector_type = window_detector
        self.window_detector = self._initialize_window_detector(
            window_detector,
            dft_min_window_days,
            dft_max_window_days,
            dft_n_windows,
            dft_confidence_threshold,
            verbose
        )

        # Time series parameters
        self.ts_operation_weight_multiplier = ts_operation_weight_multiplier

        # Window step options
        if window_step_options is None:
            self.window_step_options = ['D', 'H', 'W', 'M']
        else:
            self.window_step_options = window_step_options

        self.time_step = time_step

        # Store user-provided windows (will be overridden by DFT if needed)
        self.user_provided_windows = window_sizes
        self.user_provided_lags = lag_periods

        # These will be set during fit() based on mode
        self.window_sizes = None
        self.lag_periods = None

        # Parameters for date/time column
        self.datetime_col = datetime_col
        self.groupby_cols = groupby_cols or []
        self.original_data = None
        self.feature_columns = None

        # Tracking variables for DFT results
        self.detected_windows = None
        self.confidence_scores = None
        self.detection_strategy = None

        # Validate task_type input
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be either 'classification' or 'regression'")

        if self.verbose:
            print(f"BigFeat initialized with enable_time_series='{enable_time_series}'")
            print(f"  Window detector: {window_detector}")
            if enable_time_series == 'yes':
                print("  → Time series features will be generated using DFT-detected windows")
            elif enable_time_series == 'auto':
                print("  → Time series will be auto-detected based on datetime column and periodicity")
            else:
                print("  → Time series features disabled")

    def _initialize_window_detector(self, detector_type, min_window_days,
                                    max_window_days, n_windows,
                                    confidence_threshold, verbose):
        """
        Initialize the appropriate window detector based on type

        Parameters:
        -----------
        detector_type : str
            Type of detector: 'dft', 'acf', or 'lomb_scargle'

        Returns:
        --------
        detector instance with standardized interface
        """
        if detector_type == 'dft':
            from bigfeat.dft_window_detector import DFTWindowDetector
            return DFTWindowDetector(
                min_window_days=min_window_days,
                max_window_days=max_window_days,
                n_windows=n_windows,
                confidence_threshold=confidence_threshold,
                verbose=verbose
            )
        elif detector_type == 'acf':
            from bigfeat.acf_window_detector import ACFWindowDetector
            return ACFWindowDetector(
                min_window_days=min_window_days,
                max_window_days=max_window_days,
                n_windows=n_windows,
                confidence_threshold=confidence_threshold,
                verbose=verbose
            )
        elif detector_type == 'lomb_scargle':
            from bigfeat.lomb_scargle_window_detector import LombScargleWindowDetector
            return LombScargleWindowDetector(
                min_window_days=min_window_days,
                max_window_days=max_window_days,
                n_windows=n_windows,
                confidence_threshold=confidence_threshold,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown window_detector: {detector_type}. "
                             f"Must be 'dft', 'acf', or 'lomb_scargle'")

    def _setup_time_series(self, X, y=None):
        """
        Setup time series configuration based on enable_time_series_mode
        Called at the beginning of fit()

        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        y : array-like, optional
            Target variable

        Returns:
        --------
        bool
            Whether time series should be enabled
        """
        mode = self.enable_time_series_mode

        # Case 1: Explicitly disabled
        if mode == 'no':
            if self.verbose:
                print("\n=== Time Series: DISABLED ===")
            self.enable_time_series = False
            return False

        # Case 2: Explicitly enabled ('yes')
        if mode == 'yes':
            if self.verbose:
                print("\n=== Time Series: ENABLED (forced) ===")

            # Must have datetime column
            if self.datetime_col is None:
                if self.verbose:
                    print("Error: enable_time_series='yes' requires datetime_col to be specified")
                raise ValueError("datetime_col must be specified when enable_time_series='yes'")

            # Get feature columns
            if isinstance(X, pd.DataFrame):
                feature_cols = self._identify_feature_columns(X)
            else:
                feature_cols = self.feature_columns or [f'feature_{i}' for i in range(X.shape[1])]

            # Detect optimal windows using DFT
            if self.user_provided_windows is None:
                if self.verbose:
                    print(f"\nRunning {self.window_detector_type.upper()} to detect optimal window sizes...")

                try:
                    self.detected_windows, self.confidence_scores = \
                        self.window_detector.detect_optimal_windows(
                            self.original_data if isinstance(X, pd.DataFrame) else pd.DataFrame(X),
                            self.datetime_col,
                            feature_cols,
                            sampling_rate=self.time_step
                        )

                    self.window_sizes = self.detected_windows
                    self.detection_strategy = 'dft'

                    if self.verbose:
                        avg_conf = np.mean(list(self.confidence_scores.values()))
                        print(f"{self.window_detector_type.upper()} detected {len(self.window_sizes)} windows with avg confidence: {avg_conf:.2f}")

                except Exception as e:
                    if self.verbose:
                        print(f"Warning: DFT detection failed: {str(e)}")
                        print("Falling back to default windows")
                    self.window_sizes = self._get_default_windows()
                    self.detection_strategy = 'default'
            else:
                # Use user-provided windows
                self.window_sizes = self._parse_time_periods(self.user_provided_windows)
                self.detection_strategy = 'user_provided'
                if self.verbose:
                    print(f"Using {len(self.window_sizes)} user-provided window sizes")

            # Set lag periods
            if self.user_provided_lags is None:
                # Derive from window sizes
                self.lag_periods = [
                    self.window_sizes[0],  # Smallest window
                    self.window_sizes[min(1, len(self.window_sizes) - 1)],
                    self.window_sizes[min(len(self.window_sizes) // 2, len(self.window_sizes) - 1)]
                ]
            else:
                self.lag_periods = self._parse_time_periods(self.user_provided_lags)

            self.enable_time_series = True
            self._add_time_series_operators()
            return True

        # Case 3: Auto mode
        if mode == 'auto':
            if self.verbose:
                print("\n=== Time Series: AUTO-DETECTION ===")

            # Step 1: Try to find datetime column
            if self.datetime_col is None:
                if isinstance(X, pd.DataFrame):
                    detected_dt_col = self.window_detector.detect_datetime_column(X)
                    if detected_dt_col:
                        self.datetime_col = detected_dt_col
                    else:
                        if self.verbose:
                            print("No datetime column found → Time series DISABLED")
                        self.enable_time_series = False
                        return False
                else:
                    if self.verbose:
                        print("Input is not a DataFrame, cannot auto-detect datetime → Time series DISABLED")
                    self.enable_time_series = False
                    return False

            # Step 2: Get feature columns
            if isinstance(X, pd.DataFrame):
                feature_cols = self._identify_feature_columns(X)
            else:
                feature_cols = self.feature_columns or [f'feature_{i}' for i in range(X.shape[1])]

            # Step 3: Run DFT and assess periodicity
            if self.verbose:
                print(f"\nAssessing periodicity using {self.window_detector_type.upper()}...")

            try:
                is_periodic, avg_confidence, feature_confidences = \
                    self.window_detector.assess_periodicity(
                        self.original_data if isinstance(X, pd.DataFrame) else pd.DataFrame(X),
                        self.datetime_col,
                        feature_cols
                    )

                # Step 4: Decide based on confidence
                if is_periodic:
                    if self.verbose:
                        print(
                            f"✓ Periodicity detected (confidence={avg_confidence:.2f} > {self.dft_confidence_threshold})")
                        print(f"  → Time series ENABLED with {self.window_detector_type.upper()}-detected windows")
                    # Use smart window selection
                    self.window_sizes, self.detection_strategy = \
                        self.window_detector.smart_window_selection(
                            self.original_data if isinstance(X, pd.DataFrame) else pd.DataFrame(X),
                            self.datetime_col,
                            feature_cols
                        )

                    self.detected_windows = self.window_sizes
                    self.confidence_scores = feature_confidences

                    # Set lag periods
                    if self.user_provided_lags is None:
                        self.lag_periods = [
                            self.window_sizes[0],
                            self.window_sizes[min(1, len(self.window_sizes) - 1)],
                            self.window_sizes[min(len(self.window_sizes) // 2, len(self.window_sizes) - 1)]
                        ]
                    else:
                        self.lag_periods = self._parse_time_periods(self.user_provided_lags)

                    self.enable_time_series = True
                    self._add_time_series_operators()
            except:
                pass
            return

    def _initialize_operator_weights(self):
        """
        Initialize operator weights with enhanced time series weighting
        """
        # Start with equal weights for all operators
        self.imp_operators = np.ones(len(self.operators))

        if self.enable_time_series and hasattr(self, 'time_series_operators'):
            # Apply enhanced weighting to time series operations
            for i, op in enumerate(self.operators):
                if op in self.time_series_operators:
                    self.imp_operators[i] *= self.ts_operation_weight_multiplier
                    if self.verbose:
                        op_name = getattr(op, '__name__', str(op))
                        print(f"Applied {self.ts_operation_weight_multiplier}x weight to {op_name}")

        # Normalize weights
        self.operator_weights = self.imp_operators / self.imp_operators.sum()

        if self.verbose and self.enable_time_series:
            ts_weight_sum = sum(self.operator_weights[i] for i, op in enumerate(self.operators)
                                if hasattr(self, 'time_series_operators') and op in self.time_series_operators)
            print(f"Time series operations total weight: {ts_weight_sum:.3f} ({ts_weight_sum * 100:.1f}%)")

    def _select_window_step(self):
        """
        Randomly select a window step from available options - FIXED VERSION
        """
        if hasattr(self, 'rng') and self.window_step_options:
            # Only select from valid options for time-based operations
            valid_options = [opt for opt in self.window_step_options if opt in ['D', 'H', 'W']]
            if valid_options:
                selected_step = self.rng.choice(valid_options)
                if self.verbose:
                    print(f"Selected window step: {selected_step}")
                return selected_step
        return 'D'  # Default to daily if no valid options

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
        sort_cols = [self.datetime_col] if self.datetime_col in df.columns else []
        sort_cols.extend([col for col in self.groupby_cols if col in df.columns])

        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        return df

    def _apply_time_based_operation(self, data, feature_col, operation, window_size=None, lag_period=None):
        """
        Apply time-based series operation to a specific feature column with proper grouping - FIXED VERSION

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

            # Dynamically select window step for this operation
            current_step = self._select_window_step()

            # Check if we have groupby columns
            if self.groupby_cols and any(col in data.columns for col in self.groupby_cols):
                # Group data by groupby columns
                groupby_cols = [col for col in self.groupby_cols if col in data.columns]

                results = []
                indices = []

                for name, group in data.groupby(groupby_cols, sort=False):
                    # Sort group by datetime
                    group_sorted = group.sort_values(self.datetime_col)
                    group_with_index = group_sorted.set_index(self.datetime_col)

                    # Apply operation to this group
                    group_result = self._apply_single_group_operation(
                        group_with_index, feature_col, operation,
                        window_size, lag_period, current_step
                    )

                    # Store results with original indices
                    results.append(group_result.values)
                    indices.extend(group_sorted.index.tolist())

                # Combine results maintaining original order
                result_series = pd.Series(
                    np.concatenate(results),
                    index=indices
                )
                # Reindex to match original data order
                result = result_series.reindex(data.index).fillna(0).values

            else:
                # Single group operation
                data_sorted = data.sort_values(self.datetime_col)
                data_with_index = data_sorted.set_index(self.datetime_col)

                result_series = self._apply_single_group_operation(
                    data_with_index, feature_col, operation,
                    window_size, lag_period, current_step
                )

                # Reindex to match original data order
                result = result_series.reindex(
                    data.set_index(self.datetime_col).index
                ).fillna(0).values

            return result

        except Exception as e:
            if self.verbose:
                print(f"    Warning: Time-based operation {operation} failed for {feature_col}: {str(e)}")
            return np.zeros(len(data))

    def _apply_single_group_operation(self, data, feature_col, operation, window_size=None, lag_period=None,
                                      time_step=None):
        """
        Apply operation to a single group with datetime index - FIXED VERSION

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
        time_step : str, optional
            Time step for operations

        Returns:
        --------
        pd.Series
            Result series with datetime index
        """
        series = data[feature_col]
        current_time_step = time_step or self.time_step

        try:
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
                # FIX: Use shift with freq parameter for time-aware shifting
                result = series.shift(freq=lag_period)
                # Reindex to match original index and forward fill
                result = result.reindex(series.index, method='ffill').fillna(0)

            elif operation == 'diff':
                lag_period = lag_period or self.rng.choice(self.lag_periods)
                # FIX: Use shift with freq parameter
                lagged = series.shift(freq=lag_period)
                lagged = lagged.reindex(series.index, method='ffill').fillna(0)
                result = series - lagged
                result = result.fillna(0)

            elif operation == 'pct_change':
                lag_period = lag_period or self.rng.choice(self.lag_periods)
                # FIX: Use shift with freq parameter
                lagged = series.shift(freq=lag_period)
                lagged = lagged.reindex(series.index, method='ffill').fillna(0)
                # Avoid division by zero
                result = pd.Series(index=series.index, dtype=float)
                mask = lagged != 0
                result[mask] = (series[mask] - lagged[mask]) / lagged[mask]
                result[~mask] = 0
                result = result.fillna(0)

            elif operation == 'ewm':
                window_size = window_size or self.rng.choice(self.window_sizes)
                # FIX: Convert timedelta to numeric span for EWM
                # Use approximate number of periods based on time_step
                if current_time_step == 'D':
                    span = window_size.days
                elif current_time_step == 'H':
                    span = window_size.total_seconds() / 3600
                elif current_time_step == 'W':
                    span = window_size.days / 7
                elif current_time_step == 'M':
                    span = window_size.days / 30
                else:
                    span = window_size.days

                # Ensure span is at least 1
                span = max(1, int(span))
                result = series.ewm(span=span, adjust=False).mean()

            elif operation == 'momentum':
                lag_period = lag_period or self.rng.choice(self.lag_periods)
                # Momentum is just difference, use the fixed diff logic
                lagged = series.shift(freq=lag_period)
                lagged = lagged.reindex(series.index, method='ffill').fillna(0)
                result = series - lagged
                result = result.fillna(0)

            elif operation == 'seasonal_decompose':
                try:
                    # Simple seasonal pattern based on day of year
                    if len(series) > 7:
                        # Group by day of year and calculate mean
                        seasonal_means = series.groupby(series.index.dayofyear).transform('mean')
                        result = seasonal_means
                    else:
                        result = pd.Series(series.mean(), index=series.index)
                except Exception:
                    result = pd.Series(series.mean(), index=series.index)

            elif operation == 'trend':
                window_size = window_size or self.rng.choice(self.window_sizes)
                # Calculate trend as rolling linear regression slope
                result = series.rolling(window=window_size, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    raw=True
                ).fillna(0)

            elif operation == 'weekday_mean':
                # Calculate mean for each weekday
                weekday_means = series.groupby(series.index.dayofweek).transform('mean')
                result = weekday_means

            elif operation == 'month_mean':
                # Calculate mean for each month
                month_means = series.groupby(series.index.month).transform('mean')
                result = month_means

            else:
                result = pd.Series(0, index=series.index)

            return result

        except Exception as e:
            # If operation fails, return zeros
            if self.verbose:
                print(f"    Warning: Operation {operation} failed: {str(e)}")
            return pd.Series(0, index=series.index)

    # Time Series Utility Methods
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

    def fit(self, X, y, gen_size=5, random_state=0, iterations=5, estimator='avg',
            feat_imps=True, split_feats=None, check_corr=True, selection='stability', combine_res=True):
        """
        Generate Features using test set - Enhanced for DFT-based time series detection

        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        y : array-like
            Target variable
        gen_size : int, default=5
            Number of features to generate per iteration
        random_state : int, default=0
            Random seed for reproducibility
        iterations : int, default=5
            Number of feature generation iterations
        estimator : str, default='avg'
            Estimator type for feature importance ('avg', 'rf', 'rf_reg')
        feat_imps : bool, default=True
            Whether to compute feature importances
        split_feats : str, optional
            Feature split strategy ('comb', 'splits', or None)
        check_corr : bool, default=True
            Whether to check and remove correlated features
        selection : str, default='stability'
            Feature selection method ('stability' or 'fAnova')
        combine_res : bool, default=True
            Whether to combine results across iterations

        Returns:
        --------
        gen_feats : ndarray
            Generated and selected features
        """

        if self.verbose:
            print("\n" + "=" * 60)
            print("BigFeat Feature Generation Started")
            print("=" * 60)

        # Store original data if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            self.original_data = X.copy()
        else:
            self.original_data = X

        # === CRITICAL: Setup time series based on mode ===
        ts_enabled = self._setup_time_series(X, y)

        # Identify and extract feature columns
        if isinstance(X, pd.DataFrame):
            self.feature_columns = self._identify_feature_columns(X)

            if len(self.feature_columns) == 0:
                raise ValueError("No numeric feature columns found after filtering!")

            X_features = X[self.feature_columns].values
            X_features = np.array(X_features, dtype=float)

            if self.verbose:
                print(f"\nIdentified {len(self.feature_columns)} numeric feature columns")
        else:
            X_features = np.array(X, dtype=float)
            if hasattr(X, 'columns'):
                self.feature_columns = list(X.columns)
            else:
                self.feature_columns = [f'feature_{i}' for i in range(X_features.shape[1])]

        # Prepare time series data if enabled
        if self.enable_time_series:
            self._current_data = self._prepare_time_series_data(X)
            if self.verbose:
                print(f"✓ Time series data prepared with datetime column: '{self.datetime_col}'")

        # === Original BigFeat initialization ===
        self.selection = selection
        self.imp_operators = np.ones(len(self.operators))
        self.operator_weights = self.imp_operators / self.imp_operators.sum()
        self.gen_steps = []
        self.n_feats = X_features.shape[1]
        self.n_rows = X_features.shape[0]
        self.ig_vector = np.ones(self.n_feats) / self.n_feats
        self.comb_mat = np.ones((self.n_feats, self.n_feats))
        self.split_vec = np.ones(self.n_feats)

        # Set RNG seed
        self.rng = np.random.RandomState(seed=random_state)

        # Initialize enhanced operator weights (includes TS if enabled)
        self._initialize_operator_weights()

        # Initialize feature arrays
        gen_feats = np.zeros((self.n_rows, self.n_feats * gen_size))
        iters_comb = np.zeros((self.n_rows, self.n_feats * iterations))
        depths_comb = np.zeros(self.n_feats * iterations)
        ids_comb = np.zeros(self.n_feats * iterations, dtype=object)
        ops_comb = np.zeros(self.n_feats * iterations, dtype=object)
        self.feat_depths = np.zeros(gen_feats.shape[1])
        self.depth_range = np.arange(3) + 1
        self.depth_weights = 1 / (2 ** self.depth_range)
        self.depth_weights /= self.depth_weights.sum()

        # Scaling
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_features)
        X_scaled = self.scaler.transform(X_features)

        # Feature importance calculation
        if feat_imps:
            if self.verbose:
                print(f"\nComputing feature importances using '{estimator}' estimator...")

            self.ig_vector, estimators = self.get_feature_importances(
                X_scaled, y, estimator, random_state
            )
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

        # === Feature Generation Iterations ===
        if self.verbose:
            print(f"\nStarting {iterations} feature generation iterations...")

        for iteration in range(iterations):
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
                if self.enable_time_series:
                    ts_weight_sum = sum(
                        self.operator_weights[i] for i, op in enumerate(self.operators)
                        if hasattr(self, 'time_series_operators') and op in self.time_series_operators
                    )
                    print(f"  Time series operations weight: {ts_weight_sum:.3f} ({ts_weight_sum * 100:.1f}%)")

            self.tracking_ops = []
            self.tracking_ids = []
            gen_feats = np.zeros((self.n_rows, self.n_feats * gen_size))
            self.feat_depths = np.zeros(gen_feats.shape[1])

            # Generate features
            for i in range(gen_feats.shape[1]):
                dpth = self.rng.choice(self.depth_range, p=self.depth_weights)
                ops = []
                ids = []
                gen_feats[:, i] = self.feat_with_depth(X_scaled, dpth, ops, ids)

                # Clean generated feature if time series is enabled
                if self.enable_time_series:
                    gen_feats[:, i] = self._clean_feature(gen_feats[:, i])

                self.feat_depths[i] = dpth
                self.tracking_ops.append(ops)
                self.tracking_ids.append(ids)

            self.tracking_ids = np.array(self.tracking_ids + [[]], dtype='object')[:-1]
            self.tracking_ops = np.array(self.tracking_ops + [[]], dtype='object')[:-1]

            # Feature selection within iteration
            if self.verbose:
                print(f"  Selecting top {self.n_feats} features from {gen_feats.shape[1]} generated...")

            imps, estimators = self.get_feature_importances(gen_feats, y, estimator, random_state)
            total_feats = np.argsort(imps)
            feat_args = total_feats[-self.n_feats:]
            gen_feats = gen_feats[:, feat_args]
            self.tracking_ids = self.tracking_ids[feat_args]
            self.tracking_ops = self.tracking_ops[feat_args]
            self.feat_depths = self.feat_depths[feat_args]

            # Store iteration results
            depths_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.feat_depths
            ids_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.tracking_ids
            ops_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.tracking_ops
            iters_comb[:, iteration * self.n_feats:(iteration + 1) * self.n_feats] = gen_feats

            # Update operator importance
            for i, op in enumerate(self.operators):
                for feat in self.tracking_ops:
                    for feat_op in feat:
                        if op == feat_op[0]:
                            weight_increment = 1
                            if hasattr(self, 'time_series_operators') and op in self.time_series_operators:
                                weight_increment *= self.ts_operation_weight_multiplier
                            self.imp_operators[i] += weight_increment

            self.operator_weights = self.imp_operators / self.imp_operators.sum()

        if self.verbose and self.enable_time_series:
            ts_weight_sum = sum(
                self.operator_weights[i] for i, op in enumerate(self.operators)
                if hasattr(self, 'time_series_operators') and op in self.time_series_operators
            )
            print(f"\n✓ Final time series operations weight: {ts_weight_sum:.3f} ({ts_weight_sum * 100:.1f}%)")

        # === Final Feature Selection ===
        if selection == 'stability' and iterations > 1 and combine_res:
            if self.verbose:
                print(f"\nCombining results across {iterations} iterations...")

            imps, estimators = self.get_feature_importances(iters_comb, y, estimator, random_state)
            total_feats = np.argsort(imps)
            feat_args = total_feats[-self.n_feats:]
            gen_feats = iters_comb[:, feat_args]
            self.tracking_ids = ids_comb[feat_args]
            self.tracking_ops = ops_comb[feat_args]
            self.feat_depths = depths_comb[feat_args]

        # Check correlations
        if selection == 'stability' and check_corr:
            if self.verbose:
                print("Checking for correlated features...")

            gen_feats, to_drop_cor = self.check_correlations(gen_feats)
            self.tracking_ids = np.delete(self.tracking_ids, to_drop_cor)
            self.tracking_ops = np.delete(self.tracking_ops, to_drop_cor)
            self.feat_depths = np.delete(self.feat_depths, to_drop_cor)

            if self.verbose and len(to_drop_cor) > 0:
                print(f"  Removed {len(to_drop_cor)} correlated features")

        # Combine with original features
        gen_feats = np.hstack((gen_feats, X_scaled))

        # fAnova selection
        if selection == 'fAnova':
            if self.verbose:
                print(f"Applying fAnova selection (k={self.n_feats})...")

            if self.task_type == 'classification':
                self.fAnova_best = SelectKBest(f_classif, k=self.n_feats)
            else:
                self.fAnova_best = SelectKBest(f_regression, k=self.n_feats)
            gen_feats = self.fAnova_best.fit_transform(gen_feats, y)

        # === Final Summary ===
        if self.verbose:
            print("\n" + "=" * 60)
            print("Feature Generation Completed")
            print("=" * 60)
            print(f"Final feature shape: {gen_feats.shape}")
            print(f"Original features: {X_features.shape[1]}")
            print(f"Generated features: {gen_feats.shape[1] - X_features.shape[1]}")

            if self.enable_time_series:
                ts_ops_count = sum(
                    1 for ops in self.tracking_ops
                    for op_info in ops
                    if len(op_info) > 0 and callable(op_info[0]) and
                    hasattr(self, 'time_series_operators') and
                    op_info[0] in self.time_series_operators
                )
                print(f"Time series operations used: {ts_ops_count}")

            print("=" * 60 + "\n")

            # Print DFT summary if time series was used
            if self.enable_time_series and self.detection_strategy:
                self.print_window_detection_summary()

        return gen_feats

    # Method to update time series weights dynamically
    def update_ts_weight_multiplier(self, new_multiplier):
        """
        Update the time series operation weight multiplier

        Parameters:
        -----------
        new_multiplier : float
            New weight multiplier for time series operations
        """
        old_multiplier = self.ts_operation_weight_multiplier
        self.ts_operation_weight_multiplier = new_multiplier

        if hasattr(self, 'imp_operators') and hasattr(self, 'time_series_operators'):
            # Update existing weights
            for i, op in enumerate(self.operators):
                if op in self.time_series_operators:
                    # Remove old multiplier and apply new one
                    self.imp_operators[i] = (self.imp_operators[i] / old_multiplier) * new_multiplier

            # Renormalize
            self.operator_weights = self.imp_operators / self.imp_operators.sum()

            if self.verbose:
                print(f"Updated time series weight multiplier from {old_multiplier} to {new_multiplier}")

    # Method to update window step options
    def update_window_step_options(self, new_options):
        """
        Update the available window step options

        Parameters:
        -----------
        new_options : list
            New list of window step options
        """
        self.window_step_options = new_options
        if self.verbose:
            print(f"Updated window step options to: {self.window_step_options}")

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

    def _get_default_windows(self):
        """
        Get default window sizes when DFT is not used or fails

        Returns:
        --------
        list of pd.Timedelta
            Default window sizes
        """
        default_windows = [
            pd.Timedelta(days=7),
            pd.Timedelta(days=14),
            pd.Timedelta(days=30),
            pd.Timedelta(days=90),
            pd.Timedelta(days=180),
            pd.Timedelta(days=365)
        ]

        # Filter by constraints if DFT parameters are set
        if hasattr(self, 'dft_min_window_days') and hasattr(self, 'dft_max_window_days'):
            filtered = [w for w in default_windows
                        if self.dft_min_window_days <= w.days <= self.dft_max_window_days]

            # Return top n_windows if that parameter exists
            if hasattr(self, 'dft_n_windows') and len(filtered) > self.dft_n_windows:
                return filtered[:self.dft_n_windows]
            return filtered

        return default_windows

    def _add_time_series_operators(self):
        """
        Add time series operators to the operator list
        Called when time series is enabled
        """
        # Check if already added to avoid duplicates
        if hasattr(self, '_ts_operators_added') and self._ts_operators_added:
            return

        # Define time series operators if not already defined
        if not hasattr(self, 'time_series_operators'):
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

        # Mark as added
        self._ts_operators_added = True

        if self.verbose:
            print(f"\n✓ Added {len(self.time_series_operators)} time series operators")
            print(f"  Window sizes: {[str(w) for w in self.window_sizes]}")
            print(f"  Lag periods: {[str(l) for l in self.lag_periods]}")

    def _identify_feature_columns(self, X):
        """
        Identify numeric feature columns, excluding datetime and groupby columns

        Parameters:
        -----------
        X : DataFrame
            Input dataframe

        Returns:
        --------
        list
            List of feature column names
        """
        if not isinstance(X, pd.DataFrame):
            # If not a DataFrame, return all columns or generate names
            if hasattr(X, 'shape'):
                return [f'feature_{i}' for i in range(X.shape[1])]
            return []

        # Exclude datetime and groupby columns
        exclude_cols = []
        if self.datetime_col and self.datetime_col in X.columns:
            exclude_cols.append(self.datetime_col)
        if hasattr(self, 'groupby_cols') and self.groupby_cols:
            exclude_cols.extend([col for col in self.groupby_cols if col in X.columns])

        # Get all potential feature columns
        all_feature_cols = [col for col in X.columns if col not in exclude_cols]

        # Filter out non-numeric columns
        numeric_feature_cols = []
        for col in all_feature_cols:
            col_dtype = str(X[col].dtype)

            # Skip datetime-like columns
            if (col_dtype.startswith('datetime') or
                    col_dtype.startswith('<M8') or
                    col_dtype.startswith('timedelta')):
                if self.verbose:
                    print(f"  Skipping datetime column '{col}' (type: {col_dtype})")
                continue

            # Check object columns more carefully
            if col_dtype == 'object':
                try:
                    sample_val = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else None
                    if sample_val is not None and hasattr(sample_val, 'year'):
                        # Likely a datetime object
                        if self.verbose:
                            print(f"  Skipping datetime-like column '{col}'")
                        continue
                except:
                    pass

            # Test if column is numeric
            try:
                test_val = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else 0
                float(test_val)
                numeric_feature_cols.append(col)
            except (ValueError, TypeError):
                if self.verbose:
                    print(f"  Skipping non-numeric column '{col}' (type: {col_dtype})")
                continue

        return numeric_feature_cols

    def get_window_detection_summary(self):
        """
        Get summary of window detection results

        Returns:
        --------
        dict
            Dictionary containing window detection summary
        """
        summary = {
            'mode': self.enable_time_series_mode,
            'time_series_enabled': self.enable_time_series,
            'datetime_col': self.datetime_col,
            'detection_strategy': getattr(self, 'detection_strategy', None),
            'window_sizes': [w.days for w in self.window_sizes] if self.window_sizes else None,
            'lag_periods': [l.days for l in self.lag_periods] if self.lag_periods else None,
            'confidence_scores': getattr(self, 'confidence_scores', {}),
            'avg_confidence': None
        }

        if summary['confidence_scores']:
            summary['avg_confidence'] = np.mean(list(summary['confidence_scores'].values()))

        return summary

    def print_window_detection_summary(self):
        """
        Print a formatted summary of window detection results
        """
        summary = self.get_window_detection_summary()

        print("\n" + "=" * 60)
        print("BigFeat Time Series Configuration Summary")
        print("=" * 60)
        print(f"Mode: {summary['mode']}")
        print(f"Window Detector: {self.window_detector_type.upper()}")
        print(f"Time Series Enabled: {summary['time_series_enabled']}")

        if summary['time_series_enabled']:
            print(f"Datetime Column: {summary['datetime_col']}")
            print(f"Detection Strategy: {summary['detection_strategy']}")
            print(f"\nWindow Sizes (days): {summary['window_sizes']}")
            print(f"Lag Periods (days): {summary['lag_periods']}")

            if summary['confidence_scores']:
                print(f"\nConfidence Scores:")
                for feat, conf in summary['confidence_scores'].items():
                    status = "STRONG" if conf > 2.0 else "MODERATE" if conf > 1.3 else "WEAK"
                    print(f"  {feat}: {conf:.2f} ({status})")
                if summary['avg_confidence']:
                    print(f"\nAverage Confidence: {summary['avg_confidence']:.2f}")
                    print(f"Threshold: {self.dft_confidence_threshold}")
        else:
            print(f"Reason: {summary['detection_strategy']}")

        print("=" * 60 + "\n")
