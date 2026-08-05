import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import bigfeat.local_utils as local_utils
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.tree import _tree
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from bigfeat.dft_window_detector import DFTWindowDetector
from functools import partial
import warnings
from datetime import timedelta
import psutil  # For dynamic memory-aware sampling


class BigFeat:
    def __init__(self,
                 task_type='classification',
                 enable_time_series='auto',  # 'yes'/'no'/'auto'
                 window_detector='ensemble',  # 'dft'/'acf'/'lomb_scargle/'ensemble'
                 window_sizes=None,
                 lag_periods=None,
                 verbose=True,
                 datetime_col=None,
                 groupby_cols=None,
                 time_step='D',
                 ts_operation_weight_multiplier=1.0,
                 window_step_options=None,

                 # DFT-related parameters
                 confidence_threshold=0.5,
                 min_window_days=1,
                 max_window_days=365,
                 n_windows=6,

                 # Downsampling parameters (NEW)
                 enable_downsampling=False,
                 max_fit_samples=0,
                 downsampling_random_state=42):
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

        confidence_threshold : float, default=0.5
            Minimum confidence score to consider periodicity reliable
            Used in 'auto' mode to decide whether to enable time series

        min_window_days : int, default=3
            Minimum window size in days for window detection

        max_window_days : int, default=365
            Maximum window size in days for window detection

        n_windows : int, default=6
            Number of window sizes to generate from window analysis

        enable_downsampling : bool, default=False
            Enable automatic downsampling for large datasets during fit().
            Uses DYNAMIC MEMORY-AWARE sampling that adapts to available system RAM.

            How it works:
            1. Detects available system memory using psutil
            2. Calculates safe sample size based on RAM and estimated feature count
            3. Uses smaller of: dynamic limit OR max_fit_samples (if set)
            4. Feature discovery (fit): Uses calculated sample
            5. Feature application (transform): Uses ALL rows

            This prevents out-of-memory errors while maximizing data usage:
            - On 8GB laptop: Might sample 50K rows
            - On 64GB server: Might sample 1M rows
            - Automatically adapts without manual tuning

            IMPORTANT for time series: Random sampling breaks temporal continuity.
            During fit(), rolling windows will average discontinuous time points,
            which may produce noisy feature importances. However, transform() on
            the full dataset will have proper time continuity. This is a deliberate
            trade-off: OOM protection vs perfect time coherence during discovery.

        max_fit_samples : int, default=0
            Upper bound for downsampling when enable_downsampling=True.

            Behavior:
            - If > 0: Acts as a cap on the dynamic sample size
              Example: If dynamic calculation suggests 500K but max_fit_samples=100K,
              it will use 100K (whichever is smaller)
            - If = 0: Fully dynamic mode - uses only memory-based calculation
              No upper limit except what memory allows

            Recommendation:
            - Keep 0 for fully adaptive behavior that maximizes memory usage
            - Set a constant number (e.g., 100K) for reproducible results across machines
            - Set higher (e.g., 500K) if you have abundant RAM and want more data

        downsampling_random_state : int, default=42
            Random seed for reproducible downsampling when enable_downsampling=True.
            Ensures same sample is selected across runs for reproducibility.
        """
        # Original initialization
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
        self.confidence_threshold = confidence_threshold
        self.min_window_days = min_window_days
        self.max_window_days = max_window_days
        self.n_windows = n_windows

        # Downsampling parameters (NEW)
        self.enable_downsampling = enable_downsampling
        self.max_fit_samples = max_fit_samples
        self.downsampling_random_state = downsampling_random_state

        # Initialize the appropriate window detector
        self.window_detector_type = window_detector
        
        # KEY CHANGE: Ensure detector is sensitive enough (0.3) even if our strict threshold is higher (0.5)
        # We want to detect "Moderate" periodicity (0.3-0.5) to enable restricted mode
        detection_threshold = min(0.3, confidence_threshold)
        
        self.window_detector = self._initialize_window_detector(
            window_detector,
            min_window_days,
            max_window_days,
            n_windows,
            detection_threshold,
            verbose
        )

        # Time series parameters
        self.ts_operation_weight_multiplier = ts_operation_weight_multiplier

        # Window step options
        if window_step_options is None:
            self.window_step_options = ['D', 'H', 'W', 'M']
        else:
            self.window_step_options = list(window_step_options)

        self.time_step = time_step

        # Store user-provided windows (will be overridden by DFT if needed)
        self.user_provided_windows = window_sizes
        self.user_provided_lags = lag_periods

        # These will be set during fit() based on mode
        self.window_sizes = None
        self.lag_periods = None

        # Parameters for date/time column
        self.datetime_col = datetime_col
        if groupby_cols is None:
            self.groupby_cols = []
        else:
            self.groupby_cols = groupby_cols
        self.original_data = None
        self.feature_columns = None

        # Tracking variables for DFT results
        self.detected_windows = None
        self.confidence_scores = None
        self.detection_strategy = None

        # Tracking variables for current data state
        self._current_data = None
        self._current_feature_index = None

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
        elif detector_type in ['ensemble', 'standard']:
            # Default to DFT for initial setup; auto-mode ensemble logic will override this later
            from bigfeat.dft_window_detector import DFTWindowDetector
            return DFTWindowDetector(
                min_window_days=min_window_days,
                max_window_days=max_window_days,
                n_windows=n_windows,
                confidence_threshold=confidence_threshold,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown window_detector: {detector_type}. "
                             f"Must be 'dft', 'acf', 'lomb_scargle', 'ensemble', or 'standard'")

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
                if self.feature_columns is not None and len(self.feature_columns) > 0:
                    feature_cols = self.feature_columns
                else:
                    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]

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
                            sampling_rate=self.time_step,
                            groupby_cols=self.groupby_cols
                        )

                    self.window_sizes = self.detected_windows
                    self.detection_strategy = self.window_detector_type

                    if self.verbose:
                        avg_conf = np.mean(list(self.confidence_scores.values()))
                        print(
                            f"{self.window_detector_type.upper()} detected {len(self.window_sizes)} windows with avg confidence: {avg_conf:.2f}")

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
                print(f"Assessing data stationarity and periodicity...")

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
                if self.feature_columns is not None and len(self.feature_columns) > 0:
                     feature_cols = self.feature_columns
                else:
                     feature_cols = [f'feature_{i}' for i in range(X.shape[1])]

            # Step 3: The Stationarity Gate (NEW)
            # Calculate avg lag-1 autocorrelation BEFORE periodicity voting
            try:
                # Defensive check for DataFrame conversion (Task 11)
                if isinstance(X, pd.DataFrame):
                    df = self.original_data if hasattr(self, 'original_data') else X
                elif hasattr(self, 'original_data') and isinstance(self.original_data, pd.DataFrame):
                     df = self.original_data
                else:
                    df = pd.DataFrame(X)

                check_cols = feature_cols[:5] # Check a wider sample
                lag1_corrs = []
                for col in check_cols:
                    if col in df.columns:
                        series = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        corr = abs(series.autocorr(lag=1))
                        if not np.isnan(corr): lag1_corrs.append(corr)

                avg_lag1 = np.mean(lag1_corrs) if lag1_corrs else 0.0
                self.avg_lag1 = avg_lag1 # Store as class attribute for benchmarking
                # FIX: Enforce scalar types
                if hasattr(avg_lag1, 'item'): avg_lag1 = avg_lag1.item()
                is_highly_non_stationary = avg_lag1 > 0.85  # Very strong trend/random walk
                
                if self.verbose and is_highly_non_stationary:
                    print(f"  ⚠ High Non-Stationarity detected (Avg Lag-1 Corr: {avg_lag1:.3f})")

            except Exception as e:
                is_highly_non_stationary = False
                if self.verbose:
                    print(f"Stationarity check failed: {e}")

            # Step 4: Ensemble Detector Voting
            if self.verbose:
                print(f"Assessing periodicity using Ensemble Voting (DFT, ACF, Lomb-Scargle)...")

            detectors = ['dft', 'acf', 'lomb_scargle']
            votes = []
            detector_results = {}
            strongest_detector = None
            max_confidence = -1.0

            # Collect votes from all detectors
            for det_type in detectors:
                try:
                    # Initialize detector
                    # Note: We use a lower threshold (0.3) for voting eligibility to catch weaker signals
                    # But individual votes still depend on the detector's internal logic
                    det = self._initialize_window_detector(
                        det_type,
                        self.min_window_days,
                        self.max_window_days,
                        self.n_windows,
                        self.confidence_threshold, # Use configured threshold
                        verbose=False # Keep it quiet during voting
                    )
                    
                    is_periodic, conf, feat_confs = det.assess_periodicity(
                        self.original_data if isinstance(X, pd.DataFrame) else pd.DataFrame(X),
                        self.datetime_col,
                        feature_cols,
                        groupby_cols=self.groupby_cols
                    )

                    # FIX: Enforce scalar types to prevent "ambiguous truth value" error
                    if hasattr(is_periodic, 'item'): is_periodic = is_periodic.item()
                    if hasattr(conf, 'item'): conf = conf.item()

                    is_periodic = bool(is_periodic)
                    conf = float(conf)
                    
                    detector_results[det_type] = {
                        'is_periodic': is_periodic,
                        'confidence': conf,
                        'feature_confidences': feat_confs,
                        'instance': det
                    }
                    
                    if is_periodic:
                        votes.append(det_type)
                        if self.verbose:
                             print(f"  ✓ {det_type.upper()}: Periodic (conf={conf:.2f})")
                    else:
                        if self.verbose:
                             print(f"  - {det_type.upper()}: Non-periodic (conf={conf:.2f})")
                             
                    if conf > max_confidence:
                        max_confidence = conf
                        strongest_detector = det_type

                except Exception as e:
                    if self.verbose:
                        print(f"  ! {det_type.upper()} failed: {e}")

            # Consensus Decision
            # We require majority (>=2) OR a very strong single vote (>0.7)
            has_consensus = len(votes) >= 2
            strong_signal = max_confidence > 0.7
            
            is_periodic_final = has_consensus or strong_signal
            
            if is_periodic_final:
                # NEW: Multi-Detector Window Pooling
                # Instead of picking ONE winner, pool windows from all valid detectors
                all_candidate_windows = []
                total_conf = 0
                valid_detectors_count = 0
                
                # We iterate through all detectors to pool distinct windows
                for det_type, res in detector_results.items():
                    if res['is_periodic']:
                        # Get windows from this specific detector
                        det_windows, strategy_suffix = res['instance'].smart_window_selection(
                            self.original_data if isinstance(X, pd.DataFrame) else pd.DataFrame(X),
                            self.datetime_col,
                            feature_cols,
                            groupby_cols=self.groupby_cols
                        )
                        all_candidate_windows.extend(det_windows)
                        total_conf += res['confidence']
                        valid_detectors_count += 1
                        
                        if self.verbose:
                             print(f"  + Pooling {len(det_windows)} windows from {det_type.upper()}")

                # Calculate consensus confidence
                self.avg_consensus_confidence = total_conf / max(1, valid_detectors_count)
                
                # Process pooled windows: Remove duplicates and limit to n_windows
                if all_candidate_windows:
                    unique_windows = sorted(set(all_candidate_windows))

                    # Keep a spread across the detected scale range rather than
                    # the n smallest.
                    #
                    # This used to be `sorted(...)[:n_windows]`, which truncates
                    # an ascending list and therefore always discards the LARGE
                    # windows. Pooling three detectors reliably produces more
                    # than n_windows candidates, so the long-range windows were
                    # dropped every time. On the Monash monthly data that left
                    # windows of 1-6 DAYS for series sampled once a month --
                    # every rolling feature collapsed to a single observation.
                    #
                    # Sampling at even quantiles keeps the shortest and longest
                    # detected scales plus a spread between them.
                    if len(unique_windows) > self.n_windows:
                        idx = np.linspace(0, len(unique_windows) - 1,
                                          self.n_windows).round().astype(int)
                        self.window_sizes = [unique_windows[i] for i in sorted(set(idx))]
                    else:
                        self.window_sizes = unique_windows


                    self.detection_strategy = "pooled_ensemble"
                    self.window_detector_type = "ensemble"
                else:
                    # Fallback (should be rare if is_periodic_final is True)
                    self.window_sizes = self._get_default_windows()
                    self.avg_consensus_confidence = 0.5
                    self.detection_strategy = "ensemble_fallback"
                    self.window_detector_type = "ensemble"

                # NEW: Safety Tiering with Stationarity Gate
                # Even if periodic, if it's highly non-stationary, force Restricted Mode
                if is_highly_non_stationary or self.avg_consensus_confidence < 0.5:
                    if self.verbose:
                        reason = "Non-Stationary Gate" if is_highly_non_stationary else "Moderate Conf"
                        print(f"⚠ {reason} triggered. Restricted TS enabled (Safe Ops Only).")
                    
                    # 1. Restrict the specialized TS operator list
                    self.time_series_operators = [self._safe_lag_feature, self._safe_diff_feature]

                    # 2. NEW: Remove rolling/seasonal ops from the active search pool
                    # This ensures the GA can ONLY pick Lag/Diff
                    # Use __name__ for safe string comparison of operators
                    restricted_names = ['_safe_rolling_mean', '_safe_rolling_std', '_safe_seasonal_decompose', '_safe_trend_feature']
                    self.operators = [op for op in self.operators if getattr(op, '__name__', '') not in restricted_names]
                else:
                    if self.verbose: 
                        print(f"✓ Strong & Stationary Signal (Conf: {self.avg_consensus_confidence:.2f}). Full TS enabled.")
                        print(f"  → Time series ENABLED using Pooled Windows: {[str(w) for w in self.window_sizes]}")
                        
                    if hasattr(self, 'time_series_operators'): 
                        delattr(self, 'time_series_operators')

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
                
            else:
                # --- Non-Periodic / Trend Check ---
                # Check for strong trend/auto-correlation (e.g., lag-1 correlation)
                # Using a simple check on the first feature (or average a few)
                
                if self.verbose: print("  → No strong periodicity consensus.")
                
                # Use the pre-calculated avg_lag1
                if is_highly_non_stationary or avg_lag1 > 0.8:  # Keep 0.8 threshold for non-periodic enabled fallback
                    if self.verbose: 
                        print(f"  ⚠ Non-periodic but STRONG TREND detected (avg lag-1 corr={avg_lag1:.2f}).")
                        print(f"  → Prioritizing stationarity operators (diff, pct_change).")
                    
                    # Override to enabled
                    self.enable_time_series = True
                    self.detection_strategy = "trend_stationarity"
                    
                    # Set restricted "Stationarity" operators
                    self.time_series_operators = [self._safe_diff_feature, self._safe_pct_change, self._safe_lag_feature]
                    
                    # Set minimal default windows
                    self.window_sizes = [pd.Timedelta(days=1), pd.Timedelta(days=7)] # Minimal set
                    self.lag_periods = [pd.Timedelta(days=1)]
                    
                    self._add_time_series_operators()
                    
                    self._trend_mode_active = True
                    
                else:
                    if self.verbose:
                        print(f"  → No strong trend (avg lag-1={avg_lag1:.2f}). Time series features DISABLED.")
                    self.enable_time_series = False

    def _initialize_operator_weights(self):
        """
        Initialize operator weights with enhanced time series weighting
        """
        # Start with equal weights for all operators
        self.imp_operators = np.ones(len(self.operators))

        if self.enable_time_series and hasattr(self, 'time_series_operators') and self.time_series_operators is not None:
            # Scale the multiplier based on detection confidence
            dynamic_multiplier = self.effective_ts_weight_multiplier

            if hasattr(self, 'avg_consensus_confidence'):
                # Boost weights if consensus is high, or penalize if signal is weak
                # If confidence is 0.8, multiplier stays near 1.0. If 0.3, it drops to 0.5.
                confidence_factor = np.clip(self.avg_consensus_confidence / 0.7, 0.5, 2.0)
                dynamic_multiplier *= confidence_factor
                if self.verbose:
                    print(f"  Adaptive Weighting: Confidence {self.avg_consensus_confidence:.2f} -> Factor {confidence_factor:.2f}")

            # Apply enhanced weighting to time series operations
            for i, op in enumerate(self.operators):
                if op in self.time_series_operators:
                    weight_mult = dynamic_multiplier
                    
                    # Boost stationarity operators if trend mode active
                    if getattr(self, '_trend_mode_active', False):
                        if op in [self._safe_diff_feature, self._safe_pct_change]:
                             weight_mult *= 3.0
                    
                    self.imp_operators[i] *= weight_mult
                    if self.verbose:
                        op_name = getattr(op, '__name__', str(op))
                        # print(f"Applied {weight_mult}x weight to {op_name}")

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
        if hasattr(self, 'rng') and self.window_step_options is not None and len(self.window_step_options) > 0:
            # Only select from valid options for time-based operations
            valid_options = [opt for opt in self.window_step_options if opt in ['D', 'H', 'W', 'M', 'Q', 'Y']]
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

        # NOTE: there was previously an "already sorted" short-circuit here,
        # keyed on a self._is_sorted flag. It was unsound. The flag was set the
        # first time this ran (during fit) and only ever reset at the top of
        # fit(), so every subsequent transform() returned X untouched --
        # skipping the datetime sort, the dtype coercion, and the
        # _original_index bookkeeping that the rest of transform() depends on.
        #
        # The result was that a row's time-series features were computed from
        # its position in the caller's array rather than from its timestamp,
        # and the fAnova branch at the end of transform() became unreachable.
        # Because the flag stayed set for *every* call, the corruption was
        # consistent, so it could not be detected by comparing two transform()
        # calls to each other -- only by checking invariance to input row
        # order. See tests/test_invariants.py.
        #
        # Sortedness is a property of the data passed in, not of the estimator,
        # so it cannot be cached on self. The check below is O(n) and cheap
        # relative to the feature generation that follows.

        # Convert to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # If we have stored feature columns, use them
            if self.feature_columns is not None:
                df = pd.DataFrame(X, columns=self.feature_columns)
            else:
                df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        # Fill in the datetime / groupby columns from the stored training data
        # ONLY where the caller's own frame does not already provide them.
        #
        # This used to overwrite unconditionally, grafting *training*
        # timestamps positionally onto whatever rows were passed in. When the
        # caller's rows were not in training order that paired each row with
        # someone else's timestamp, so the sort below produced a plausible but
        # wrong ordering, and _original_index no longer identified which input
        # row an output row came from.
        #
        # Positional alignment against self.original_data is only meaningful
        # when the incoming frame really is the training frame, in the training
        # row order -- which we cannot assume in transform().
        if self.original_data is not None:
            if (self.datetime_col not in df.columns
                    and self.datetime_col in self.original_data.columns):
                if len(df) != len(self.original_data):
                    raise ValueError(
                        f"Cannot infer '{self.datetime_col}' for transform input: "
                        f"got {len(df)} rows but the fitted data had "
                        f"{len(self.original_data)}. Pass the datetime column "
                        f"in X so rows can be aligned by timestamp."
                    )
                df[self.datetime_col] = self.original_data[self.datetime_col].values

            for col in self.groupby_cols:
                if col not in df.columns and col in self.original_data.columns:
                    if len(df) != len(self.original_data):
                        raise ValueError(
                            f"Cannot infer groupby column '{col}' for transform "
                            f"input: got {len(df)} rows but the fitted data had "
                            f"{len(self.original_data)}. Pass '{col}' in X."
                        )
                    df[col] = self.original_data[col].values

        # Ensure datetime column is datetime type
        if self.datetime_col in df.columns:
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])

        # Sort by datetime and groupby columns for proper time series order
        sort_cols = [col for col in self.groupby_cols if col in df.columns]
        if self.datetime_col in df.columns:
            sort_cols.append(self.datetime_col)
        
        if sort_cols:
            # 1. Store the original index to restore order later (Task Fix: Transform Order)
            if '_original_index' not in df.columns:
                 # Create a tracking index
                 # If dataframe has a meaningful index, preserve it
                 if isinstance(X, pd.DataFrame):
                      df['_original_index'] = X.index
                 else:
                      df['_original_index'] = np.arange(len(df))

            # 2. Temp index for sorting y (array)
            df['_sort_idx_temp'] = np.arange(len(df))
            
            df = df.sort_values(sort_cols)
            
            # Sort y if provided
            if y is not None:
                if len(y) == len(df):
                    sort_indices = df['_sort_idx_temp'].values
                    
                    if isinstance(y, (pd.Series, pd.DataFrame)):
                        # Use iloc for pandas objects
                        y = y.iloc[sort_indices].reset_index(drop=True)
                    else:
                        # Assume array-like, handle indexing
                        # Convert to numpy array if list for safety
                        y = np.array(y)[sort_indices]
                else:
                    if self.verbose: 
                        print("Warning: y length mismatch in prepare_time_series_data, skipping y sort")
            
            # Remove temp index and reset, BUT KEEP _original_index
            df = df.drop(columns=['_sort_idx_temp']).reset_index(drop=True)

        if y is not None:
            return df, y
        return df

    def _apply_time_based_operation(self, data, feature_col, operation, window_size=None, lag_period=None, time_step=None):
        """
        Apply time-based series operation to a specific feature column using vectorized operations - OPTIMIZED
        """
        try:
            if feature_col not in data.columns or self.datetime_col not in data.columns:
                return np.zeros(len(data))

            # Select window step if not provided
            current_step = time_step or self._select_window_step()

            # Determine groupby columns
            groups = [col for col in self.groupby_cols if col in data.columns]
            # Add block ID if present (Task 4) to prevent seams
            if '_block_id' in data.columns:
                groups.append('_block_id')

            # Prepare series to operate on
            # If we utilize groups, we use groupby
            if groups is not None and len(groups) > 0:
                grouped = data.groupby(groups, sort=False, group_keys=False)[feature_col]
                
                if operation == 'rolling_mean':
                    result = self._time_based_rolling(
                        data, feature_col, window_size, 'mean', groups)
                         
                elif operation == 'rolling_std':
                    result = self._time_based_rolling(
                        data, feature_col, window_size, 'std', groups)
                         
                elif operation == 'rolling_min':
                    result = self._time_based_rolling(
                        data, feature_col, window_size, 'min', groups)
                         
                elif operation == 'rolling_max':
                    result = self._time_based_rolling(
                        data, feature_col, window_size, 'max', groups)
                         
                elif operation == 'rolling_median':
                    result = self._time_based_rolling(
                        data, feature_col, window_size, 'median', groups)
                         
                elif operation == 'rolling_sum':
                    result = self._time_based_rolling(
                        data, feature_col, window_size, 'sum', groups)
                    
                elif operation == 'lag':
                    lag_period = self._resolve_lag_period(lag_period, data)
                    result = grouped.shift(lag_period).fillna(0).values
                    
                elif operation == 'diff':
                    lag_period = self._resolve_lag_period(lag_period, data)
                    result = grouped.diff(lag_period).fillna(0).values
                    
                elif operation == 'pct_change':
                    lag_period = self._resolve_lag_period(lag_period, data)
                    result = grouped.pct_change(lag_period).fillna(0).values
                    result = np.where(np.isinf(result), 0, result)
                    
                elif operation == 'momentum':
                    # Momentum is just diff
                    lag_period = self._resolve_lag_period(lag_period, data)
                    result = grouped.diff(lag_period).fillna(0).values
                    
                elif operation == 'seasonal_decompose':
                    # Apply global seasonal decomposition (ignoring blocks/groups for robustness)
                    if hasattr(data, self.datetime_col):
                         # Helper to use correct index
                         temp_series = data.set_index(self.datetime_col)[feature_col]
                         if len(temp_series) > 7:
                             res = temp_series.groupby(temp_series.index.dayofyear).transform('mean')
                             result = res.values
                         else:
                             result = np.full(len(data), temp_series.mean())
                    else:
                         result = np.zeros(len(data))

                elif operation == 'trend':
                    # Optimized Vectorized Trend (Slope of Linear Regression)
                     w_size = self._resolve_lag_period(window_size, data)
                     
                     # Check if we can use vectorized numpy operations
                     if use_vectorized_trend := True: # Enable by default
                         try:
                             # y = mx + c
                             # m = (N*sum(xy) - sum(x)*sum(y)) / (N*sum(x^2) - sum(x)^2)
                             # x is 0, 1, ..., N-1
                             
                             y_vals = data[feature_col].fillna(0).values
                             N = w_size
                             
                             # Constants for x = 0, 1, ..., N-1
                             x = np.arange(N)
                             sum_x = x.sum()
                             sum_x2 = (x ** 2).sum()
                             denominator = N * sum_x2 - sum_x ** 2
                             
                             if denominator == 0:
                                 result = np.zeros(len(data))
                             else:
                                 # Rolling sum of y
                                 # using pandas rolling is reasonably fast for sum
                                 sum_y = data[feature_col].rolling(window=N, min_periods=N).sum().fillna(0).values
                                 
                                 # Rolling sum of xy
                                 # We need convolution: sum(x[i] * y[t-(N-1)+i])
                                 # x is [0, 1, ..., N-1]
                                 # Convolution kernel should be reversed x: [N-1, ..., 1, 0]
                                 kernel = x[::-1]
                                 
                                 # Valid mode convolution returns array of length len(y) - N + 1
                                 # We need to pad result to align
                                 sum_xy_valid = np.convolve(y_vals, kernel, mode='valid')
                                 
                                 # Prepend zeros for the initial window period
                                 padding = np.zeros(N - 1)
                                 sum_xy = np.concatenate([padding, sum_xy_valid])
                                 
                                 # Calculate slope m
                                 top = N * sum_xy - sum_x * sum_y
                                 result = top / denominator
                                 
                                 # Apply proper masking later
                         except Exception as e:
                             if self.verbose: print(f"Vectorized trend failed: {e}, falling back.")
                             # Use rolling apply on groups (slow fallback)
                             def _trend_calc(x):
                                 if len(x) < 2: return 0
                                 return np.polyfit(np.arange(len(x)), x, 1)[0]
                             result = grouped.rolling(window=w_size, min_periods=2).apply(_trend_calc, raw=True).fillna(0).values
                     else:
                        # Legacy fallback
                        def _trend_calc(x):
                             if len(x) < 2: return 0
                             return np.polyfit(np.arange(len(x)), x, 1)[0]
                        result = grouped.rolling(window=w_size, min_periods=2).apply(_trend_calc, raw=True).fillna(0).values

                     # MASKING FOR TREND
                     # Since we did global convolution, we MUST mask boundaries
                     if groups:
                         # Calculate group changes
                         # Mark the first w_size-1 rows of each group as 0 (invalid trend)
                         # We can use the generic masking logic if we refactor, but for now apply here
                         group_mask = data[groups].ne(data[groups].shift()).any(axis=1)
                         
                         # Get indices where groups change
                         change_indices = np.where(group_mask)[0]
                         
                         # Also the very first index is a start
                         if 0 not in change_indices:
                             change_indices = np.insert(change_indices, 0, 0)
                             
                         for idx in change_indices:
                             end_idx = min(idx + w_size - 1, len(result))
                             result[idx:end_idx] = 0

                elif operation in ('weekday_mean', 'month_mean'):
                     # Expanding mean within each calendar group, over PRIOR
                     # observations only.
                     #
                     # This used to be a plain groupby(...).transform('mean'),
                     # which averages the whole column within each weekday /
                     # month. That is look-ahead leakage: a row's feature value
                     # was computed partly from rows dated after it, including
                     # -- at transform() time -- rows from the future relative
                     # to the point being predicted. It inflated apparent
                     # performance for exactly the seasonal signals these
                     # operators are meant to capture.
                     #
                     # shift(1) drops the current row so a value never depends
                     # on itself; expanding().mean() then averages only the
                     # earlier rows in the same calendar group. Early rows have
                     # no prior observation in their group and are left at 0.
                     if self.datetime_col in data.columns:
                         dt_series = data[self.datetime_col]
                         if operation == 'weekday_mean':
                             calendar_key = dt_series.dt.dayofweek
                         else:
                             calendar_key = dt_series.dt.month

                         # Respect entity/block boundaries when present, so a
                         # series never borrows history from another series.
                         group_key = [calendar_key] if not groups else list(groups) + [calendar_key]

                         means = data.groupby(group_key, sort=False)[feature_col].transform(
                             lambda s: s.shift(1).expanding().mean()
                         )
                         result = means.fillna(0).values
                     else:
                         result = np.zeros(len(data))

                elif operation == 'ewm':
                    span = self._resolve_window_span(window_size, current_step)
                    result = grouped.ewm(span=span, adjust=False).mean().values

                else:
                    # Fallback for very complex custom ops
                    return self._apply_time_based_operation_loop(data, feature_col, operation, window_size, lag_period, current_step)

            else:
                # No grouping - Global Time Series
                series = data[feature_col]
                
                if operation == 'rolling_mean':
                    result = self._time_based_rolling(data, feature_col, window_size, 'mean', groups)
                elif operation == 'rolling_std':
                    result = self._time_based_rolling(data, feature_col, window_size, 'std', groups)
                    result = np.nan_to_num(result)
                elif operation == 'rolling_min':
                    result = self._time_based_rolling(data, feature_col, window_size, 'min', groups)
                elif operation == 'rolling_max':
                    result = self._time_based_rolling(data, feature_col, window_size, 'max', groups)
                elif operation == 'rolling_median':
                    result = self._time_based_rolling(data, feature_col, window_size, 'median', groups)
                elif operation == 'rolling_sum':
                    result = self._time_based_rolling(data, feature_col, window_size, 'sum', groups)
                
                elif operation == 'lag':
                    lag_period = self._resolve_lag_period(lag_period, data)
                    result = series.shift(lag_period).fillna(0).values
                    
                elif operation == 'diff':
                    lag_period = self._resolve_lag_period(lag_period, data)
                    result = series.diff(lag_period).fillna(0).values

                elif operation == 'pct_change':
                    lag_period = self._resolve_lag_period(lag_period, data)
                    result = series.pct_change(lag_period).fillna(0).values
                    result = np.where(np.isinf(result), 0, result)
                    
                elif operation == 'momentum':
                    lag_period = self._resolve_lag_period(lag_period, data)
                    result = series.diff(lag_period).fillna(0).values

                elif operation == 'ewm':
                     span = self._resolve_window_span(window_size, current_step)
                     result = series.ewm(span=span, adjust=False).mean().values

                else:
                     # Fallback
                     return self._apply_time_based_operation_loop(data, feature_col, operation, window_size, lag_period, current_step)

            # Ensure numpy array
            if hasattr(result, 'values'):
                result = result.values
            return np.nan_to_num(result)

        except Exception as e:
            if self.verbose:
                print(f"    Warning: Time-based operation {operation} failed for {feature_col}: {str(e)}")
            return np.zeros(len(data))

    def _vectorized_rolling(self, grouped, window_size, func, data, groups):
        """Helper for vectorized rolling on groupby"""
        # Determine if we can use integer window
        use_int_window = isinstance(window_size, (int, np.integer))
        
        if use_int_window:
            # Integer window - preserves original index in MultiIndex (groups..., orig_idx)
            roller = grouped.rolling(window=window_size, min_periods=1)
            if func == 'mean': res = roller.mean()
            elif func == 'std': res = roller.std()
            elif func == 'min': res = roller.min()
            elif func == 'max': res = roller.max()
            elif func == 'median': res = roller.median()
            elif func == 'sum': res = roller.sum()
            
            # Align back to original index
            # Drop group levels (0 to n-1)
            # Result index: (g1, g2, ..., orig_idx)
            # Since data is pre-sorted by groups, the result values are already aligned with data.values
            return res.values
        else:
            # Time-based window - Requires 'on' parameter and merging
            # If using 'on', we need to pass the rolling object differently
            # grouped.rolling(..., on=...) not directly available on SeriesGroupBy object easily in strict vector way without setup
            
            # Hybrid approach: Use integer approximation if possible, or accept loop for time-based if irregular?
            # Or use apply which is safer but slower?
            # Let's try to map generic time window to integer if data is regular-ish
            # BUT user asked for '7D' specifically.
            
            # Fallback to loop for time-based windows on groups to ensure correctness if 'on' is complex
            # OR try the merge strategy
            
            # Let's use the loop LOGIC for time-based properties if we can't vectorize easily WITHOUT bugs.
            # But the requirement is to fix performance.
            # Iterative groupby is the bottleneck.
            
            # If we assume 'daily' step for data, convert '7D' to 7.
            # Most data in BigFeat context is likely treated as steps.
            # But let's check if we can convert.
            
            est_rows = self._estimate_window_rows(window_size, data)
            return self._vectorized_rolling(grouped, est_rows, func, data, groups)

    def _vectorized_rolling_global(self, data, feature_col, window_size, func):
        """Helper for global rolling"""
        series = data[feature_col]
        
        # Sort temporarily by time for global calculation if needed
        # Since we switched to Group-First sorting, the global data might not be time-sorted
        if self.datetime_col in data.columns:
             temp_data = data.sort_values(self.datetime_col)
             series_for_rolling = temp_data[feature_col]
             
             if isinstance(window_size, (int, np.integer)):
                 roller = series_for_rolling.rolling(window=window_size, min_periods=1)
             else:
                 roller = temp_data.rolling(window=window_size, on=self.datetime_col, min_periods=1)[feature_col]
        else:
             series_for_rolling = series
             if isinstance(window_size, (int, np.integer)):
                 roller = series.rolling(window=window_size, min_periods=1)
             else:
                 # Should not happen if datetime_col is missing but just in case
                 roller = data.rolling(window=window_size, on=self.datetime_col, min_periods=1)[feature_col]
        
        if func == 'mean': res = roller.mean()
        elif func == 'std': res = roller.std()
        elif func == 'min': res = roller.min()
        elif func == 'max': res = roller.max()
        elif func == 'median': res = roller.median()
        elif func == 'sum': res = roller.sum()
        
        if self.datetime_col in data.columns:
             # Align back to the original (Group-sorted) index
             return res.reindex(data.index).values
        else:
             return res.values

    def _resolve_lag_period(self, lag_period, data):
        """Resolve lag period to integer"""
        if lag_period is None: 
            return 1
        if isinstance(lag_period, (int, np.integer)):
            return lag_period
        return self._estimate_window_rows(lag_period, data)

    def _resolve_window_span(self, window_size, step):
        """Resolve window size to span for EWM"""
        if isinstance(window_size, (int, np.integer)):
            return window_size
        return max(1, window_size.days) # Simplified

    def _estimate_window_rows(self, period, data):
        """Estimate a row count for a time period, from the DATA's own spacing.

        This is a fallback for the few operations that still need an integer
        window (EWM spans, and lag periods where the caller did not supply a
        frequency). Time-based rolling no longer routes through here -- see
        _time_based_rolling.

        The previous implementation converted using self.time_step, which
        defaults to 'D', so a 90-day window became 90 ROWS regardless of how
        the data was actually sampled. On monthly data a 90-day window should
        span 3 rows; it spanned 90, i.e. the whole series. Measured against
        genuine time-based rolling on the Monash benchmark datasets, 100% of
        rows were wrong on every frequency tested.

        Now the spacing is measured from the data when possible.
        """
        if isinstance(period, (int, np.integer)):
            return int(period)

        if isinstance(period, str):
            period = pd.Timedelta(period)

        median_step = self._median_time_step(data)
        if median_step is not None and median_step > pd.Timedelta(0):
            return max(1, int(round(period / median_step)))

        # No usable timestamps: fall back to the configured nominal step.
        step_to_delta = {
            'D': pd.Timedelta(days=1), 'H': pd.Timedelta(hours=1),
            'h': pd.Timedelta(hours=1), 'W': pd.Timedelta(days=7),
            'M': pd.Timedelta(days=30), 'Q': pd.Timedelta(days=91),
            'Y': pd.Timedelta(days=365),
        }
        nominal = step_to_delta.get(self.time_step, pd.Timedelta(days=1))
        return max(1, int(round(period / nominal)))

    def _median_time_step(self, data):
        """Median spacing between consecutive timestamps, or None.

        Uses the median rather than the mean so that gaps between entities --
        or an occasional missing observation -- do not distort the estimate.
        """
        if self.datetime_col is None or not hasattr(data, 'columns'):
            return None
        if self.datetime_col not in data.columns:
            return None
        try:
            ts = pd.to_datetime(data[self.datetime_col])
            groups = [c for c in self.groupby_cols if c in data.columns]
            if groups:
                diffs = ts.groupby([data[c] for c in groups]).diff().dropna()
            else:
                diffs = ts.sort_values().diff().dropna()
            if len(diffs) == 0:
                return None
            step = diffs.median()
            return step if step > pd.Timedelta(0) else None
        except Exception:
            return None

    def _time_based_rolling(self, data, feature_col, window_size, func, groups):
        """Rolling aggregation over a genuine time window.

        Replaces the previous row-count approximation. A pd.Timedelta window
        is passed straight to pandas, which selects rows by timestamp, so the
        window means the same thing regardless of sampling frequency or of
        calendar units having unequal lengths (28-31 day months, 90-92 day
        quarters, 365-366 day years).

        This also fixes a second defect in the old grouped path: it rolled
        GLOBALLY across the whole frame and masked the first rows of each
        group afterwards, so a group's early rows averaged in the preceding
        group's values before being zeroed. Grouping happens before rolling
        here, so values never cross an entity or block boundary.
        """
        if isinstance(window_size, str):
            window_size = pd.Timedelta(window_size)

        # An integer window has no time semantics; honour it as a row count.
        if isinstance(window_size, (int, np.integer)):
            if groups:
                res = data.groupby(groups, sort=False)[feature_col].rolling(
                    window=int(window_size), min_periods=1)
                res = getattr(res, func)().reset_index(level=list(range(len(groups))),
                                                       drop=True)
                return res.reindex(data.index).fillna(0).values
            res = data[feature_col].rolling(window=int(window_size), min_periods=1)
            return getattr(res, func)().fillna(0).values

        # Time-based window: pandas requires a monotonic datetime index.
        ts = pd.to_datetime(data[self.datetime_col])
        frame = pd.DataFrame({feature_col: data[feature_col].values,
                              '_ts': ts.values}, index=data.index)
        for col in groups:
            frame[col] = data[col].values

        if groups:
            pieces = []
            for _, chunk in frame.groupby(groups, sort=False):
                chunk = chunk.sort_values('_ts')
                rolled = getattr(
                    chunk.set_index('_ts')[feature_col].rolling(window_size,
                                                                min_periods=1),
                    func)()
                pieces.append(pd.Series(rolled.values, index=chunk.index))
            out = pd.concat(pieces).reindex(data.index)
        else:
            ordered = frame.sort_values('_ts')
            rolled = getattr(
                ordered.set_index('_ts')[feature_col].rolling(window_size,
                                                              min_periods=1),
                func)()
            out = pd.Series(rolled.values, index=ordered.index).reindex(data.index)

        return out.fillna(0).values

    def _apply_time_based_operation_loop(self, data, feature_col, operation, window_size=None, lag_period=None, time_step=None):
        """
        Original iterative implementation as fallback
        """
        try:
            # Use provided time_step or select new one (though usually fallback is called with one)
            current_step = time_step or self._select_window_step()

            # Check if we have groupby columns
            if self.groupby_cols is not None and len(self.groupby_cols) > 0 and any(col in data.columns for col in self.groupby_cols):
                # Group data by groupby columns
                groupby_cols = [col for col in self.groupby_cols if col in data.columns]
                
                # Check for block ID (Task 4) - fallback should also respect it if possible, 
                # but legacy loop might not easily unless we add it to groupby cols
                if '_block_id' in data.columns:
                    groupby_cols.append('_block_id')

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
        except Exception:
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
                if window_size is None:
                    window_size = self.rng.choice(self.window_sizes)
                result = series.rolling(window=window_size, min_periods=1).mean()

            elif operation == 'rolling_std':
                if window_size is None:
                    window_size = self.rng.choice(self.window_sizes)
                result = series.rolling(window=window_size, min_periods=1).std().fillna(0)

            elif operation == 'rolling_min':
                if window_size is None:
                    window_size = self.rng.choice(self.window_sizes)
                result = series.rolling(window=window_size, min_periods=1).min()

            elif operation == 'rolling_max':
                if window_size is None:
                    window_size = self.rng.choice(self.window_sizes)
                result = series.rolling(window=window_size, min_periods=1).max()

            elif operation == 'rolling_median':
                if window_size is None:
                    window_size = self.rng.choice(self.window_sizes)
                result = series.rolling(window=window_size, min_periods=1).median()

            elif operation == 'rolling_sum':
                if window_size is None:
                    window_size = self.rng.choice(self.window_sizes)
                result = series.rolling(window=window_size, min_periods=1).sum()

            elif operation == 'lag':
                if lag_period is None:
                    lag_period = self.rng.choice(self.lag_periods)
                # FIX: Use shift with freq parameter for time-aware shifting
                result = series.shift(freq=lag_period)
                # Reindex to match original index and forward fill
                result = result.reindex(series.index, method='ffill').fillna(0)

            elif operation == 'diff':
                if lag_period is None:
                    lag_period = self.rng.choice(self.lag_periods)
                # FIX: Use shift with freq parameter
                lagged = series.shift(freq=lag_period)
                lagged = lagged.reindex(series.index, method='ffill').fillna(0)
                result = series - lagged
                result = result.fillna(0)

            elif operation == 'pct_change':
                if lag_period is None:
                    lag_period = self.rng.choice(self.lag_periods)
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
                if window_size is None:
                    window_size = self.rng.choice(self.window_sizes)
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
                elif current_time_step == 'Q':
                    span = window_size.days / 90
                elif current_time_step == 'Y':
                    span = window_size.days / 365
                else:
                    span = window_size.days

                # Ensure span is at least 1
                span = max(1, int(span))
                result = series.ewm(span=span, adjust=False).mean()

            elif operation == 'momentum':
                if lag_period is None:
                    lag_period = self.rng.choice(self.lag_periods)
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
                if window_size is None:
                    window_size = self.rng.choice(self.window_sizes)
                # Calculate trend as rolling linear regression slope
                result = series.rolling(window=window_size, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    raw=True
                ).fillna(0)

            elif operation in ('weekday_mean', 'month_mean'):
                # Expanding mean within the calendar group over PRIOR rows only.
                # A plain transform('mean') here averaged the whole group,
                # including rows dated after the row being computed -- see the
                # matching fix in _apply_time_based_operation.
                if operation == 'weekday_mean':
                    calendar_key = series.index.dayofweek
                else:
                    calendar_key = series.index.month
                result = series.groupby(calendar_key).transform(
                    lambda s: s.shift(1).expanding().mean()
                ).fillna(0)

            else:
                result = pd.Series(0, index=series.index)

            return result

        except Exception as e:
            # If operation fails, return zeros
            if self.verbose:
                print(f"    Warning: Operation {operation} failed: {str(e)}")
            return pd.Series(0, index=series.index)

    def _apply_group_mask(self, result, data, groups, w_size):
        """
        Apply mask to result array to prevent data leakage between groups in global rolling.
        Sets the first (w_size - 1) elements of each group to 0.
        """
        if groups is None or len(groups) == 0:
            return result
            
        try:
            # Calculate group boundaries
            # This identifies rows where the group key is different from the previous row
            # Since data is sorted by group, this finds the start of each new time series
            group_mask = data[groups].ne(data[groups].shift()).any(axis=1)
            
            # Get indices where groups change
            change_indices = np.where(group_mask)[0]
            
            # Ensure the very first index (0) is treated as a start
            if 0 not in change_indices:
                change_indices = np.insert(change_indices, 0, 0)
            
            # Mask the beginning of each group
            # For a rolling window of size W, the first W-1 points are invalid
            # because they would include data from the previous group
            mask_len = int(w_size) - 1
            if mask_len <= 0:
                return result
                
            # Efficient masking
            # If many groups, we can optimize further, but this loop is O(n_groups)
            if mask_len > 0:
                 # Create an array of indices to mask for all groups at once
                 offsets = np.arange(mask_len)
                 mask_indices = (change_indices[:, None] + offsets).flatten()
                 # Filter indices within bounds
                 mask_indices = mask_indices[mask_indices < len(result)]
                 result[mask_indices] = 0
                
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Group masking failed: {e}")
            return result

    # Time Series Utility Methods
    def _clean_feature(self, feature_data):
        try:
            feature_data = np.asarray(feature_data, dtype=float)
            # Use a safe maximum for float32 (approx 1e38)
            # We use a more conservative 1e30 to prevent overflow in subsequent multiplications
            safe_max = 1e30 
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=safe_max, neginf=-safe_max)
            return np.clip(feature_data, -safe_max, safe_max)
        except Exception:
            return np.zeros_like(feature_data)

    def _validate_feature(self, feature_data):
        """Validate features for stability and usefulness"""
        try:
            if len(feature_data) == 0:
                return False
            feature_data = np.asarray(feature_data, dtype=float)
            if not bool(np.isfinite(feature_data).all()):
                return False
            if float(np.std(feature_data)) < 1e-10:
                return False
            if float(np.max(np.abs(feature_data))) > 1e8:
                return False
            return True
        except Exception:
            return False

    # Safe Time Series Operations that use time-based operations
    def _safe_rolling_mean(self, feature_data, window_size=None, time_step=None, context_data=None, **kwargs):
        """Safe rolling mean calculation using time-based operations"""
        # Task 5: Use explicit context_data if provided, else fallback to global state
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'rolling_mean', window_size=window_size, time_step=time_step)
        else:
            # Fallback to original implementation
            try:
                if window_size is None:
                    window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                if isinstance(window_size, pd.Timedelta):
                    # Estimate integer window if timedelta passed to fallback
                    window_size = max(1, window_size.days)
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).mean().bfill().values
                return self._clean_feature(result)
            except (ValueError, TypeError) as e:
                # Task 6: Specific catch and logging
                if self.verbose: print(f"Error in rolling_mean: {e}")
                return feature_data
            except Exception as e:
                if self.verbose: print(f"Unexpected error in rolling_mean: {e}")
                return feature_data

    def _safe_rolling_std(self, feature_data, window_size=None, time_step=None, context_data=None, **kwargs):
        """Safe rolling standard deviation using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'rolling_std', window_size=window_size, time_step=time_step)
        else:
            try:
                if window_size is None:
                    window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                if isinstance(window_size, pd.Timedelta):
                    window_size = max(1, window_size.days)
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).std().fillna(0).values
                return self._clean_feature(result)
            except Exception as e:
                if self.verbose: print(f"Error in rolling_std: {e}")
                return np.zeros_like(feature_data)

    def _safe_rolling_min(self, feature_data, window_size=None, time_step=None, context_data=None, **kwargs):
        """Safe rolling minimum using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'rolling_min', window_size=window_size, time_step=time_step)
        else:
            try:
                if window_size is None:
                    window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                if isinstance(window_size, pd.Timedelta):
                    window_size = max(1, window_size.days)
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).min().bfill().values
                return self._clean_feature(result)
            except Exception as e:
                if self.verbose: print(f"Error in rolling_min: {e}")
                return feature_data

    def _safe_rolling_max(self, feature_data, window_size=None, time_step=None, context_data=None, **kwargs):
        """Safe rolling maximum using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'rolling_max', window_size=window_size, time_step=time_step)
        else:
            try:
                if window_size is None:
                    window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                if isinstance(window_size, pd.Timedelta):
                    window_size = max(1, window_size.days)
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).max().bfill().values
                return self._clean_feature(result)
            except Exception as e:
                if self.verbose: print(f"Error in rolling_max: {e}")
                return feature_data

    def _safe_rolling_median(self, feature_data, window_size=None, time_step=None, context_data=None, **kwargs):
        """Safe rolling median using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'rolling_median', window_size=window_size, time_step=time_step)
        else:
            try:
                if window_size is None:
                    window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                if isinstance(window_size, pd.Timedelta):
                    window_size = max(1, window_size.days)
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).median().bfill().values
                return self._clean_feature(result)
            except Exception as e:
                if self.verbose: print(f"Error in rolling_median: {e}")
                return feature_data

    def _safe_rolling_sum(self, feature_data, window_size=None, time_step=None, context_data=None, **kwargs):
        """Safe rolling sum using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'rolling_sum', window_size=window_size, time_step=time_step)
        else:
            try:
                if window_size is None:
                    window_size = self.rng.choice([3, 5, 7, 10, 14, 21, 30])
                if isinstance(window_size, pd.Timedelta):
                    window_size = max(1, window_size.days)
                window_size = min(window_size, len(feature_data))
                result = pd.Series(feature_data).rolling(window=window_size, min_periods=1).sum().fillna(0).values
                return self._clean_feature(result)
            except Exception as e:
                if self.verbose: print(f"Error in rolling_sum: {e}")
                return np.zeros_like(feature_data)

    def _safe_lag_feature(self, feature_data, lag_period=None, context_data=None, **kwargs):
        """Safe lag feature creation using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'lag', lag_period=lag_period)
        else:
            try:
                lag_periods = lag_period
                if lag_periods is None:
                    lag_periods = self.rng.choice([1, 2, 3, 5, 7, 10])
                if isinstance(lag_periods, pd.Timedelta):
                    lag_periods = max(1, lag_periods.days)
                lag_periods = min(lag_periods, len(feature_data) - 1)
                result = pd.Series(feature_data).shift(lag_periods).bfill().values
                return self._clean_feature(result)
            except Exception as e:
                if self.verbose: print(f"Error in lag_feature: {e}")
                return feature_data

    def _safe_diff_feature(self, feature_data, lag_period=None, context_data=None, **kwargs):
        """Safe difference calculation using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'diff', lag_period=lag_period)
        else:
            try:
                periods = lag_period
                if periods is None:
                    periods = self.rng.choice([1, 2, 3, 5, 7, 10])
                if isinstance(periods, pd.Timedelta):
                    periods = max(1, periods.days)
                periods = min(periods, len(feature_data) - 1)
                result = pd.Series(feature_data).diff(periods).fillna(0).values
                return self._clean_feature(result)
            except Exception as e:
                if self.verbose: print(f"Error in diff_feature: {e}")
                return np.zeros_like(feature_data)

    def _safe_pct_change(self, feature_data, lag_period=None, context_data=None, **kwargs):
        """Safe percentage change using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'pct_change', lag_period=lag_period)
        else:
            try:
                periods = lag_period
                if periods is None:
                    periods = self.rng.choice([1, 2, 3, 5, 7, 10])
                if isinstance(periods, pd.Timedelta):
                    periods = max(1, periods.days)
                periods = min(periods, len(feature_data) - 1)
                result = pd.Series(feature_data).pct_change(periods).fillna(0).values
                result = np.where(np.isinf(result), 0, result)
                return self._clean_feature(result)
            except Exception as e:
                if self.verbose: print(f"Error in pct_change: {e}")
                return np.zeros_like(feature_data)

    def _safe_ewm(self, feature_data, window_size=None, time_step=None, context_data=None, **kwargs):
        """Safe exponential moving average using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'ewm', window_size=window_size, time_step=time_step)
        else:
            try:
                # Approximate alpha from window_size if passed, otherwise random
                if window_size is not None:
                    if isinstance(window_size, pd.Timedelta):
                        span = max(1, window_size.days)
                    else:
                        span = max(1, window_size)
                    alpha = 2 / (span + 1)
                else:
                    alpha = self.rng.choice([0.1, 0.2, 0.3, 0.5])
                result = pd.Series(feature_data).ewm(alpha=alpha, adjust=False).mean().values
                return self._clean_feature(result)
            except Exception as e:
                if self.verbose: print(f"Error in ewm: {e}")
                return feature_data

    def _safe_momentum(self, feature_data, lag_period=None, context_data=None, **kwargs):
        """Safe momentum calculation using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'momentum', lag_period=lag_period)
        else:
            try:
                periods = lag_period
                if periods is None:
                    periods = self.rng.choice([1, 2, 3, 5, 7, 10])
                if isinstance(periods, pd.Timedelta):
                    periods = max(1, periods.days)
                periods = min(periods, len(feature_data) - 1)
                series = pd.Series(feature_data)
                momentum = series - series.shift(periods)
                result = momentum.fillna(0).values
                return self._clean_feature(result)
            except Exception as e:
                if self.verbose: print(f"Error in momentum: {e}")
                return np.zeros_like(feature_data)

    def _safe_seasonal_decompose(self, feature_data, context_data=None, **kwargs):
        """Safe seasonal decomposition using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'seasonal_decompose')
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
            except Exception as e:
                if self.verbose: print(f"Error in seasonal_decompose: {e}")
                return feature_data

    def _safe_trend_feature(self, feature_data, window_size=None, time_step=None, context_data=None, **kwargs):
        """Safe trend feature using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'trend', window_size=window_size, time_step=time_step)
        else:
            try:
                if window_size is None:
                    window_size = self.rng.choice([7, 14, 30, 60])
                if isinstance(window_size, pd.Timedelta):
                    window_size = max(1, window_size.days)
                window_size = min(window_size, len(feature_data))
                series = pd.Series(feature_data)
                # Simple trend as rolling linear regression slope
                result = series.rolling(window=window_size, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
                )
                return self._clean_feature(result.fillna(0).values)
            except Exception as e:
                if self.verbose: print(f"Error in trend_feature: {e}")
                return np.zeros_like(feature_data)

    def _safe_weekday_mean(self, feature_data, context_data=None, **kwargs):
        """Safe weekday mean using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'weekday_mean')
        else:
            # Fallback: create simple cyclical feature
            try:
                result = np.sin(2 * np.pi * np.arange(len(feature_data)) / 7)
                return self._clean_feature(result * np.std(feature_data) + np.mean(feature_data))
            except Exception as e:
                if self.verbose: print(f"Error in weekday_mean: {e}")
                return feature_data

    def _safe_month_mean(self, feature_data, context_data=None, **kwargs):
        """Safe month mean using time-based operations"""
        data_source = context_data if context_data is not None else getattr(self, '_current_data', None)
        
        if self.enable_time_series and data_source is not None:
            feature_col = self._resolve_feature_col(kwargs.get('feature_index'))
            if feature_col is None:
                return self._clean_feature(feature_data)
            return self._apply_time_based_operation(data_source, feature_col, 'month_mean')
        else:
            # Fallback: create simple cyclical feature
            try:
                result = np.sin(2 * np.pi * np.arange(len(feature_data)) / 30)
                return self._clean_feature(result * np.std(feature_data) + np.mean(feature_data))
            except Exception as e:
                if self.verbose: print(f"Error in month_mean: {e}")
                return feature_data

    def _resolve_feature_col(self, feature_index=None):
        """Map a feature index to its column name for time-based operations.

        Pass feature_index explicitly wherever possible. The fallback to
        self._current_feature_index exists only for the non-time-series
        fallback paths and for backwards compatibility with recipes stored
        before the index was recorded.

        The shared-state version was ambiguous inside binary expression nodes:
        the index is assigned as each leaf is resolved, but a time-series
        operator reads it later, when the parent node fires. With two
        different leaves the second one's index had already overwritten the
        first, so both branches operated on the same -- often wrong -- column,
        and which column that was depended on evaluation order rather than on
        the recipe.
        """
        if feature_index is None:
            feature_index = getattr(self, '_current_feature_index', None)
        if feature_index is None:
            return None

        feature_index = int(feature_index)
        if self.feature_columns is not None and len(self.feature_columns) > 0:
            if 0 <= feature_index < len(self.feature_columns):
                return self.feature_columns[feature_index]
        return f'feature_{feature_index}'

    def _reset_fit_state(self):
        """Clear state carried over from any previous fit() on this object.

        Several attributes were accumulated in place and never cleared, so
        calling fit() twice on one estimator did not give the same result as
        fitting a fresh one:

        * self.operators / self.unary_operators had the time-series operators
          appended (and, in restricted mode, entries filtered out) directly on
          the instance lists, guarded by a _ts_operators_added flag that was
          never reset. A second fit on data with a different periodicity
          verdict kept the first fit's operator pool.
        * _effective_ts_weight_multiplier persisted the pre-flight penalty.
        * time_series_operators / _trend_mode_active persisted the previous
          run's restricted-mode decision.

        Rebuild the operator pool from the base definitions each time.
        """
        self.operators = [np.multiply, np.add, np.subtract, np.abs, np.square]
        self.unary_operators = [np.abs, np.square, local_utils.original_feat]

        for attr in ('_ts_operators_added', 'time_series_operators',
                     '_trend_mode_active', '_effective_ts_weight_multiplier',
                     'avg_consensus_confidence', 'avg_lag1'):
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def effective_ts_weight_multiplier(self):
        """The TS operator weight multiplier in force for the current fit.

        Normally this is the constructor argument. The pre-flight check may
        halve it for a single fit when time-series features show no measurable
        benefit; that per-fit penalty is stored separately so it cannot
        compound across successive fit() calls or silently overwrite what the
        caller configured.
        """
        return getattr(self, '_effective_ts_weight_multiplier',
                       self.ts_operation_weight_multiplier)

    @staticmethod
    def _normalize_to_distribution(weights):
        """Scale a non-negative weight vector so it sums to 1.

        Used for the sampling distributions that drive feature selection
        (ig_vector, split_vec). These were previously normalized with a bare
        `v /= v.sum()`, which produces NaN whenever the weights sum to zero --
        and they legitimately do for degenerate input. A frame of constant or
        all-zero columns gives every feature zero importance, so fit() died
        with "ValueError: probabilities contain NaN" from rng.choice, rather
        than reporting anything useful about the data.

        Falls back to a uniform distribution when there is no signal to
        preserve, which lets generation proceed instead of crashing.
        """
        weights = np.asarray(weights, dtype=float)
        # Guard against NaN/inf arriving from an upstream estimator.
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        # Negative importances are not meaningful as sampling probabilities.
        weights = np.clip(weights, 0.0, None)

        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            return np.ones_like(weights) / max(len(weights), 1)
        return weights / total

    def _calculate_block_params(self, total_limit, max_window_size=None, padding=0):
        """
        Calculate optimal block sampling parameters (detector-agnostic).

        Parameters:
        -----------
        total_limit : int
            Maximum total samples allowed (from memory calculation)
        max_window_size : int, optional
            Maximum window size in days. If None, queries active detector.
        padding : int, default=0
            Padding/warm-up size to include in memory calculations.

        Returns:
        --------
        tuple of (int, int)
            (n_blocks, block_size) - block_size is the CORE size (excluding padding)
        """
        # 1. Dynamic parameter resolution
        if max_window_size is None:
            if hasattr(self, 'window_detector') and hasattr(self.window_detector, 'max_window_days'):
                max_window_size = self.window_detector.max_window_days
                if self.verbose:
                    print(f"  Querying {self.window_detector_type} detector: max_window={max_window_size} days")
            elif hasattr(self, 'max_window_days'):
                max_window_size = self.max_window_days
                if self.verbose:
                    print(f"  Using configured max_window: {max_window_size} days")
            else:
                max_window_size = 365
                if self.verbose:
                    print(f"  Using default max_window: {max_window_size} days")

        # 2. Safety Calculation: Block must be 3x window size
        min_core_size = max_window_size * 3
        # Total cost per block includes padding
        min_total_size = min_core_size + padding

        if self.verbose:
            print(f"  Minimum safe block size: {min_core_size} + {padding} padding = {min_total_size} rows")

        # 3. Emergency handling
        if total_limit < min_total_size:
            # If we can't fit even one safe block + padding, try to squeeze core size
            # We must prioritize having at least ONE block with some history
            # Reduced core size:
            min_core_size = int(max_window_size * 1.1)
            min_total_size = min_core_size + padding
            
            if self.verbose:
                print(f"  ⚠️  Memory tight! Reducing core to: {min_core_size}")

            if total_limit < min_total_size:
                # Critical: Can't even fit 1.1x window + padding?
                # Just take whatever fits, even if it breaks windows
                min_core_size = max(1, total_limit - padding)
                if min_core_size < 1: min_core_size = 1 # Edge case
                min_total_size = min_core_size + padding
                
                if self.verbose:
                    print(f"  ⚠️  CRITICAL: Using minimum core: {min_core_size}")

        # 4. Diversity Optimization
        ideal_n_blocks = 10
        # Calculate max rows available per block if we split equally
        max_rows_per_block = total_limit // ideal_n_blocks
        
        if max_rows_per_block >= min_total_size:
            n_blocks = ideal_n_blocks
            # Block size returned is CORE size
            block_size = max_rows_per_block - padding
            if self.verbose:
                print(f"  ✓ Optimal: {n_blocks} blocks × ({block_size} core + {padding} pad) rows")
        else:
            # Maximizing number of valid blocks
            n_blocks = max(1, total_limit // min_total_size)
            block_size = min_core_size
            if self.verbose:
                print(f"  ⚠️  Constrained: {n_blocks} blocks × ({block_size} core + {padding} pad) rows")

        return n_blocks, block_size

    def fit(self, X, y, gen_size=5, random_state=0, iterations=5, estimator='avg',
            feat_imps=True, split_feats=None, check_corr=True, selection='stability', combine_res=True,
            n_features=None, max_depth=None):
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
        
        self._reset_fit_state()

        if self.verbose:
            print("\n" + "=" * 60)
            print("BigFeat Feature Generation Started")
            print("=" * 60)

        # Store original data if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            self.original_data = X.copy()
        else:
            self.original_data = X

        # --- Automatic Target Scaling (Log-Transform) ---
        # For strictly positive and highly skewed regression targets
        self._target_log_transformed = False
        if self.task_type == 'regression':
            try:
                # Convert y to series for easy checks
                y_series = pd.Series(np.ravel(y))
                if bool((y_series > 0).all()):
                     skewness = y_series.skew()
                     if skewness > 2.0:  # Highly skewed
                         if self.verbose: 
                             print(f"  → High skewness detected (skew={skewness:.2f}). Applying log-transform to target.")
                         
                         y = np.log1p(y)
                         self._target_log_transformed = True
            except Exception as e:
                if self.verbose: 
                    print(f"Warning: Target scaling check failed: {e}")

        # === Setup time series based on mode ===
        ts_enabled = self._setup_time_series(X, y)

        # Identify feature columns before the pre-flight check below, which
        # needs them. They used to be assigned only after that block, so
        # self.feature_columns was still None when the check ran, the check
        # raised TypeError immediately, and the bare `except` swallowed it --
        # meaning this "protection against hallucinated seasonality" never
        # actually executed on any fit.
        if isinstance(X, pd.DataFrame):
            self.feature_columns = self._identify_feature_columns(X)
            if len(self.feature_columns) == 0:
                raise ValueError("No numeric feature columns found after filtering!")
        elif hasattr(X, 'columns'):
            self.feature_columns = list(X.columns)
        else:
            self.feature_columns = [
                f'feature_{i}' for i in range(np.asarray(X).shape[1])
            ]

        # Detection Cross-Validation (Pre-Flight Check)
        if ts_enabled and self.enable_time_series:
            try:
                # Perform a quick check to see if derived TS features actually effective
                # This protects against "hallucinated" seasonality in noisy data
                if self.verbose: print("  Running Pre-flight Detection Cross-Validation...")
                
                # 1. Prepare small split (last 20% or max 1000 rows to save time)
                # Validation on END of series is better for TS
                n_cv = min(len(X), 1000)
                split_idx = int(n_cv * 0.8)
                
                # Use the sorted internal data if available, else X
                # (Note: _setup_time_series might set self._current_data but _prepare_time_series_data is called LATER in fit)
                # So we use X directly, assuming we can get a column.
                
                if isinstance(X, pd.DataFrame) and self.datetime_col in X.columns:
                     # Sort temporarily for this check
                     df_cv = X.sort_values(self.datetime_col).tail(n_cv)
                     target_cv = y[-n_cv:] if len(y) >= n_cv else y
                     
                     # Split train/test
                     train_cv = df_cv.iloc[:split_idx]
                     test_cv = df_cv.iloc[split_idx:]
                     y_train_cv = target_cv[:split_idx]
                     y_test_cv = target_cv[split_idx:]
                     
                     if len(train_cv) > 20:
                         # 2. Baseline Model (Raw Features only)
                         # Clean raw features
                         raw_cols = [c for c in self.feature_columns if c in df_cv.columns]
                         X_base_train = train_cv[raw_cols].fillna(0).values
                         X_base_test = test_cv[raw_cols].fillna(0).values
                         
                         model_base = LinearRegression()
                         model_base.fit(X_base_train, y_train_cv)
                         preds_base = model_base.predict(X_base_test)
                         mae_base = np.mean(np.abs(y_test_cv - preds_base))
                         
                         # 3. TS Enhanced Model (Raw + Top Pooled Features)
                         # Generate 2-3 key features: Lag-1, Rolling Mean (best window), Rolling Std
                         X_ts_train = X_base_train.copy()
                         X_ts_test = X_base_test.copy()
                         
                         # Best window
                         best_win = self.window_sizes[0] if self.window_sizes else 3
                         # Feature to use (use first one)
                         target_feat = raw_cols[0] if raw_cols else None
                         
                         if target_feat:
                             # Compute the rolling features ONCE over the full
                             # ordered series, then slice train/test out of the
                             # result.
                             #
                             # Previously each side was rolled independently.
                             # That restarted the window at the start of the
                             # test slice, so test rows saw no real history and
                             # their features did not match what transform()
                             # would produce -- the code carried a comment
                             # acknowledging this ("Rolling on test is leaky")
                             # and did it anyway. Rolling over the concatenated
                             # series is both correct and causal: pandas'
                             # rolling only ever looks backwards, so test rows
                             # draw on training history without any test row
                             # influencing an earlier one.
                             if isinstance(best_win, pd.Timedelta):
                                 win_rows = max(1, int(best_win.days))
                             else:
                                 win_rows = max(1, int(best_win))

                             full_series = df_cv[target_feat]
                             roll_mean = full_series.rolling(
                                 window=win_rows, min_periods=1).mean().fillna(0).values
                             roll_std = full_series.rolling(
                                 window=win_rows, min_periods=1).std().fillna(0).values

                             X_ts_train = np.hstack([
                                 X_ts_train,
                                 roll_mean[:split_idx].reshape(-1, 1),
                                 roll_std[:split_idx].reshape(-1, 1),
                             ])
                             X_ts_test = np.hstack([
                                 X_ts_test,
                                 roll_mean[split_idx:].reshape(-1, 1),
                                 roll_std[split_idx:].reshape(-1, 1),
                             ])
                             
                             model_ts = LinearRegression()
                             model_ts.fit(X_ts_train, y_train_cv)
                             preds_ts = model_ts.predict(X_ts_test)
                             mae_ts = np.mean(np.abs(y_test_cv - preds_ts))
                             
                             # 4. Compare
                             if mae_base > 0:
                                 improvement = (mae_base - mae_ts) / mae_base
                             else:
                                 improvement = 0
                                 
                             if self.verbose:
                                 print(f"  CV Result: Base MAE={mae_base:.4f}, TS MAE={mae_ts:.4f} (Imp: {improvement*100:.1f}%)")
                                 
                             # 5. Penalize if no improvement
                             if improvement < 0.05: # Less than 5% improvement
                                 if self.verbose: print("  ⚠ Weak TS improvement. Reducing operator weights.")
                                 # Halve for THIS fit only. This used to
                                 # overwrite the constructor argument, so the
                                 # penalty compounded across successive fit()
                                 # calls on the same estimator: two fits on
                                 # weak data left the multiplier at 0.25 of
                                 # what the caller asked for, with no way to
                                 # recover it short of rebuilding the object.
                                 self._effective_ts_weight_multiplier = (
                                     self.ts_operation_weight_multiplier * 0.5
                                 )

                                 # Re-initialize weights to apply reduction
                                 self._initialize_operator_weights()
            except Exception as e:
                if self.verbose: print(f"  CV Check skipped: {e}")

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
            # Sort X and y by datetime to ensure alignment (Task 4/6)
            self._current_data, y_sorted = self._prepare_time_series_data(X, y)
            
            # Update y to match sorted X order
            if y_sorted is not None:
                y = y_sorted
            
            # Update X_features to match sorted order
            X_features = self._current_data[self.feature_columns].values
            X_features = np.array(X_features, dtype=float)
            
            if self.verbose:
                print(f"✓ Time series data prepared and sorted with '{self.datetime_col}'")

        # === DYNAMIC MEMORY-AWARE DOWNSAMPLING LOGIC ===
        # Intelligently adapts sample size based on available system memory
        if self.enable_downsampling:
            # 1. Get available system memory
            mem = psutil.virtual_memory()
            available_ram_bytes = mem.available

            # 2. Estimate memory consumption
            n_cols = X_features.shape[1]
            estimated_features_generated = gen_size * iterations
            bytes_per_row = (n_cols + estimated_features_generated) * 8 * 4

            # 3. Calculate safe row limit
            safe_ram_limit = available_ram_bytes * 0.5
            dynamic_max_samples = int(safe_ram_limit / bytes_per_row)

            # 4. Determine final limit
            if self.max_fit_samples > 0:
                limit = min(self.max_fit_samples, dynamic_max_samples)
                limit_source = "user-defined" if limit == self.max_fit_samples else "memory-based"
            else:
                limit = dynamic_max_samples
                limit_source = "fully dynamic"

            # 5. Apply BLOCK SAMPLING if needed
            if len(X_features) > limit:
                if self.verbose:
                    print(f"\n{'=' * 60}")
                    print("CONTIGUOUS BLOCK SAMPLING ENABLED")
                    print(f"{'=' * 60}")
                    print(f"Available RAM: {available_ram_bytes / 1e9:.2f} GB")
                    print(f"Dataset size: {len(X_features):,} rows → {limit:,} rows")

                # === Calculate block parameters ===
                # Determine padding size (max window needed for rolling ops)
                padding_size = 0
                if self.enable_time_series:
                    if hasattr(self, 'window_detector') and hasattr(self.window_detector, 'max_window_days'):
                         padding_size = self.window_detector.max_window_days
                    elif hasattr(self, 'max_window_days'):
                         padding_size = self.max_window_days
                    else:
                         padding_size = 365
                    
                    if self.verbose:
                        print(f"  Padding size for rolling history: {padding_size}")

                n_blocks, block_size = self._calculate_block_params(limit, padding=padding_size)

                if self.verbose:
                    print(f"\n📦 Block Sampling Strategy:")
                    print(f"  Blocks: {n_blocks} × {block_size} rows")
                    print(f"  Total: {n_blocks * block_size:,} rows")
                    print(f"  Seed: {self.downsampling_random_state}")

                # Create RNG
                downsample_rng = np.random.RandomState(seed=self.downsampling_random_state)

                # Calculate valid start positions
                max_start = len(X_features) - block_size

                if max_start < 0:
                    # Dataset smaller than block - use all
                    sample_indices = np.arange(len(X_features))
                else:
                    # Sample block starting positions
                    n_blocks_actual = min(n_blocks, max_start + 1)
                    start_indices = downsample_rng.choice(
                        max_start + 1,
                        n_blocks_actual,
                        replace=False
                    )

                    # Generate contiguous blocks
                    sample_indices = []
                    block_ids_list = []
                    for block_idx, start in enumerate(sorted(start_indices)):
                        # Ensure we don't go out of bounds
                        # Task Fix: Add padding for "warm-up" history
                        actual_start = max(0, start - padding_size)
                        end = min(start + block_size, len(X_features))
                        
                        indices = range(actual_start, end)
                        sample_indices.extend(indices)
                        block_ids_list.extend([block_idx] * len(indices))

                    sample_indices = np.array(sample_indices)
                    block_ids_array = np.array(block_ids_list)

                    if self.verbose:
                        print(f"\n  ✓ Sampled {n_blocks_actual} contiguous blocks:")
                        for i, start in enumerate(sorted(start_indices)[:5]):  # Show first 5
                            print(f"    Block {i + 1}: [{start:,} : {start + block_size - 1:,}]")
                        if n_blocks_actual > 5:
                            print(f"    ... and {n_blocks_actual - 5} more blocks")

                # Apply sampling
                X_features_sampled = X_features[sample_indices]
                y_sampled = y[sample_indices] if y is not None else None

                # Sync time series data
                if self.enable_time_series and hasattr(self, '_current_data') and self._current_data is not None:
                    if self.verbose:
                        print(f"\n  ↻ Syncing time series data store...")
                    self._full_current_data_backup = self._current_data

                    if isinstance(self._current_data, pd.DataFrame):
                        self._current_data = self._current_data.iloc[sample_indices].reset_index(drop=True)
                        # Add block ID to prevent seams (Task 4)
                        self._current_data['_block_id'] = block_ids_array
                    else:
                        # If simple array, convert to DF to support block IDs?
                        # Probably unlikely to hit this execution path if enabling time series, as prepare_ts returns DF.
                        self._current_data = self._current_data[sample_indices]

                # Store metadata
                self.n_rows_original = len(X_features)
                self.n_rows_fit = len(X_features_sampled)
                self._was_downsampled = True
                self._downsampling_method = 'contiguous_blocks'
                # Store original full data for final transform
                self._X_features_full = X_features
                # Defensive copy for original data
                if isinstance(self.original_data, pd.DataFrame):
                    self._original_data_full = self.original_data.copy()
                else:
                    self._original_data_full = self.original_data

                if self.verbose:
                    print(f"\n{'=' * 60}")
                    reduction_pct = (1 - self.n_rows_fit / self.n_rows_original) * 100
                    print(
                        f"✓ SAMPLING COMPLETE: {self.n_rows_original:,} → {self.n_rows_fit:,} ({reduction_pct:.1f}% reduction)")
                    print(f"\n✓ Phase Separation Strategy:")
                    print(f"  1. LEARN features on {self.n_rows_fit:,} sampled rows (memory-safe)")
                    print(f"  2. APPLY features to all {self.n_rows_original:,} rows (full coverage)")
                    print(f"  → Output will match original dimensions")
                    print(f"{'=' * 60}\n")

                X_for_fit = X_features_sampled
                y_for_fit = y_sampled

            else:
                # No downsampling needed
                X_for_fit = X_features
                y_for_fit = y
                self._was_downsampled = False
        else:
            # Downsampling disabled
            X_for_fit = X_features
            y_for_fit = y
            self._was_downsampled = False

        # === Original BigFeat initialization ===
        self.selection = selection
        self.imp_operators = np.ones(len(self.operators))
        self.operator_weights = self.imp_operators / self.imp_operators.sum()
        self.gen_steps = []
        if n_features is not None:
             self.n_feats = n_features
        else:
             self.n_feats = X_for_fit.shape[1]
        self.n_rows = X_for_fit.shape[0]
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
        if max_depth is not None:
            self.depth_range = np.arange(max_depth) + 1
        else:
            self.depth_range = np.arange(3) + 1
        self.depth_weights = 1 / (2 ** self.depth_range)
        self.depth_weights /= self.depth_weights.sum()

        # Scaling
        self.scaler = RobustScaler()
        self.scaler.fit(X_for_fit)
        X_scaled = self.scaler.transform(X_for_fit)

        # Feature importance calculation
        if feat_imps:
            if self.verbose:
                print(f"\nComputing feature importances using '{estimator}' estimator...")

            self.ig_vector, estimators = self.get_feature_importances(
                X_scaled, y_for_fit, estimator, random_state
            )
            self.ig_vector = self._normalize_to_distribution(self.ig_vector)

            for tree in estimators:
                paths = self.get_paths(tree, np.arange(X_scaled.shape[1]))
                self.get_split_feats(paths, self.split_vec)
            self.split_vec = self._normalize_to_distribution(self.split_vec)

            if split_feats == "comb":
                self.ig_vector = np.multiply(self.ig_vector, self.split_vec)
                self.ig_vector = self._normalize_to_distribution(self.ig_vector)
            elif split_feats == "splits":
                self.ig_vector = self.split_vec

        # === Feature Generation Iterations ===
        previous_imps = None  # Track importances for Elitism

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
            start_gen_idx = 0

            # --- Elitism Injection ---
            if iteration > 0 and previous_imps is not None:
                # Always preserve the top 20% of features from the previous iteration
                n_elite = max(1, self.n_feats // 5)
                # Use argsort on previous_imps (which corresponds to the successful n_feats from last round)
                # Note: previous_imps stores importances of the n_feats SELECTED in the last round.
                # However, we need to fetch the actual feature DATA.
                # The 'iters_comb' stores ALL history. The last round's successful feats are in iters_comb corresponding to (iteration-1).
                
                # Check if we have valid history in iters_comb
                # The features selected in (iteration-1) are stored in columns:
                # [(iteration-1)*self.n_feats : iteration*self.n_feats]
                # BUT wait. 'iters_comb' stores the SELECTED features from each iteration.
                # So we can just take the top N from the previous batch in iters_comb.
                
                # Identify indices of top performing features relative to the previous batch
                elite_local_indices = np.argsort(previous_imps)[-n_elite:]
                
                # Calculate the global column indices in iters_comb for these elite features
                prev_batch_start = (iteration - 1) * self.n_feats
                elite_global_indices = prev_batch_start + elite_local_indices
                
                if self.verbose: 
                    print(f"  ★ Elitism: Injecting top {n_elite} features from previous generation")

                # Inject into current generation pool
                # We place them at the BEGINNING of gen_feats
                gen_feats[:, :n_elite] = iters_comb[:, elite_global_indices]
                
                # Also carry over their metadata
                # Note: tracking_ops is a list, tracking_ids is a list/array
                # We need to append them to the tracking lists.
                # Since tracking_ops/ids are re-initialized to empty lists [] above, we just populate them.
                
                # Fetch elite metadata from history arrays
                elite_ops = ops_comb[elite_global_indices]
                elite_ids = ids_comb[elite_global_indices]
                elite_depths = depths_comb[elite_global_indices]
                
                self.tracking_ops.extend(elite_ops)
                self.tracking_ids.extend(elite_ids)
                
                # Update depths array
                self.feat_depths[:n_elite] = elite_depths
                
                # Update start index for random generation so we don't overwrite elite feats
                start_gen_idx = n_elite

            # Pre-calculate correlations for Feature-Target Filter
            y_flat = np.ravel(y_for_fit)
            y_flat = np.nan_to_num(y_flat)
            y_std = np.std(y_flat)
            
            input_feat_corrs = np.zeros(X_scaled.shape[1])
            if y_std > 1e-9:
                for ft_idx in range(X_scaled.shape[1]):
                    ft_data = X_scaled[:, ft_idx]
                    ft_std = np.std(ft_data)
                    if ft_std > 1e-9:
                        try:
                            c_val = abs(np.corrcoef(ft_data, y_flat)[0, 1])
                            if not np.isnan(c_val):
                                input_feat_corrs[ft_idx] = c_val
                        except: pass

            # Generate features
            for i in range(start_gen_idx, gen_feats.shape[1]):
                # Retry loop to force GA to find signals stronger than parents
                max_retries = 3
                best_attempt = None
                best_corr_diff = -np.inf
                
                for attempt in range(max_retries):
                    dpth = self.rng.choice(self.depth_range, p=self.depth_weights)
                    ops = []
                    ids = []
                    
                    # Generate candidate
                    feat_val = self.feat_with_depth(X_scaled, dpth, ops, ids, context_data=getattr(self, '_current_data', None))
                    feat_val = self._clean_feature(feat_val)
                    
                    # --- Feature-Target Correlation Filter ---
                    gen_corr = 0.0
                    feat_std = np.std(feat_val)
                    
                    if y_std > 1e-9 and feat_std > 1e-9:
                        try:
                            c_new = abs(np.corrcoef(feat_val, y_flat)[0, 1])
                            if not np.isnan(c_new):
                                gen_corr = c_new
                        except: pass
                    
                    # Compare with parents
                    max_parent_corr = 0.0
                    if ids is not None and len(ids) > 0:
                        # Use unique IDs to avoid redundant lookups
                        max_parent_corr = max([input_feat_corrs[pid] for pid in ids])
                    
                    diff = gen_corr - max_parent_corr
                    
                    if best_attempt is None or diff > best_corr_diff:
                        best_attempt = (feat_val, dpth, ops, ids)
                        best_corr_diff = diff
                    
                    # If we beat or match the parents, stop retrying
                    if diff >= -1e-9:
                        break
                
                # Use best attempt
                final_feat, final_dpth, final_ops, final_ids = best_attempt

                gen_feats[:, i] = final_feat
                self.feat_depths[i] = final_dpth
                self.tracking_ops.append(final_ops)
                self.tracking_ids.append(final_ids)

            self.tracking_ids = np.array(self.tracking_ids + [[]], dtype='object')[:-1]
            self.tracking_ops = np.array(self.tracking_ops + [[]], dtype='object')[:-1]

            # Feature selection within iteration
            if self.verbose:
                print(f"  Selecting top {self.n_feats} features from {gen_feats.shape[1]} generated...")

            imps, estimators = self.get_feature_importances(gen_feats, y_for_fit, estimator, random_state)
            total_feats = np.argsort(imps)
            feat_args = total_feats[-self.n_feats:]
            gen_feats = gen_feats[:, feat_args]
            self.tracking_ids = self.tracking_ids[feat_args]
            self.tracking_ops = self.tracking_ops[feat_args]
            self.feat_depths = self.feat_depths[feat_args]

            # Save importances for next iteration's elitism
            previous_imps = imps[feat_args]

            # Store iteration results
            depths_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.feat_depths
            ids_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.tracking_ids
            ops_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.tracking_ops
            iters_comb[:, iteration * self.n_feats:(iteration + 1) * self.n_feats] = gen_feats

            # Update operator importance
            # Update operator importance with Diversity Control and Smoothing
            # 1. Decay existing weights (Smoothing) - ensure history matters but fades
            decay_factor = 0.8
            self.imp_operators *= decay_factor

            # 2. Count current iteration usage efficiently
            op_counts = np.zeros(len(self.operators))
            total_ops_used = 0
            
            # Create a quick lookup for operator indices
            op_to_idx = {op: i for i, op in enumerate(self.operators)}
            
            for feat_ops in self.tracking_ops:
                for op_info in feat_ops:
                    # op_info is tuple: (op_function, depth, params)
                    op_func = op_info[0]
                    if op_func in op_to_idx:
                        idx = op_to_idx[op_func]
                        op_counts[idx] += 1
                        total_ops_used += 1

            # 3. Apply updates with Diversity Penalty
            for i, count in enumerate(op_counts):
                if count > 0:
                    increment = count
                    
                    # Diversity Penalty: If one operator dominates (> 50% of usage)
                    usage_share = count / total_ops_used if total_ops_used > 0 else 0
                    if usage_share > 0.5:
                        increment *= 0.1  # Heavy penalty for monotony to force exploration
                        if self.verbose and iteration == 0: 
                             print(f"  Note: Diversity penalty applied to dominant operator '{self.operators[i].__name__}'")

                    # Time Series Multiplier (Reward for TS operators if configured)
                    if hasattr(self, 'time_series_operators') and self.operators[i] in self.time_series_operators:
                        increment *= self.effective_ts_weight_multiplier
                    
                    self.imp_operators[i] += increment

            # 4. Normalize with Floor (Prevent Starvation)
            # Calculate raw probabilities
            raw_weights = self.imp_operators / self.imp_operators.sum()
            
            # Apply dynamic floor (e.g., 5% or 1/(2*N) to ensure nothing hits true zero)
            min_weight = min(0.05, 1.0 / (2 * len(self.operators)))
            weights_floored = np.maximum(raw_weights, min_weight)
            
            # Renormalize to ensure sum is 1.0
            self.operator_weights = weights_floored / weights_floored.sum()

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

            imps, estimators = self.get_feature_importances(iters_comb, y_for_fit, estimator, random_state)
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
            gen_feats = self.fAnova_best.fit_transform(gen_feats, y_for_fit)

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

        if self._was_downsampled:
            if self.verbose:
                print(f"\n{'=' * 60}")
                print("APPLYING LEARNED FEATURES TO FULL DATASET")
                print(f"{'=' * 60}")
                print(f"Learned from: {self.n_rows_fit:,} sampled rows")
                print(f"Applying to: {self.n_rows_original:,} full rows")

            # Restore full dataset for transform
            original_current_data = getattr(self, '_current_data', None)  # Save sampled version safely

            if self.enable_time_series:
                # Restore full time series data
                if hasattr(self, '_full_current_data_backup'):
                    self._current_data = self._full_current_data_backup
                else:
                    # Reconstruct from original data
                    if isinstance(self._original_data_full, pd.DataFrame):
                        self._current_data = self._prepare_time_series_data(self._original_data_full)

            # Transform full dataset using learned features
            if isinstance(self._original_data_full, pd.DataFrame):
                gen_feats = self.transform(self._original_data_full)
            else:
                # For numpy arrays, need to reconstruct DataFrame with feature columns
                if self.feature_columns is not None and len(self.feature_columns) > 0:

                    full_df = pd.DataFrame(self._X_features_full, columns=self.feature_columns)
                    # Add datetime/groupby columns if available
                    if isinstance(self.original_data, pd.DataFrame):
                        if self.datetime_col and self.datetime_col in self.original_data.columns:
                            full_df[self.datetime_col] = self.original_data[self.datetime_col].values
                        for col in self.groupby_cols:
                            if col in self.original_data.columns:
                                full_df[col] = self.original_data[col].values
                    gen_feats = self.transform(full_df)
                else:
                    gen_feats = self.transform(self._X_features_full)

            if self.verbose:
                print(f"✓ Transform complete: {gen_feats.shape}")
                print(f"  Output dimensions now match original input")
                print(f"{'=' * 60}\n")

            # Clean up temporary storage
            if hasattr(self, '_X_features_full'):
                delattr(self, '_X_features_full')
            if hasattr(self, '_original_data_full'):
                delattr(self, '_original_data_full')
            if hasattr(self, '_full_current_data_backup'):
                delattr(self, '_full_current_data_backup')
        
        # Restore original order for fit output (if not downsampled)
        # If downsampled, transform() loop above already handled this.
        # If NOT downsampled, gen_feats is still in time-sorted order from the fit loop.
        if not self._was_downsampled and self.enable_time_series and hasattr(self, '_current_data') and self._current_data is not None and hasattr(self._current_data, 'columns') and '_original_index' in self._current_data.columns:
            # Create DataFrame with the restored index to align back to input X
            res_df = pd.DataFrame(gen_feats)
            res_df.index = self._current_data['_original_index']
            
            # We need to sort by index to restore original order
            # (Assuming original index values are monotonic 0..N or similar, or just relying on index align)
            res_df = res_df.sort_index()
            return res_df.values

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
            if self.feature_columns is not None and len(self.feature_columns) > 0:
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
                # Use standard identification logic
                feature_cols = self._identify_feature_columns(X)
                X_features = X[feature_cols].values
                X_features = np.array(X_features, dtype=float)

            # Update current data for time series operations
            context_data = None
            if self.enable_time_series:
                context_data = self._prepare_time_series_data(X)
                self._current_data = context_data # Maintain for backward compatibility
                
                # CRITICAL FIX: Use sorted features for generation to match rolling window order
                X_features = context_data[self.feature_columns].values
                X_features = np.array(X_features, dtype=float)
        else:
            X_features = X
            X_features = np.array(X_features, dtype=float)
            context_data = None

        X_scaled = self.scaler.transform(X_features)
        self.n_rows = X_scaled.shape[0]
        gen_feats = np.zeros((self.n_rows, len(self.tracking_ids)))

        for i in range(gen_feats.shape[1]):
            dpth = self.feat_depths[i]
            op_ls = self.tracking_ops[i].copy()
            id_ls = self.tracking_ids[i].copy()
            gen_feats[:, i] = self.feat_with_depth_gen(X_scaled, dpth, op_ls, id_ls, context_data=context_data)
            # Clean generated feature unconditionally (prevent overflow/NaNs)
            gen_feats[:, i] = self._clean_feature(gen_feats[:, i])
                
        # Combine generated features with original scaled features
        gen_feats = np.hstack((gen_feats, X_scaled))

        # Apply the fitted feature selector BEFORE restoring row order.
        #
        # This used to sit after the time-series early-return below, which made
        # it unreachable whenever time series was enabled: fit() would return k
        # selected columns while transform() returned the full unselected
        # width, so train and test matrices silently disagreed in shape.
        #
        # Selecting columns and restoring row order are independent, so doing
        # the column selection first is safe and lets both paths share it.
        if self.selection == 'fAnova':
            gen_feats = self.fAnova_best.transform(gen_feats)

        # Restore original row order if time series sorting was applied
        if self.enable_time_series and context_data is not None and '_original_index' in context_data.columns:
            # Create DataFrame with the restored index to align back to input X
            res_df = pd.DataFrame(gen_feats)
            res_df.index = context_data['_original_index']

            # Reindex to match original input X
            if isinstance(X, pd.DataFrame):
                res_df = res_df.reindex(X.index)
            else:
                # If X was array, _original_index is 0..N
                res_df = res_df.sort_index()

            return res_df.values

        return gen_feats

    # Enhanced feat_with_depth method to support datetime-aware operations with deterministic parameters
    def feat_with_depth(self, X, depth, op_ls, feat_ls, context_data=None):
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
        
        # Generate parameters for time series operators
        params = {}
        if self.enable_time_series and hasattr(self, 'time_series_operators') and op in self.time_series_operators:
             # Identify which type of parameter is needed
            op_name = getattr(op, '__name__', str(op))
            
            # Select time step for this operation branch
            params['time_step'] = self.rng.choice(self.window_step_options) if self.window_step_options else 'D'
            
            if 'lag' in op_name or 'diff' in op_name or 'pct_change' in op_name or 'momentum' in op_name:
                params['lag_period'] = self.rng.choice(self.lag_periods)
            elif 'ewm' in op_name:
                params['window_size'] = self.rng.choice(self.window_sizes)
            elif 'seasonal' in op_name or 'weekday' in op_name or 'month' in op_name:
                # No parameters needed or fixed logic
                pass
            elif 'trend' in op_name:
                # Trend usually uses window
                params['window_size'] = self.rng.choice(self.window_sizes)
            else:
                # Default to window size for rolling operations
                params['window_size'] = self.rng.choice(self.window_sizes)

        if op in self.binary_operators:
            feat_1 = self.feat_with_depth(X, depth, op_ls, feat_ls, context_data)
            feat_2 = self.feat_with_depth(X, depth, op_ls, feat_ls, context_data)
            op_ls.append((op, depth, params))
            
            # Apply with parameters if time series operator
            if self.enable_time_series and op in self.time_series_operators:
                result = op(feat_1, feat_2, context_data=context_data, **params)
            else:
                result = op(feat_1, feat_2)
            
            return self._clean_feature(result)

        elif op in self.unary_operators:
            # Note where this subtree's leaves begin, so we can identify which
            # column the operator consumed rather than relying on shared state.
            leaves_before = len(feat_ls)
            feat_1 = self.feat_with_depth(X, depth, op_ls, feat_ls, context_data)

            # Record which input column this operator actually consumed, so the
            # recipe is self-describing. Reading it off self._current_feature_index
            # at apply time was ambiguous: inside a binary node the second leaf's
            # index had already overwritten the first's by the time the parent
            # operator ran.
            #
            # A unary operator applies to the value produced by its subtree. For
            # a bare leaf that is unambiguous. For a deeper subtree there is no
            # single source column, so we take the FIRST leaf of that subtree --
            # matching what the old shared-state code did when it happened to be
            # correct, and now stated explicitly in the recipe.
            if self.enable_time_series and op in self.time_series_operators:
                params = dict(params)
                subtree_leaves = feat_ls[leaves_before:]
                params['feature_index'] = (
                    int(subtree_leaves[0]) if subtree_leaves else None
                )

            op_ls.append((op, depth, params))

            # Apply with parameters if time series operator
            if self.enable_time_series and op in self.time_series_operators:
                result = op(feat_1, context_data=context_data, **params)
            else:
                result = op(feat_1)

            return self._clean_feature(result)

    def feat_with_depth_gen(self, X, depth, op_ls, feat_ls, context_data=None):
        """ Reproduce generated features with new data - Enhanced to handle datetime-aware time series operators """
        if depth == 0:
            feat_ind = feat_ls.pop()
            # Set current feature index for time series operations
            if self.enable_time_series:
                self._current_feature_index = feat_ind
            return X[:, feat_ind]

        depth -= 1
        op_info = op_ls.pop()
        op = op_info[0]
        # Helper to extract params safely (handle legacy format without params)
        # Robustly check for parameters without triggering array-truth errors
        if isinstance(op_info, (list, tuple)) and len(op_info) > 2:
            params = op_info[2]
        else:
            params = {}

        if op in self.binary_operators:
            feat_1 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls, context_data)
            feat_2 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls, context_data)
            
            if self.enable_time_series and op in self.time_series_operators:
                result = op(feat_2, feat_1, context_data=context_data, **params)
            else:
                result = op(feat_2, feat_1)
                
            return self._clean_feature(result)

        elif op in self.unary_operators:
            feat_1 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls, context_data)
            
            if self.enable_time_series and op in self.time_series_operators:
                result = op(feat_1, context_data=context_data, **params)
            else:
                result = op(feat_1)
                
            return self._clean_feature(result)

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

            # Define CV strategy
            if self.enable_time_series:
                 cv_strategy = TimeSeriesSplit(n_splits=5) # Increased for consistency
            else:
                 cv_strategy = 5

            models_score[estimator] = cross_val_score(model, X, y, cv=cv_strategy, scoring=scorer).mean()

        best_estimator = max(models_score, key=models_score.get)
        best_model = estimators_dic[best_estimator]
        best_model.fit(X, y)
        return best_model

    def get_feature_importances(self, X, y, estimator, random_state, sample_count=1, sample_size=5, n_jobs=1):
        """Return feature importances by specified method - Original method"""
        importance_sum = np.zeros(X.shape[1])
        total_estimators = []

        if self.enable_time_series:
            # Use TimeSeriesSplit to evaluate importance across time
            tscv = TimeSeriesSplit(n_splits=sample_size)
            # Inspect splits to handle potential small dataset issues gracefully if needed,
            # though sklearn handles basic splitting.
            # We convert to list to iterate easily or handle max splits.
            try:
                folds = list(tscv.split(X))
            except ValueError:
                # Fallback if too few samples for splits
                folds = [] 
        else:
            folds = [None] * sample_count 
            
        # Determine number of iterations based on mode
        # If time series, we iterate through folds. If not, we iterate sample_count times.
        if self.enable_time_series:
            iterations = len(folds)
            if iterations == 0 and sample_count > 0:
                 # Fallback to single pass on full data if splitting failed (e.g. tiny data)
                 # Or just fallback to random? Let's stick to full data to respect time order.
                 iterations = 1
                 folds = [(np.arange(len(X)-1), np.arange(len(X)-1, len(X)))] # Dummy split
        else:
            iterations = sample_count

        for i in range(iterations):
            if self.enable_time_series and len(folds) > 0:
                train_index, test_index = folds[i]
                sampled_ind = test_index
            else:
                 # Draw from self.rng (seeded by fit's random_state) rather than the
                 # global numpy RNG, so that fit() is reproducible from random_state
                 # alone without the caller also having to seed numpy globally.
                 sampled_ind = self.rng.choice(np.arange(self.n_rows), size=self.n_rows // sample_size, replace=False)

            sampled_X = X[sampled_ind]
            
            # Safe indexing for y
            if hasattr(y, 'iloc'):
                sampled_y = y.iloc[sampled_ind]
            elif hasattr(y, 'values'):
                 sampled_y = y.values[sampled_ind]
            else:
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
                if lgb_imps.sum() > 0:
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
        """ Check correlations among the selected features - Robust method """
        cor_thresh = 0.85 if self.enable_time_series else 0.7
        
        # Safe conversion to DataFrame
        df_feats = pd.DataFrame(feats)
        corr_matrix = df_feats.corr().abs()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Identify columns to drop
        # Explicitly check each column to ensure scalar boolean evaluation
        to_drop = [column for column in upper.columns if bool(np.any(upper[column] > cor_thresh))]
        
        # Drop columns
        df_reduced = df_feats.drop(columns=to_drop)
        
        return df_reduced.values, to_drop

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

        # Collapse runs of identical consecutive paths.
        #
        # The index used to be `path_list[i - 1]`, which at i == 0 wraps to
        # path_list[-1] -- the LAST path. So whenever a tree's first and last
        # root-to-leaf paths happened to match, the first path was silently
        # dropped and never counted in the split-frequency vector.
        new_list = []
        for i, current in enumerate(path_list):
            if i == 0 or current != path_list[i - 1]:
                new_list.append(current)
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
        if hasattr(self, 'min_window_days') and hasattr(self, 'max_window_days'):
            filtered = [w for w in default_windows
                        if self.min_window_days <= w.days <= self.max_window_days]

            # Return top n_windows if that parameter exists
            if hasattr(self, 'n_windows') and len(filtered) > self.n_windows:
                return filtered[:self.n_windows]
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
        if hasattr(self, 'groupby_cols') and self.groupby_cols is not None and len(self.groupby_cols) > 0:
            exclude_cols.extend([col for col in self.groupby_cols if col in X.columns])

        # Get all potential feature columns
        all_feature_cols = [col for col in X.columns if col not in exclude_cols]

        # Filter out non-numeric columns
        numeric_feature_cols = []
        for col in all_feature_cols:
            is_numeric = False
            
            # 1. Fast Pandas check for proper numeric types
            if pd.api.types.is_numeric_dtype(X[col]):
                 # Double check it's not a boolean (treated as numeric by some but maybe we want float/int)
                 # BigFeat handles bool as 0/1 usually fine.
                 is_numeric = True
            
            # 2. Check for object columns containing numbers (fallback)
            else:
                col_dtype = str(X[col].dtype)
                
                # Skip explicit datetime types (redundant if is_numeric_dtype is False but safe)
                if (col_dtype.startswith('datetime') or 
                    col_dtype.startswith('<M8') or 
                    col_dtype.startswith('timedelta')):
                    if self.verbose: 
                        print(f"  Skipping datetime column '{col}' (type: {col_dtype})")
                    continue
                    
                # Handle object/string columns
                try:
                    valid_vals = X[col].dropna()
                    if len(valid_vals) > 0:
                        val = valid_vals.iloc[0]
                        # Check it's not a datetime object
                        if hasattr(val, 'year') and hasattr(val, 'month'):
                            if self.verbose:
                                print(f"  Skipping datetime-like column '{col}'")
                            continue
                            
                        # Try casting to float (Task 8 fix)
                        # Secure against array-like objects being in the cell
                        if hasattr(val, '__len__') and not isinstance(val, str):
                             # excessive recursion protection
                             pass
                        else:
                             float(val)
                             is_numeric = True
                             
                except (ValueError, TypeError):
                    pass
            
            if is_numeric:
                numeric_feature_cols.append(col)

        return numeric_feature_cols


    def get_window_detection_summary(self):
        """
        Get summary of window detection results

        Returns:
        --------
        dict
            Dictionary containing DFT detection summary
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
                    # Use generic confidence threshold if available, else dft specific
                    threshold = getattr(self, 'confidence_threshold', getattr(self, 'confidence_threshold', 0))
                    print(f"Threshold: {threshold}")
        else:
            print(f"Reason: {summary['detection_strategy']}")

        print("=" * 60 + "\n")
