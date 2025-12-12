import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import List, Tuple, Dict, Optional
import warnings


class ACFWindowDetector:
    """
    Autocorrelation Function (ACF) based window size detection for BigFeat.

    Advantages:
    - Simple and interpretable
    - Robust to noise
    - Works well with short time series
    - Directly identifies periodic patterns through correlation peaks

    Best for:
    - Regular sampling intervals
    - Clear periodic patterns
    - Shorter time series (50-500 samples)
    """

    def __init__(self,
                 min_window_days: int = 3,
                 max_window_days: int = 365,
                 n_windows: int = 6,
                 confidence_threshold: float = 0.3,
                 min_peak_height: float = 0.2,
                 min_peak_prominence: float = 0.1,
                 verbose: bool = True):
        """
        Initialize ACF Window Detector

        Parameters:
        -----------
        min_window_days : int
            Minimum window size in days
        max_window_days : int
            Maximum window size in days
        n_windows : int
            Number of window sizes to return
        confidence_threshold : float
            Minimum confidence score (ACF peak height) to consider periodicity reliable
        min_peak_height : float
            Minimum ACF value to consider as a significant peak (0-1)
        min_peak_prominence : float
            Minimum prominence for peak detection
        verbose : bool
            Whether to print progress messages
        """
        self.min_window_days = min_window_days
        self.max_window_days = max_window_days
        self.n_windows = n_windows
        self.confidence_threshold = confidence_threshold
        self.min_peak_height = min_peak_height
        self.min_peak_prominence = min_peak_prominence
        self.verbose = verbose

    def detect_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Automatically detect datetime column in DataFrame"""
        datetime_candidates = []

        for col in df.columns:
            col_dtype = str(df[col].dtype)

            if col_dtype.startswith('datetime') or col_dtype.startswith('<M8'):
                datetime_candidates.append((col, 'explicit'))
                continue

            if col_dtype == 'object':
                try:
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if sample is not None:
                        if hasattr(sample, 'year') and hasattr(sample, 'month'):
                            datetime_candidates.append((col, 'object_datetime'))
                        else:
                            pd.to_datetime(sample)
                            datetime_candidates.append((col, 'string_datetime'))
                except:
                    continue

            col_lower = col.lower()
            if any(hint in col_lower for hint in ['date', 'time', 'timestamp', 'datetime']):
                if col not in [c[0] for c in datetime_candidates]:
                    datetime_candidates.append((col, 'name_hint'))

        if datetime_candidates:
            explicit = [c for c in datetime_candidates if c[1] == 'explicit']
            if explicit:
                if self.verbose:
                    print(f"Auto-detected datetime column: '{explicit[0][0]}'")
                return explicit[0][0]

            if self.verbose:
                print(
                    f"Auto-detected datetime column: '{datetime_candidates[0][0]}' (type: {datetime_candidates[0][1]})")
            return datetime_candidates[0][0]

        if self.verbose:
            print("Warning: No datetime column detected")
        return None

    def _preprocess_signal(self, series: np.ndarray) -> np.ndarray:
        """
        Preprocess time series signal for ACF analysis

        Parameters:
        -----------
        series : ndarray
            Input time series

        Returns:
        --------
        ndarray
            Preprocessed signal
        """
        # Handle NaN values
        series = pd.Series(series).ffill().bfill().fillna(0).values

        # Remove linear trend
        x = np.arange(len(series))
        if len(series) > 1:
            coeffs = np.polyfit(x, series, 1)
            trend = np.polyval(coeffs, x)
            detrended = series - trend
        else:
            detrended = series - np.mean(series)

        return detrended

    def _compute_acf(self, series: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
        """
        Compute autocorrelation function

        Parameters:
        -----------
        series : ndarray
            Input time series
        max_lag : int, optional
            Maximum lag to compute (default: len(series) - 1)

        Returns:
        --------
        acf_values : ndarray
            Autocorrelation values for each lag
        """
        if max_lag is None:
            max_lag = len(series) - 1

        max_lag = min(max_lag, len(series) - 1)

        # Normalize series
        series_mean = np.mean(series)
        series_std = np.std(series)

        if series_std < 1e-10:
            # Constant series, return zeros
            return np.zeros(max_lag + 1)

        series_normalized = (series - series_mean) / series_std

        acf_values = np.zeros(max_lag + 1)
        acf_values[0] = 1.0  # ACF at lag 0 is always 1

        # Compute ACF for each lag
        for lag in range(1, max_lag + 1):
            if lag < len(series):
                # Calculate correlation coefficient
                acf_values[lag] = np.corrcoef(
                    series_normalized[:-lag],
                    series_normalized[lag:]
                )[0, 1]

        # Replace NaN values with 0
        acf_values = np.nan_to_num(acf_values, nan=0.0)

        return acf_values

    def _find_acf_peaks(self, acf_values: np.ndarray, min_lag: int = 1) -> Tuple[np.ndarray, Dict]:
        """
        Find significant peaks in ACF

        Parameters:
        -----------
        acf_values : ndarray
            Autocorrelation values
        min_lag : int
            Minimum lag to consider for peaks

        Returns:
        --------
        peak_lags : ndarray
            Lag values at peaks
        peak_metadata : dict
            Additional information about peaks
        """
        # Only consider lags >= min_lag
        acf_subset = acf_values[min_lag:]

        if len(acf_subset) < 3:
            return np.array([]), {'peak_heights': np.array([]), 'n_peaks': 0}

        # Find peaks with height and prominence thresholds
        peaks, properties = find_peaks(
            acf_subset,
            height=self.min_peak_height,
            distance=max(1, min_lag // 2),  # Minimum distance between peaks
            prominence=self.min_peak_prominence
        )

        # Adjust peak indices to account for min_lag offset
        peak_lags = peaks + min_lag

        if len(peak_lags) == 0:
            return np.array([]), {'peak_heights': np.array([]), 'n_peaks': 0}

        # Get peak heights
        peak_heights = acf_values[peak_lags]

        # Sort by height (strongest correlations first)
        sorted_indices = np.argsort(peak_heights)[::-1]
        peak_lags = peak_lags[sorted_indices]
        peak_heights = peak_heights[sorted_indices]

        metadata = {
            'peak_heights': peak_heights,
            'n_peaks': len(peak_lags),
            'properties': properties
        }

        return peak_lags, metadata

    def detect_optimal_windows(self,
                               df: pd.DataFrame,
                               datetime_col: str,
                               feature_cols: List[str],
                               sampling_rate: str = 'D') -> Tuple[List[pd.Timedelta], Dict[str, float]]:
        """
        Detect optimal window sizes using Autocorrelation Function

        Parameters:
        -----------
        df : DataFrame
            Time series data
        datetime_col : str
            Name of datetime column
        feature_cols : list
            List of feature column names to analyze
        sampling_rate : str
            Sampling rate of the time series ('D' for daily, 'H' for hourly, etc.)

        Returns:
        --------
        window_sizes : list of pd.Timedelta
            Optimal window sizes for BigFeat
        confidence_scores : dict
            Confidence score for each feature (based on peak height)
        """
        if datetime_col not in df.columns:
            raise ValueError(f"Datetime column '{datetime_col}' not found in DataFrame")

        # Sort by datetime
        df_sorted = df.sort_values(datetime_col).reset_index(drop=True)

        detected_periods = []
        confidence_scores = {}

        if self.verbose:
            print(f"\n=== ACF Window Detection ===")
            print(f"Analyzing {len(feature_cols)} features...")

        for col in feature_cols:
            if col not in df_sorted.columns:
                warnings.warn(f"Feature column '{col}' not found, skipping")
                continue

            try:
                # Extract series
                series = df_sorted[col].values

                if len(series) < 10:
                    warnings.warn(f"Series '{col}' too short for ACF analysis, skipping")
                    continue

                # Preprocess signal
                processed_signal = self._preprocess_signal(series)

                # Compute ACF up to max_window_days
                max_lag = min(len(series) - 1, self.max_window_days)
                acf_values = self._compute_acf(processed_signal, max_lag)

                # Find peaks in ACF
                peak_lags, peak_metadata = self._find_acf_peaks(acf_values, min_lag=self.min_window_days)

                if len(peak_lags) == 0:
                    if self.verbose:
                        print(f"  '{col}': No significant peaks found")
                    continue

                # Convert lags to days based on sampling rate
                peak_periods = [self._convert_to_days(lag, sampling_rate) for lag in peak_lags]

                # Filter by bounds
                valid_periods = [p for p in peak_periods
                                 if self.min_window_days <= p <= self.max_window_days]

                if valid_periods:
                    # Take top 3 periods per feature
                    detected_periods.extend(valid_periods[:3])

                    # Confidence based on strongest peak height
                    max_peak_height = peak_metadata['peak_heights'][0] if len(peak_metadata['peak_heights']) > 0 else 0
                    confidence = float(max_peak_height)
                    confidence_scores[col] = confidence

                    if self.verbose:
                        print(f"  '{col}': detected {len(peak_lags)} peaks")
                        print(f"    Top periods: {[f'{p:.1f}' for p in valid_periods[:3]]} days")
                        print(f"    Peak heights: {[f'{h:.3f}' for h in peak_metadata['peak_heights'][:3]]}")
                        print(f"    Confidence: {confidence:.3f}")

            except Exception as e:
                warnings.warn(f"ACF analysis failed for feature '{col}': {str(e)}")
                continue

        if not detected_periods:
            if self.verbose:
                print("\nNo valid periods detected, using default windows")
            return self._get_default_windows(), {}

        # Generate multi-scale windows from detected periods
        window_sizes = self._generate_multiscale_windows(detected_periods)

        if self.verbose:
            print(f"\nFinal window sizes: {[w.days for w in window_sizes]} days")
            avg_conf = np.mean(list(confidence_scores.values())) if confidence_scores else 0.0
            print(f"Average confidence: {avg_conf:.3f}")

        return window_sizes, confidence_scores

    def _convert_to_days(self, period_samples: float, sampling_rate: str) -> float:
        """Convert period in samples to days based on sampling rate"""
        rate_to_days = {
            'D': 1.0,  # Daily
            'H': 1 / 24.0,  # Hourly
            'W': 7.0,  # Weekly
            'M': 30.0,  # Monthly (approximate)
            'Q': 90.0,  # Quarterly (approximate)
            'Y': 365.0  # Yearly (approximate)
        }
        days_per_sample = rate_to_days.get(sampling_rate, 1.0)
        return period_samples * days_per_sample

    def _generate_multiscale_windows(self, detected_periods: List[float]) -> List[pd.Timedelta]:
        """Generate multi-scale windows from detected periods"""
        unique_periods = list(set([int(p) for p in detected_periods if p > 0]))

        # Add harmonics and sub-harmonics
        multi_scale_periods = []
        for period in unique_periods:
            if period >= 4:
                multi_scale_periods.append(period // 2)  # Sub-harmonic
            multi_scale_periods.append(period)  # Fundamental
            multi_scale_periods.append(period * 2)  # Harmonic
            if period <= self.max_window_days // 4:
                multi_scale_periods.append(period * 4)  # 2nd harmonic

        # Remove duplicates and apply constraints
        multi_scale_periods = sorted(set([
            p for p in multi_scale_periods
            if self.min_window_days <= p <= self.max_window_days
        ]))

        # Select evenly spaced windows if we have too many
        if len(multi_scale_periods) > self.n_windows:
            indices = np.linspace(0, len(multi_scale_periods) - 1, self.n_windows, dtype=int)
            selected_periods = [multi_scale_periods[i] for i in indices]
        else:
            selected_periods = multi_scale_periods

        # Ensure we have at least some windows
        if not selected_periods:
            selected_periods = [self.min_window_days]

        # Convert to Timedelta
        window_sizes = [pd.Timedelta(days=int(p)) for p in selected_periods]

        return window_sizes

    def _get_default_windows(self) -> List[pd.Timedelta]:
        """Return default window sizes when ACF detection fails"""
        default_days = [7, 14, 30, 90, 180, 365]
        default_days = [d for d in default_days
                        if self.min_window_days <= d <= self.max_window_days]
        return [pd.Timedelta(days=d) for d in default_days[:self.n_windows]]

    def assess_periodicity(self,
                           df: pd.DataFrame,
                           datetime_col: str,
                           feature_cols: List[str]) -> Tuple[bool, float, Dict[str, float]]:
        """
        Assess if time series data exhibits strong periodicity using ACF

        Parameters:
        -----------
        df : DataFrame
            Time series data
        datetime_col : str
            Name of datetime column
        feature_cols : list
            List of feature columns to analyze

        Returns:
        --------
        is_periodic : bool
            Whether data exhibits strong periodicity
        avg_confidence : float
            Average confidence score across features
        feature_confidences : dict
            Individual confidence scores per feature
        """
        _, confidence_scores = self.detect_optimal_windows(df, datetime_col, feature_cols)

        if not confidence_scores:
            return False, 0.0, {}

        avg_confidence = np.mean(list(confidence_scores.values()))
        is_periodic = avg_confidence >= self.confidence_threshold

        if self.verbose:
            print(f"\nACF Periodicity Assessment:")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  Threshold: {self.confidence_threshold}")
            print(f"  Result: {'PERIODIC' if is_periodic else 'NON-PERIODIC'}")

        return is_periodic, avg_confidence, confidence_scores

    def smart_window_selection(self,
                               df: pd.DataFrame,
                               datetime_col: str,
                               feature_cols: List[str]) -> Tuple[List[pd.Timedelta], str]:
        """
        Smart window selection with hybrid strategy
        Compatible with BigFeat's DFTWindowDetector API
        """
        windows, confidence_scores = self.detect_optimal_windows(df, datetime_col, feature_cols)

        if not confidence_scores:
            return self._get_default_windows(), 'standard'

        avg_confidence = np.mean(list(confidence_scores.values()))

        if avg_confidence >= self.confidence_threshold:
            strategy = 'acf'
        elif avg_confidence > self.confidence_threshold * 0.6:
            # Hybrid: combine ACF with standard windows
            standard_windows = self._get_default_windows()
            combined = list(set(windows + standard_windows))
            combined_sorted = sorted(combined, key=lambda x: x.days)[:self.n_windows]
            return combined_sorted, 'hybrid'
        else:
            strategy = 'standard'
            windows = self._get_default_windows()

        return windows, strategy