import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from typing import List, Tuple, Dict, Optional
import warnings


class LombScargleWindowDetector:
    """
    Lomb-Scargle Periodogram based window size detection for BigFeat.

    Advantages:
    - Handles unevenly sampled data
    - Robust to missing values
    - Good for irregular time series
    - Provides clear frequency domain representation

    Best for:
    - Irregular sampling intervals
    - Missing data
    - Astronomical/environmental time series
    - Data with gaps
    """

    def __init__(self,
                 min_window_days: int = 3,
                 max_window_days: int = 365,
                 n_windows: int = 6,
                 confidence_threshold: float = 0.3,
                 min_peak_power: float = 0.1,
                 min_peak_prominence: float = 0.05,
                 n_frequencies: int = 1000,
                 verbose: bool = True):
        """
        Initialize Lomb-Scargle Window Detector

        Parameters:
        -----------
        min_window_days : int
            Minimum window size in days
        max_window_days : int
            Maximum window size in days
        n_windows : int
            Number of window sizes to return
        confidence_threshold : float
            Minimum confidence score to consider periodicity reliable
        min_peak_power : float
            Minimum power to consider as a significant peak
        min_peak_prominence : float
            Minimum prominence for peak detection
        n_frequencies : int
            Number of frequencies to evaluate in the periodogram
        verbose : bool
            Whether to print progress messages
        """
        self.min_window_days = min_window_days
        self.max_window_days = max_window_days
        self.n_windows = n_windows
        self.confidence_threshold = confidence_threshold
        self.min_peak_power = min_peak_power
        self.min_peak_prominence = min_peak_prominence
        self.n_frequencies = n_frequencies
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
        Preprocess time series signal for Lomb-Scargle analysis

        Parameters:
        -----------
        series : ndarray
            Input time series

        Returns:
        --------
        ndarray
            Preprocessed signal (mean-centered and normalized)
        """
        # Remove mean
        series_centered = series - np.nanmean(series)

        # Normalize by standard deviation
        series_std = np.nanstd(series)
        if series_std > 1e-10:
            series_normalized = series_centered / series_std
        else:
            series_normalized = series_centered

        return series_normalized

    def _compute_lomb_scargle(self,
                              times: np.ndarray,
                              values: np.ndarray,
                              min_period: float,
                              max_period: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Lomb-Scargle periodogram

        Parameters:
        -----------
        times : ndarray
            Time values (must be sorted, can be irregularly spaced)
        values : ndarray
            Signal values (can have NaN for missing data)
        min_period : float
            Minimum period to consider (in same units as times)
        max_period : float
            Maximum period to consider (in same units as times)

        Returns:
        --------
        frequencies : ndarray
            Frequency values
        power : ndarray
            Power spectrum (normalized)
        periods : ndarray
            Period values corresponding to frequencies
        """
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        times_clean = times[valid_mask]
        values_clean = values[valid_mask]

        if len(times_clean) < 10:
            raise ValueError("Too few valid data points for Lomb-Scargle analysis")

        # Generate frequency grid
        min_freq = 1.0 / max_period
        max_freq = 1.0 / min_period

        # Adjust n_frequencies based on data length
        n_freqs = min(self.n_frequencies, len(times_clean) * 10)
        frequencies = np.linspace(min_freq, max_freq, n_freqs)

        # Compute Lomb-Scargle periodogram
        # Angular frequencies for scipy.signal.lombscargle
        angular_freqs = 2 * np.pi * frequencies

        power = signal.lombscargle(times_clean, values_clean, angular_freqs, normalize=True)

        # Convert to periods
        periods = 1.0 / frequencies

        return frequencies, power, periods

    def detect_optimal_windows(self,
                               df: pd.DataFrame,
                               datetime_col: str,
                               feature_cols: List[str],
                               sampling_rate: str = 'D') -> Tuple[List[pd.Timedelta], Dict[str, float]]:
        """
        Detect optimal window sizes using Lomb-Scargle periodogram

        Parameters:
        -----------
        df : DataFrame
            Time series data (can have irregular sampling or missing values)
        datetime_col : str
            Name of datetime column
        feature_cols : list
            List of feature column names to analyze
        sampling_rate : str
            Expected sampling rate ('D' for daily, 'H' for hourly, etc.)
            Used for unit conversion

        Returns:
        --------
        window_sizes : list of pd.Timedelta
            Optimal window sizes for BigFeat
        confidence_scores : dict
            Confidence score for each feature (based on peak power)
        """
        if datetime_col not in df.columns:
            raise ValueError(f"Datetime column '{datetime_col}' not found in DataFrame")

        # Sort by datetime
        df_sorted = df.sort_values(datetime_col).reset_index(drop=True)

        # Convert datetime to numeric (seconds since first observation)
        times = (df_sorted[datetime_col] - df_sorted[datetime_col].iloc[0]).dt.total_seconds().values

        detected_periods = []
        confidence_scores = {}

        if self.verbose:
            print(f"\n=== Lomb-Scargle Window Detection ===")
            print(f"Analyzing {len(feature_cols)} features...")

        for col in feature_cols:
            if col not in df_sorted.columns:
                warnings.warn(f"Feature column '{col}' not found, skipping")
                continue

            try:
                # Extract series
                series = df_sorted[col].values

                # Check for valid data
                n_valid = np.sum(~np.isnan(series))
                if n_valid < 10:
                    warnings.warn(f"Series '{col}' has too few valid points ({n_valid}), skipping")
                    continue

                # Preprocess signal
                series_processed = self._preprocess_signal(series)

                # Convert period bounds to seconds
                min_period_sec = self.min_window_days * 86400  # days to seconds
                max_period_sec = self.max_window_days * 86400

                # Compute Lomb-Scargle periodogram
                frequencies, power, periods_sec = self._compute_lomb_scargle(
                    times, series_processed, min_period_sec, max_period_sec
                )

                # Find peaks in power spectrum
                peaks, properties = find_peaks(
                    power,
                    height=self.min_peak_power,
                    distance=max(1, len(power) // 100),  # At least 1% of spectrum apart
                    prominence=self.min_peak_prominence
                )

                if len(peaks) == 0:
                    if self.verbose:
                        print(f"  '{col}': No significant peaks found")
                    continue

                # Sort peaks by power
                peak_powers = power[peaks]
                peak_periods_sec = periods_sec[peaks]
                sorted_indices = np.argsort(peak_powers)[::-1]

                peak_periods_sec = peak_periods_sec[sorted_indices]
                peak_powers = peak_powers[sorted_indices]

                # Convert to days
                peak_periods_days = peak_periods_sec / 86400

                # Filter by bounds
                valid_periods = [p for p in peak_periods_days
                                 if self.min_window_days <= p <= self.max_window_days]

                if valid_periods:
                    # Take top 3 periods per feature
                    detected_periods.extend(valid_periods[:3])

                    # Confidence based on maximum power (normalized)
                    max_power = peak_powers[0] if len(peak_powers) > 0 else 0
                    # Square root for better scaling to [0, 1] range
                    confidence = float(np.sqrt(max_power))
                    confidence_scores[col] = confidence

                    if self.verbose:
                        print(f"  '{col}': detected {len(peaks)} peaks ({n_valid}/{len(series)} valid points)")
                        print(f"    Top periods: {[f'{p:.1f}' for p in valid_periods[:3]]} days")
                        print(f"    Peak powers: {[f'{p:.3f}' for p in peak_powers[:3]]}")
                        print(f"    Confidence: {confidence:.3f}")

            except Exception as e:
                warnings.warn(f"Lomb-Scargle analysis failed for feature '{col}': {str(e)}")
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
        """Return default window sizes when Lomb-Scargle detection fails"""
        default_days = [7, 14, 30, 90, 180, 365]
        default_days = [d for d in default_days
                        if self.min_window_days <= d <= self.max_window_days]
        return [pd.Timedelta(days=d) for d in default_days[:self.n_windows]]

    def assess_periodicity(self,
                           df: pd.DataFrame,
                           datetime_col: str,
                           feature_cols: List[str]) -> Tuple[bool, float, Dict[str, float]]:
        """
        Assess if time series data exhibits strong periodicity using Lomb-Scargle

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
            print(f"\nLomb-Scargle Periodicity Assessment:")
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
            strategy = 'lomb_scargle'
        elif avg_confidence > self.confidence_threshold * 0.6:
            # Hybrid: combine Lomb-Scargle with standard windows
            standard_windows = self._get_default_windows()
            combined = list(set(windows + standard_windows))
            combined_sorted = sorted(combined, key=lambda x: x.days)[:self.n_windows]
            return combined_sorted, 'hybrid'
        else:
            strategy = 'standard'
            windows = self._get_default_windows()

        return windows, strategy