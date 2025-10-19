import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from typing import List, Tuple, Dict, Optional, Union
import warnings


class DFTWindowDetector:
    """
    Automated window size detection using Discrete Fourier Transform (DFT)
    for BigFeat time series feature engineering.
    """

    def __init__(self,
                 min_window_days: int = 3,
                 max_window_days: int = 365,
                 n_windows: int = 6,
                 confidence_threshold: float = 0.3,
                 verbose: bool = True):
        """
        Initialize DFT Window Detector

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
            (ratio of dominant peak to second peak)
        verbose : bool
            Whether to print progress messages
        """
        self.min_window_days = min_window_days
        self.max_window_days = max_window_days
        self.n_windows = n_windows
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

    def detect_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Automatically detect datetime column in DataFrame

        Parameters:
        -----------
        df : DataFrame
            Input dataframe

        Returns:
        --------
        str or None
            Name of detected datetime column, or None if not found
        """
        datetime_candidates = []

        for col in df.columns:
            col_dtype = str(df[col].dtype)

            # Check for explicit datetime types
            if col_dtype.startswith('datetime') or col_dtype.startswith('<M8'):
                datetime_candidates.append((col, 'explicit'))
                continue

            # Check for object columns that might be datetime
            if col_dtype == 'object':
                try:
                    # Try parsing a sample
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if sample is not None:
                        if hasattr(sample, 'year') and hasattr(sample, 'month'):
                            datetime_candidates.append((col, 'object_datetime'))
                        else:
                            # Try parsing as string
                            pd.to_datetime(sample)
                            datetime_candidates.append((col, 'string_datetime'))
                except:
                    continue

            # Check column name hints
            col_lower = col.lower()
            if any(hint in col_lower for hint in ['date', 'time', 'timestamp', 'datetime']):
                if col not in [c[0] for c in datetime_candidates]:
                    datetime_candidates.append((col, 'name_hint'))

        if datetime_candidates:
            # Prioritize explicit datetime types
            explicit = [c for c in datetime_candidates if c[1] == 'explicit']
            if explicit:
                if self.verbose:
                    print(f"Auto-detected datetime column: '{explicit[0][0]}'")
                return explicit[0][0]

            # Otherwise return first candidate
            if self.verbose:
                print(
                    f"Auto-detected datetime column: '{datetime_candidates[0][0]}' (type: {datetime_candidates[0][1]})")
            return datetime_candidates[0][0]

        if self.verbose:
            print("Warning: No datetime column detected")
        return None

    def _preprocess_signal(self, series: np.ndarray) -> np.ndarray:
        """
        Preprocess time series signal for DFT analysis

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
        series = pd.Series(series).fillna(method='ffill').fillna(method='bfill').fillna(0).values

        # Remove linear trend
        x = np.arange(len(series))
        if len(series) > 1:
            coeffs = np.polyfit(x, series, 1)
            trend = np.polyval(coeffs, x)
            detrended = series - trend
        else:
            detrended = series - np.mean(series)

        # Apply Hamming window to reduce spectral leakage
        window = np.hamming(len(detrended))
        windowed_signal = detrended * window

        return windowed_signal

    def _compute_dft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute DFT and extract positive frequencies

        Parameters:
        -----------
        signal : ndarray
            Preprocessed time series signal

        Returns:
        --------
        frequencies : ndarray
            Positive frequency values
        magnitudes : ndarray
            Magnitude spectrum for positive frequencies
        periods : ndarray
            Period values corresponding to frequencies (in samples)
        """
        # Compute FFT
        fft_vals = fft(signal)
        freqs = fftfreq(len(signal))
        magnitude = np.abs(fft_vals)

        # Consider only positive frequencies
        pos_idx = freqs > 0
        freq_pos = freqs[pos_idx]
        mag_pos = magnitude[pos_idx]

        # Convert frequencies to periods (in samples)
        # Avoid division by zero
        periods = np.where(freq_pos > 0, 1.0 / freq_pos, 0)

        return freq_pos, mag_pos, periods

    def _compute_confidence(self, magnitudes: np.ndarray) -> float:
        """
        Compute confidence score for periodicity detection

        Parameters:
        -----------
        magnitudes : ndarray
            Magnitude spectrum

        Returns:
        --------
        float
            Confidence score between 0 and 1, where 1 indicates strong periodicity
            (computed as 1 - ratio of second peak to dominant peak)
        """
        if len(magnitudes) < 2:
            return 0.0

        sorted_mags = np.sort(magnitudes)[::-1]

        # Avoid division by zero
        if sorted_mags[0] > 0:
            confidence = 1.0 - (sorted_mags[1] / sorted_mags[0])
        else:
            confidence = 0.0

        # Ensure confidence is in [0, 1]
        confidence = np.clip(confidence, 0.0, 1.0)

        return confidence

    def detect_optimal_windows(self,
                               df: pd.DataFrame,
                               datetime_col: str,
                               feature_cols: List[str],
                               sampling_rate: str = 'D') -> Tuple[List[pd.Timedelta], Dict[str, float]]:
        """
        Detect optimal window sizes using DFT for multiple features

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
            Confidence score for each feature
        """
        if datetime_col not in df.columns:
            raise ValueError(f"Datetime column '{datetime_col}' not found in DataFrame")

        # Sort by datetime
        df_sorted = df.sort_values(datetime_col).reset_index(drop=True)

        detected_periods = []
        confidence_scores = {}

        for col in feature_cols:
            if col not in df_sorted.columns:
                warnings.warn(f"Feature column '{col}' not found, skipping")
                continue

            try:
                # Extract series
                series = df_sorted[col].values

                if len(series) < 10:
                    warnings.warn(f"Series '{col}' too short for DFT analysis, skipping")
                    continue

                # Preprocess signal
                processed_signal = self._preprocess_signal(series)

                # Compute DFT
                frequencies, magnitudes, periods = self._compute_dft(processed_signal)

                # Find dominant frequency
                dominant_idx = np.argmax(magnitudes)
                dominant_period = periods[dominant_idx]

                # Apply constraints (convert to days based on sampling rate)
                period_days = self._convert_to_days(dominant_period, sampling_rate)

                if period_days < self.min_window_days:
                    period_days = self.min_window_days
                elif period_days > self.max_window_days:
                    period_days = self.max_window_days

                detected_periods.append(period_days)

                # Compute confidence
                confidence = self._compute_confidence(magnitudes)
                confidence_scores[col] = confidence

                if self.verbose:
                    print(f"Feature '{col}': detected period = {period_days:.1f} days, confidence = {confidence:.2f}")

            except Exception as e:
                warnings.warn(f"DFT failed for feature '{col}': {str(e)}")
                detected_periods.append(30)  # Default to monthly
                confidence_scores[col] = 0.0

        if not detected_periods:
            warnings.warn("No valid periods detected, using default windows")
            return self._get_default_windows(), {}

        # Generate multi-scale windows
        window_sizes = self._generate_multiscale_windows(detected_periods)

        return window_sizes, confidence_scores

    def _convert_to_days(self, period_samples: float, sampling_rate: str) -> float:
        """
        Convert period in samples to days based on sampling rate

        Parameters:
        -----------
        period_samples : float
            Period in number of samples
        sampling_rate : str
            Sampling rate ('D', 'H', 'W', 'M', etc.)

        Returns:
        --------
        float
            Period in days
        """
        # Convert sampling rate to days per sample
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
        """
        Generate multi-scale windows from detected periods

        Parameters:
        -----------
        detected_periods : list
            List of detected periods in days

        Returns:
        --------
        list of pd.Timedelta
            Multi-scale window sizes
        """
        unique_periods = list(set([int(p) for p in detected_periods if p > 0]))

        # Add harmonics and sub-harmonics
        multi_scale_periods = []
        for period in unique_periods:
            if period >= 4:  # Only add sub-harmonic if period is large enough
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

        # Select top n_windows by even spacing
        if len(multi_scale_periods) > self.n_windows:
            indices = np.linspace(0, len(multi_scale_periods) - 1,
                                  self.n_windows, dtype=int)
            selected_periods = [multi_scale_periods[i] for i in indices]
        else:
            selected_periods = multi_scale_periods

        # Ensure we have at least some windows
        if not selected_periods:
            selected_periods = [self.min_window_days]

        # Convert to Timedelta
        window_sizes = [pd.Timedelta(days=int(p)) for p in selected_periods]

        if self.verbose:
            print(f"\nGenerated {len(window_sizes)} window sizes: {[w.days for w in window_sizes]} days")

        return window_sizes

    def _get_default_windows(self) -> List[pd.Timedelta]:
        """
        Return default window sizes when DFT fails

        Returns:
        --------
        list of pd.Timedelta
            Default window sizes
        """
        default_days = [7, 14, 30, 90, 180, 365]
        default_days = [d for d in default_days
                        if self.min_window_days <= d <= self.max_window_days]
        return [pd.Timedelta(days=d) for d in default_days[:self.n_windows]]

    def assess_periodicity(self,
                           df: pd.DataFrame,
                           datetime_col: str,
                           feature_cols: List[str]) -> Tuple[bool, float, Dict[str, float]]:
        """
        Assess if time series data exhibits strong periodicity

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
        _, confidence_scores = self.detect_optimal_windows(
            df, datetime_col, feature_cols
        )

        if not confidence_scores:
            return False, 0.0, {}

        avg_confidence = np.mean(list(confidence_scores.values()))
        is_periodic = avg_confidence >= self.confidence_threshold

        if self.verbose:
            print(f"\nPeriodicity Assessment:")
            print(f"  Average confidence: {avg_confidence:.2f}")
            print(f"  Threshold: {self.confidence_threshold}")
            print(f"  Result: {'PERIODIC' if is_periodic else 'NON-PERIODIC'}")

        return is_periodic, avg_confidence, confidence_scores

    def smart_window_selection(self,
                               df: pd.DataFrame,
                               datetime_col: str,
                               feature_cols: List[str]) -> Tuple[List[pd.Timedelta], str]:
        """
        Smart window selection with hybrid strategy

        Returns DFT-detected windows for periodic data,
        standard windows for non-periodic data,
        or hybrid for moderate periodicity

        Parameters:
        -----------
        df : DataFrame
            Time series data
        datetime_col : str
            Name of datetime column
        feature_cols : list
            List of feature columns

        Returns:
        --------
        window_sizes : list of pd.Timedelta
            Selected window sizes
        strategy : str
            Strategy used ('dft', 'hybrid', 'standard')
        """
        dft_windows, confidence_scores = self.detect_optimal_windows(
            df, datetime_col, feature_cols
        )

        if not confidence_scores:
            if self.verbose:
                print("Using standard windows (no valid DFT detection)")
            return self._get_default_windows(), 'standard'

        avg_confidence = np.mean(list(confidence_scores.values()))

        if avg_confidence >= self.confidence_threshold:
            if self.verbose:
                print(f"Using DFT-detected windows (strong periodicity, confidence={avg_confidence:.2f})")
            return dft_windows, 'dft'

        elif avg_confidence > self.confidence_threshold * 0.6:  # 0.5 * 0.6 = 0.3
            # Hybrid approach: combine DFT with standard windows
            if self.verbose:
                print(f"Using hybrid windows (moderate periodicity, confidence={avg_confidence:.2f})")

            standard_windows = self._get_default_windows()
            combined = list(set(dft_windows + standard_windows))
            combined_sorted = sorted(combined, key=lambda x: x.days)[:self.n_windows]

            return combined_sorted, 'hybrid'

        else:
            if self.verbose:
                print(f"Using standard windows (weak periodicity, confidence={avg_confidence:.2f})")
            return self._get_default_windows(), 'standard'