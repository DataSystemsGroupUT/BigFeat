import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from typing import List, Tuple, Dict, Optional, Union
import warnings

from bigfeat.window_detector_base import BaseWindowDetector


class DFTWindowDetector(BaseWindowDetector):
    """
    Automated window size detection using Discrete Fourier Transform (DFT)
    for BigFeat time series feature engineering.
    """

    STRATEGY_NAME = 'dft'

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
        super().__init__(min_window_days=min_window_days,
                         max_window_days=max_window_days,
                         n_windows=n_windows,
                         confidence_threshold=confidence_threshold,
                         verbose=verbose)


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
        series = pd.Series(series).ffill().bfill().fillna(0).values

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

    def _top_spectral_periods(self, magnitudes: np.ndarray,
                              periods: np.ndarray, k: int = 3) -> list:
        """Periods (in samples) of the top-k spectral peaks above the noise floor.

        Replaces a single np.argmax. Height-sorting is CORRECT here, unlike in
        the lag domain (see acf_window_detector._find_acf_peaks): in the
        frequency domain a fundamental is STRONGER than its harmonics, whereas
        in the lag domain common multiples of several periods exceed the
        fundamentals. The floor of 3x the spectrum median mirrors the
        peak-to-background logic of _compute_confidence -- the median estimates
        the noise floor because only a handful of bins carry real signal.
        """
        if len(magnitudes) == 0 or len(periods) == 0:
            return []
        floor = 3.0 * float(np.median(magnitudes))
        peak_idx, _ = find_peaks(magnitudes, height=max(floor, 1e-12))
        if len(peak_idx) == 0:
            # No structured peak; fall back to the single dominant bin so the
            # caller's behaviour on borderline signals is unchanged.
            peak_idx = np.array([int(np.argmax(magnitudes))])

        # Deduplicate in PERIOD space before taking top-k. When the true
        # frequency falls between FFT bins, spectral leakage splits one peak
        # across adjacent bins -- on a 1000-sample series with an 11-day
        # period, the three tallest bins are all lobes of the SAME peak
        # (periods ~10.9/11.1/11.2), and a genuine second period never makes
        # the cut. Keep only the tallest bin within each +-15% period
        # neighbourhood, then take the k tallest distinct peaks.
        by_height = peak_idx[np.argsort(magnitudes[peak_idx])[::-1]]
        chosen = []
        for i in by_height:
            p_i = float(periods[i])
            if p_i <= 0:
                continue
            if any(abs(p_i - c) <= 0.15 * max(p_i, c) for c in chosen):
                continue
            chosen.append(p_i)
            if len(chosen) >= k:
                break
        return chosen

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

        magnitudes = np.asarray(magnitudes, dtype=float)
        peak = magnitudes.max()
        if not np.isfinite(peak) or peak <= 0:
            return 0.0

        # Peak-to-background ratio.
        #
        # This was previously `1 - sorted[1] / sorted[0]`, which is close to
        # the OPPOSITE of what it claims to measure. np.sort places the two
        # largest bins adjacent to each other, and for a genuinely periodic
        # signal those two are neighbouring frequency bins of the SAME peak,
        # split by spectral leakage. Their ratio is therefore near 1 for
        # strong periodicity, driving the confidence toward 0. Measured on a
        # clean 7-day sine wave the detector reported confidence 0.244 and
        # periodic=False, while on white noise it reported 0.075 -- barely
        # distinguishable, and both below the 0.3 threshold.
        #
        # Comparing the peak against the MEDIAN of the spectrum is the
        # standard approach: the median is robust to the handful of bins that
        # carry a real signal, so it estimates the noise floor. A pure sine
        # gives a very large ratio, white noise gives a ratio near 1.
        background = np.median(magnitudes)
        if background <= 0:
            # Degenerate spectrum (e.g. a constant series): no periodicity.
            return 0.0

        ratio = peak / background
        # Map the ratio onto [0, 1). ratio == 1 (no peak above background)
        # gives 0; large ratios saturate toward 1. The scale factor is chosen
        # so that a ratio of ~10 lands near the 0.75 mark.
        confidence = 1.0 - np.exp(-(ratio - 1.0) / 6.0)

        return float(np.clip(confidence, 0.0, 1.0))

    def detect_optimal_windows(self,
                               df: pd.DataFrame,
                               datetime_col: str,
                               feature_cols: List[str],
                               sampling_rate: str = 'D',
                               groupby_cols: Optional[List[str]] = None) -> Tuple[List[pd.Timedelta], Dict[str, float]]:
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

        if self.verbose:
             print(f"DEBUG DFT: Input shape: {df.shape}, Groupby: {groupby_cols}")
             if groupby_cols is not None and len(groupby_cols) > 0:
                  print(f"DEBUG DFT: Unique groups in {groupby_cols[0]}: {df[groupby_cols[0]].unique()}")

        # Handle grouped data
        if groupby_cols is not None and len(groupby_cols) > 0:
            # === FIX: Multi-Series Robust Sampling ===
            try:
                # 1. Get a sample of unique group IDs (e.g., first 5)
                # We assume the first column in groupby_cols is the main ID
                main_id_col = groupby_cols[0]
                unique_ids = df[main_id_col].unique()
                
                # Sample up to 5 groups to get a representative average
                n_samples = min(5, len(unique_ids))
                sampled_ids = unique_ids[:n_samples] 
                
                if self.verbose:
                    print(f"DFT: Sampling {n_samples} series to estimate periodicity...")

                # 2. Accumulate periods and confidence across samples
                all_detected_periods = []
                total_confidence_scores = {}
                
                for group_id in sampled_ids:
                    # Filter for this specific group
                    # (This is fast because we do it only 5 times)
                    mask = (df[main_id_col] == group_id)
                    single_series_df = df[mask]
                    
                    # Sort this small slice
                    single_sorted = single_series_df.sort_values(datetime_col)
                    
                    # Run Detection on this slice
                    for col in feature_cols:
                        if col not in single_sorted.columns: continue
                        
                        series = single_sorted[col].values
                        if len(series) < 10: continue

                        processed = self._preprocess_signal(series)
                        freqs, mags, periods = self._compute_dft(processed)
                        
                        # Get confidence
                        conf = self._compute_confidence(mags)
                        
                        # Accumulate
                        if col not in total_confidence_scores:
                            total_confidence_scores[col] = []
                        total_confidence_scores[col].append(conf)
                        
                        # Get top-k spectral periods (Fix 4: a single argmax
                        # returned only the strongest component, so a second
                        # genuine period was never proposed)
                        for period_val in self._top_spectral_periods(mags, periods):
                             period_days = self._convert_to_days(period_val, sampling_rate)

                             if period_days < self.min_window_days:
                                 period_days = self.min_window_days
                             elif period_days > self.max_window_days:
                                 period_days = self.max_window_days

                             all_detected_periods.append(period_days)

                # 3. Average the confidence scores
                final_confidence_scores = {
                    col: float(np.mean(scores)) for col, scores in total_confidence_scores.items()
                }
                
                # Use the collected periods
                if not all_detected_periods:
                    warnings.warn("No valid periods detected in samples, using default windows")
                    return self._get_default_windows(), {}
                
                # Generate multi-scale windows from ALL detected periods
                sanitized_periods = self._apply_seasonal_bias(all_detected_periods)
                window_sizes = self._generate_multiscale_windows(sanitized_periods)
                
                return window_sizes, final_confidence_scores

            except Exception as e:
                warnings.warn(f"Sampling failed: {e}. Falling back to default.")
                return self._get_default_windows(), {}

        # ---------------------------------------------------------
        # SINGLE SERIES LOGIC (Only reached if groupby_cols is None or empty)
        # ---------------------------------------------------------
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

                # Top-k spectral peaks (Fix 4; see _top_spectral_periods)
                first_period_days = None
                for period_val in self._top_spectral_periods(magnitudes, periods):
                    period_days = self._convert_to_days(period_val, sampling_rate)

                    if period_days < self.min_window_days:
                        period_days = self.min_window_days
                    elif period_days > self.max_window_days:
                        period_days = self.max_window_days

                    detected_periods.append(period_days)
                    if first_period_days is None:
                        first_period_days = period_days
                period_days = first_period_days if first_period_days is not None \
                    else self.min_window_days

                # Compute confidence
                confidence = self._compute_confidence(magnitudes)
                confidence_scores[col] = float(confidence)

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
        sanitized_periods = self._apply_seasonal_bias(detected_periods)
        window_sizes = self._generate_multiscale_windows(sanitized_periods)

        return window_sizes, confidence_scores




    COMMON_SEASONALITIES = [7, 14, 30, 90, 180, 365]

    def _apply_seasonal_bias(self, detected_days: List[float]) -> List[float]:
        """
        Shift detected windows toward common human-centric seasonalities if close.
        Prevents DFT from locking onto "hallucinated" windows near common periods.

        Parameters:
        -----------
        detected_days : list of float
            List of detected periods in days

        Returns:
        --------
        list of float
            Sanitized list with periods snapped to common seasonalities if applicable
        """
        sanitized = []
        for d in detected_days:
            snapped = False
            for common in self.COMMON_SEASONALITIES:
                # If within 10% of a common seasonality, snap to it
                if 0.9 <= (d / common) <= 1.1:
                    sanitized.append(common)
                    snapped = True
                    break
            if not snapped:
                sanitized.append(d)
        return sanitized


