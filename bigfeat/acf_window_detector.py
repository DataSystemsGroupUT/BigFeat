import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import List, Tuple, Dict, Optional
import warnings

from bigfeat.window_detector_base import BaseWindowDetector


class ACFWindowDetector(BaseWindowDetector):
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

    STRATEGY_NAME = 'acf'

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
        super().__init__(min_window_days=min_window_days,
                         max_window_days=max_window_days,
                         n_windows=n_windows,
                         confidence_threshold=confidence_threshold,
                         verbose=verbose)
        self.min_peak_height = min_peak_height
        self.min_peak_prominence = min_peak_prominence


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

        # Require a minimum overlap between the two shifted copies.
        #
        # Without this the lag could run up to len(series) - 1, leaving as few
        # as one or two overlapping points. np.corrcoef of two points is
        # ALWAYS exactly +/-1, so white noise produced |ACF| = 1.0 at long
        # lags -- spurious peaks that sorted to the top by height and became
        # both the reported period and the confidence. Measured on 365 samples
        # of white noise: max |ACF| was 0.158 for lags <= 180 (correct) but
        # 1.000 for lags > 300, and the detector reported periodic=True with
        # confidence 0.973 and windows of 163-363 days.
        #
        # MIN_OVERLAP samples is the usual rule of thumb for a meaningful
        # correlation estimate; it also bounds max_lag to roughly n/2, which
        # is the standard advice for ACF-based period detection.
        MIN_OVERLAP = 30
        usable_lag = max(1, len(series) - MIN_OVERLAP)
        max_lag = min(max_lag, len(series) - 1, usable_lag, len(series) // 2)

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

        # Accept fundamentals in ASCENDING-lag order, not by height.
        #
        # For a signal with periods P and Q the ACF at common multiples
        # (k*P, k*Q, lcm) is HIGHER than at the fundamentals, because every
        # component realigns there: on a 7+30-day test signal ACF(210)=0.997
        # vs ACF(7)=0.652. Sorting by height therefore returned harmonics
        # (210, 91, 301) and missed both true periods. The fundamental is the
        # first peak clearing the significance floor, so we walk lags upward,
        # and after accepting a lag we mask its harmonic train so the next
        # acceptance is an independent period rather than an echo.
        #
        # Each candidate is VERIFIED at its multiples: a true period L echoes
        # at small multiples of L, an isolated noise spike does not. The echo
        # is required at ANY of 2L/3L, not all of them -- in a multi-period
        # signal the OTHER component can sit near anti-phase at exactly 2L
        # and cancel the echo there (planted {12, 52}: ACF(24) = 0.018
        # because cos(2*pi*24/52) ~ -0.97, while 3L = 36 shows 0.322).
        # Verification is skipped when even 2L exceeds the computed range --
        # nothing to test against -- which is safe because any multiple of an
        # already-accepted shorter period has been masked by then.
        max_lag = len(acf_values) - 1
        order = np.argsort(peak_lags)                 # ascending lag
        cand_lags = peak_lags[order].astype(int)
        cand_heights = peak_heights[order].astype(float)

        def _echo_at(mult_lag, min_height):
            lo = max(min_lag, int(np.floor(mult_lag * 0.85)))
            hi = min(max_lag, int(np.ceil(mult_lag * 1.15)))
            if hi <= lo:
                return False
            return float(np.nanmax(acf_values[lo:hi + 1])) >= min_height

        accepted, accepted_heights = [], []
        masked = np.zeros(len(cand_lags), dtype=bool)
        for i, (lag, height) in enumerate(zip(cand_lags, cand_heights)):
            if masked[i]:
                continue
            multiples_in_range = [m * lag for m in (2, 3) if m * lag <= max_lag]
            if multiples_in_range and not any(
                    _echo_at(m, height / 2) for m in multiples_in_range):
                continue                              # isolated spike: reject
            accepted.append(int(lag))
            accepted_heights.append(float(height))
            # mask this fundamental's harmonic train among later candidates
            tol = max(2.0, 0.15 * lag)
            for j in range(i + 1, len(cand_lags)):
                k = round(cand_lags[j] / lag)
                if k >= 2 and abs(cand_lags[j] - k * lag) <= tol:
                    masked[j] = True
            if len(accepted) >= 3:
                break

        peak_lags = np.asarray(accepted, dtype=int)
        peak_heights = np.asarray(accepted_heights, dtype=float)

        if len(peak_lags) == 0:
            return np.array([]), {'peak_heights': np.array([]), 'n_peaks': 0}

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
                               sampling_rate: str = 'D',
                               groupby_cols: Optional[List[str]] = None) -> Tuple[List[pd.Timedelta], Dict[str, float]]:
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

        # Handle grouped data
        if groupby_cols is not None and len(groupby_cols) > 0:
            try:
                 first_row = df.iloc[0]
                 mask = np.ones(len(df), dtype=bool)
                 for col in groupby_cols:
                     if col in df.columns:
                         mask &= (df[col] == first_row[col])
                 df = df[mask].copy()
                 if self.verbose:
                     print(f"ACF: Analying single time series (first group) with {len(df)} samples")
            except Exception as e:
                warnings.warn(f"Failed to extract single series for ACF: {e}. using full dataset.")

        # Sort by datetime
        df_sorted = df.sort_values(datetime_col).reset_index(drop=True)

        detected_periods = []

        _fundamental_pairs = []

        self.last_detected_periods = _fundamental_pairs
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

                # Convert the day-based window bounds into SAMPLE counts.
                #
                # max_window_days was previously used directly as max_lag, i.e.
                # a duration in days used as a count of samples. At hourly
                # sampling a 365-day ceiling became 365 hours (~15 days), so
                # the configured range was unreachable; at monthly sampling it
                # asked for 365 lags over a far longer span than intended.
                days_per_sample = self._convert_to_days(1, sampling_rate)
                if days_per_sample <= 0:
                    days_per_sample = 1.0
                max_lag_samples = max(1, int(round(self.max_window_days / days_per_sample)))
                min_lag_samples = max(1, int(round(self.min_window_days / days_per_sample)))

                max_lag = min(len(series) - 1, max_lag_samples)
                acf_values = self._compute_acf(processed_signal, max_lag)

                # Find peaks in ACF
                peak_lags, peak_metadata = self._find_acf_peaks(acf_values, min_lag=min_lag_samples)

                if len(peak_lags) == 0:
                    if self.verbose:
                        print(f"  '{col}': No significant peaks found")
                    continue

                # Convert lags to days based on sampling rate
                peak_periods = [self._convert_to_days(lag, sampling_rate) for lag in peak_lags]

                # Filter by bounds
                valid_periods = [p for p in peak_periods
                                 if self.min_window_days <= p <= self.max_window_days]

                if len(valid_periods) > 0:
                    # Take top 3 periods per feature
                    detected_periods.extend(valid_periods[:3])
                    for _vp, _vh in zip(valid_periods[:3],
                                        peak_metadata['peak_heights'][:3]):
                        _fundamental_pairs.append((float(_vp), float(_vh)))

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
            avg_conf = float(np.mean(list(confidence_scores.values()))) if confidence_scores else 0.0
            print(f"Average confidence: {avg_conf:.3f}")

        return window_sizes, confidence_scores





