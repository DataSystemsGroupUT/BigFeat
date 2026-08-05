"""Shared behaviour for the periodicity window detectors.

DFT, ACF and Lomb-Scargle each analyse a series to propose rolling-window
sizes. They were written as three independent classes with no common base,
which left roughly 145 lines of exact duplication and -- more dangerously --
several methods that shared a NAME while diverging in behaviour.

This module holds the parts that genuinely are common: locating the datetime
column, converting sample counts to days, expanding detected periods into a
multi-scale window ladder, the default window ladder, and the periodicity
verdict. Subclasses implement the transform-specific parts:

    _preprocess_signal(series)    -> conditioned signal
    detect_optimal_windows(...)   -> (windows, confidences)

and set STRATEGY_NAME.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# The window ladder used when detection fails or finds nothing.
DEFAULT_WINDOW_DAYS = [7, 14, 30, 90, 180, 365]

# Days spanned by one sample at a given pandas frequency alias.
#
# Both the legacy uppercase aliases and the modern lowercase ones are listed.
# The lookup used to be a bare dict.get(rate, 1.0), so ANY unrecognised code
# silently fell back to "one sample == one day". Passing pandas' current
# lowercase 'h' therefore inflated every detected hourly period by 24x, and
# an unhandled alias such as '5T' produced silently wrong windows with no
# warning. Unknown codes are now parsed by pandas, and only fall back if that
# fails as well.
_RATE_TO_DAYS = {
    'D': 1.0, 'd': 1.0,
    'H': 1 / 24.0, 'h': 1 / 24.0,
    'T': 1 / 1440.0, 'min': 1 / 1440.0,
    'S': 1 / 86400.0, 's': 1 / 86400.0,
    'W': 7.0, 'w': 7.0,
    'M': 30.0, 'ME': 30.0, 'MS': 30.0,
    'Q': 91.0, 'QE': 91.0, 'QS': 91.0,
    'Y': 365.0, 'YE': 365.0, 'YS': 365.0, 'A': 365.0,
}


class BaseWindowDetector:
    """Common interface and shared implementation for window detectors."""

    #: Identifies the detector in smart_window_selection's return value.
    STRATEGY_NAME = 'base'

    def __init__(self,
                 min_window_days: int = 3,
                 max_window_days: int = 365,
                 n_windows: int = 6,
                 confidence_threshold: float = 0.3,
                 verbose: bool = True):
        self.min_window_days = min_window_days
        self.max_window_days = max_window_days
        self.n_windows = n_windows
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

    # -- interface ---------------------------------------------------------

    def _preprocess_signal(self, series: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def detect_optimal_windows(self, df, datetime_col, feature_cols,
                               sampling_rate='D', groupby_cols=None):
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------

    def detect_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Return the most likely datetime column, or None.

        Prefers columns already of datetime dtype, then object columns that
        parse cleanly, then columns whose NAME hints at a timestamp.
        """
        if df is None or not hasattr(df, 'columns'):
            return None

        candidates = [c for c in df.columns
                      if pd.api.types.is_datetime64_any_dtype(df[c])]
        if candidates:
            return candidates[0]

        for col in df.columns:
            if df[col].dtype == object:
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                except (TypeError, ValueError):
                    continue
                if parsed.notna().mean() > 0.9:
                    return col

        name_hints = ('date', 'time', 'timestamp', 'datetime', 'dt')
        for col in df.columns:
            if any(hint in str(col).lower() for hint in name_hints):
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                except (TypeError, ValueError):
                    continue
                if parsed.notna().mean() > 0.9:
                    return col
        return None

    def _convert_to_days(self, period_samples: float, sampling_rate: str) -> float:
        """Convert a period expressed in SAMPLES into days."""
        days_per_sample = _RATE_TO_DAYS.get(sampling_rate)

        if days_per_sample is None:
            # Let pandas interpret anything not in the table (e.g. '15min').
            try:
                days_per_sample = (pd.tseries.frequencies.to_offset(sampling_rate)
                                   .nanos / 8.64e13)
            except Exception:
                if self.verbose:
                    print(f"  Warning: unrecognised sampling rate "
                          f"{sampling_rate!r}; assuming daily")
                days_per_sample = 1.0

        return period_samples * days_per_sample

    def _get_default_windows(self) -> List[pd.Timedelta]:
        """Window ladder used when detection finds nothing usable."""
        filtered = [pd.Timedelta(days=d) for d in DEFAULT_WINDOW_DAYS
                    if self.min_window_days <= d <= self.max_window_days]
        if not filtered:
            filtered = [pd.Timedelta(days=self.min_window_days)]
        return filtered[:self.n_windows]

    def _generate_multiscale_windows(self,
                                     detected_periods: List[float]
                                     ) -> List[pd.Timedelta]:
        """Expand detected periods into a ladder of related window sizes.

        For each period this adds a sub-harmonic (half) and harmonics (double,
        quadruple) where they fit inside the configured bounds, so that
        rolling features can capture both faster and slower structure.
        """
        if not detected_periods:
            return self._get_default_windows()

        candidates = set()
        for period in detected_periods:
            if period <= 0:
                continue
            candidates.add(period)
            if period >= 4:
                candidates.add(period / 2)
            if period <= self.max_window_days // 4:
                candidates.add(period * 2)
                candidates.add(period * 4)

        valid = sorted({int(round(p)) for p in candidates
                        if self.min_window_days <= p <= self.max_window_days})
        if not valid:
            return self._get_default_windows()

        return [pd.Timedelta(days=d) for d in valid[:self.n_windows]]

    def assess_periodicity(self, df, datetime_col, feature_cols,
                           groupby_cols=None) -> Tuple[bool, float, Dict]:
        """Decide whether the data is periodic enough to justify TS features.

        Returns (is_periodic, average_confidence, per_feature_confidences).

        A feature-level consensus is required in addition to average
        confidence: at least half the analysed features must individually
        clear the threshold. Previously only DFT applied this rule while ACF
        and Lomb-Scargle used the average alone, so one strongly periodic
        column among many noisy ones was enough to enable time-series
        features for the whole frame -- and the three classes disagreed while
        exposing an identically named method.
        """
        try:
            _, confidences = self.detect_optimal_windows(
                df, datetime_col, feature_cols, groupby_cols=groupby_cols)
        except Exception as exc:
            if self.verbose:
                print(f"  {self.STRATEGY_NAME}: periodicity assessment failed: {exc}")
            return False, 0.0, {}

        if not confidences:
            return False, 0.0, {}

        values = list(confidences.values())
        avg_confidence = float(np.mean(values))
        n_periodic = sum(1 for v in values if v >= self.confidence_threshold)
        ratio = n_periodic / len(values)

        is_periodic = bool(avg_confidence >= self.confidence_threshold
                           and ratio >= 0.5)

        if self.verbose:
            print(f"  {self.STRATEGY_NAME}: avg confidence {avg_confidence:.3f}, "
                  f"{n_periodic}/{len(values)} features periodic "
                  f"-> {'periodic' if is_periodic else 'not periodic'}")

        return is_periodic, avg_confidence, confidences

    def smart_window_selection(self, df, datetime_col, feature_cols,
                               groupby_cols=None) -> Tuple[List[pd.Timedelta], str]:
        """Return (windows, strategy) for the given data."""
        is_periodic, confidence, _ = self.assess_periodicity(
            df, datetime_col, feature_cols, groupby_cols)

        if is_periodic:
            windows, _ = self.detect_optimal_windows(
                df, datetime_col, feature_cols, groupby_cols=groupby_cols)
            return windows, self.STRATEGY_NAME

        if confidence >= self.confidence_threshold * 0.6:
            # Weak but non-zero signal: blend detected windows with defaults.
            try:
                detected, _ = self.detect_optimal_windows(
                    df, datetime_col, feature_cols, groupby_cols=groupby_cols)
            except Exception:
                detected = []
            merged = sorted({*detected, *self._get_default_windows()})
            return merged[:self.n_windows], 'hybrid'

        return self._get_default_windows(), 'standard'
