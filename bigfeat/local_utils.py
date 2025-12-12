import numpy as np
import pandas as pd
import scipy.stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# Basic utility functions
def unary_cube(arr):
    """Original cube transformation with overflow protection"""
    try:
        result = np.power(np.clip(arr, -100, 100), 3)
        return np.clip(result, -1e10, 1e10)
    except:
        return arr


def unary_multinv(arr):
    """Safe multiplicative inverse"""
    try:
        # Avoid division by zero
        arr_safe = np.where(np.abs(arr) < 1e-10, 1e-10, arr)
        result = 1 / arr_safe
        return np.clip(result, -1e10, 1e10)
    except:
        return arr


def unary_sqrtabs(arr):
    """Square root of absolute value with sign preservation"""
    try:
        result = np.sqrt(np.abs(arr)) * np.sign(arr)
        return np.where(np.isfinite(result), result, 0)
    except:
        return arr


def unary_logabs(arr):
    """Safe logarithm of absolute value with sign preservation"""
    try:
        abs_arr = np.abs(arr)
        # Avoid log(0) by using a small positive number
        abs_arr = np.where(abs_arr < 1e-10, 1e-10, abs_arr)
        result = np.log(abs_arr) * np.sign(arr)
        return np.where(np.isfinite(result), result, 0)
    except:
        return arr


def convert_with_max(arr):
    """Convert array to float32 with overflow protection"""
    try:
        arr = np.asarray(arr, dtype=float)
        arr[arr > np.finfo(np.dtype('float32')).max] = np.finfo(np.dtype('float32')).max
        arr[arr < np.finfo(np.dtype('float32')).min] = np.finfo(np.dtype('float32')).min
        return np.float32(arr)
    except:
        return np.float32(arr)


def mode(ar1):
    """Safe mode calculation"""
    try:
        if len(ar1) == 0:
            return 0
        mode_result = scipy.stats.mode(ar1, keepdims=True)
        return float(mode_result.mode[0])
    except:
        return np.mean(ar1) if len(ar1) > 0 else 0


def ar_range(ar1):
    """Safe range calculation"""
    try:
        if len(ar1) == 0:
            return 0
        return float(ar1.max() - ar1.min())
    except:
        return 0


def percentile_25(ar1):
    """Safe 25th percentile"""
    try:
        if len(ar1) == 0:
            return 0
        return float(np.percentile(ar1, 25))
    except:
        return np.median(ar1) if len(ar1) > 0 else 0


def percentile_75(ar1):
    """Safe 75th percentile"""
    try:
        if len(ar1) == 0:
            return 0
        return float(np.percentile(ar1, 75))
    except:
        return np.median(ar1) if len(ar1) > 0 else 0


def group_by(ar1, ar2):
    """Enhanced group by operation with error handling"""
    try:
        group_by_ops = [np.mean, np.std, np.max, np.min, np.sum, mode, len, ar_range, np.median, percentile_25,
                        percentile_75]
        group_by_op = np.random.choice(group_by_ops)
        temp_df = pd.DataFrame({'ar1': ar1, 'ar2': ar2})
        group_res = temp_df.groupby(['ar1'])['ar2'].apply(group_by_op).to_dict()
        result = temp_df['ar1'].map(group_res).values
        return np.where(np.isfinite(result), result, np.mean(ar2))
    except:
        return ar2  # Fallback to original array


def original_feat(ar1):
    """Return original feature"""
    return ar1


# Enhanced Time Series Utility Functions
def safe_rolling_operation(arr, window, operation, fill_method='bfill', **kwargs):
    """Safe wrapper for rolling operations"""
    try:
        if len(arr) == 0:
            return arr

        series = pd.Series(arr)
        window = min(window, len(arr))  # Ensure window doesn't exceed data length

        if operation == 'mean':
            result = series.rolling(window=window, min_periods=1).mean()
        elif operation == 'std':
            result = series.rolling(window=window, min_periods=1).std()
        elif operation == 'min':
            result = series.rolling(window=window, min_periods=1).min()
        elif operation == 'max':
            result = series.rolling(window=window, min_periods=1).max()
        elif operation == 'median':
            result = series.rolling(window=window, min_periods=1).median()
        elif operation == 'sum':
            result = series.rolling(window=window, min_periods=1).sum()
        elif operation == 'skew':
            min_periods = min(window, 3)
            result = series.rolling(window=window, min_periods=min_periods).skew()
        elif operation == 'kurt':
            min_periods = min(window, 4)
            result = series.rolling(window=window, min_periods=min_periods).kurt()
        elif operation == 'quantile':
            q = kwargs.get('quantile', 0.5)
            result = series.rolling(window=window, min_periods=1).quantile(q)
        else:
            return arr

        # Handle filling
        if fill_method == 'bfill':
            result = result.bfill()
        elif fill_method == 'ffill':
            result = result.ffill()
        else:
            result = result.fillna(0)

        # Ensure finite values
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
        return result.values

    except Exception as e:
        # Return original array on error
        return arr


def exponential_moving_average(arr, alpha=0.3):
    """Enhanced exponential moving average with validation"""
    try:
        if len(arr) == 0:
            return arr

        alpha = np.clip(alpha, 0.01, 0.99)  # Ensure valid alpha
        series = pd.Series(arr)
        result = series.ewm(alpha=alpha, adjust=False).mean()
        return result.bfill().values

    except Exception as e:
        return arr


def bollinger_bands_upper(arr, window=20, num_std=2):
    """Enhanced Bollinger Bands upper band with validation"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.fillna(0)
        upper_band = rolling_mean + (rolling_std * num_std)
        return upper_band.bfill().values

    except Exception as e:
        return arr


def bollinger_bands_lower(arr, window=20, num_std=2):
    """Enhanced Bollinger Bands lower band with validation"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.fillna(0)
        lower_band = rolling_mean - (rolling_std * num_std)
        return lower_band.bfill().values

    except Exception as e:
        return arr


def rsi(arr, window=14):
    """Enhanced Relative Strength Index with validation"""
    try:
        if len(arr) <= 1:
            return np.full_like(arr, 50.0)

        window = min(window, len(arr))
        series = pd.Series(arr)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

        # Avoid division by zero
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        rsi_values = 100 - (100 / (1 + rs))

        return rsi_values.fillna(50).replace([np.inf, -np.inf], 50).values

    except Exception as e:
        return np.full_like(arr, 50.0)


def macd(arr, fast=12, slow=26, signal=9):
    """Enhanced MACD with validation"""
    try:
        if len(arr) == 0:
            return arr

        series = pd.Series(arr)
        ema_fast = series.ewm(span=min(fast, len(arr))).mean()
        ema_slow = series.ewm(span=min(slow, len(arr))).mean()
        macd_line = ema_fast - ema_slow
        return macd_line.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)


def stochastic_oscillator(arr, window=14):
    """Enhanced Stochastic Oscillator with validation"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)
        rolling_min = series.rolling(window=window, min_periods=1).min()
        rolling_max = series.rolling(window=window, min_periods=1).max()

        # Avoid division by zero
        denominator = rolling_max - rolling_min
        denominator = denominator.replace(0, 1e-10)

        stoch_k = 100 * ((series - rolling_min) / denominator)
        return stoch_k.fillna(50).replace([np.inf, -np.inf], 50).values

    except Exception as e:
        return np.full_like(arr, 50.0)


def williams_r(arr, window=14):
    """Enhanced Williams %R with validation"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)
        rolling_max = series.rolling(window=window, min_periods=1).max()
        rolling_min = series.rolling(window=window, min_periods=1).min()

        # Avoid division by zero
        denominator = rolling_max - rolling_min
        denominator = denominator.replace(0, 1e-10)

        williams = -100 * ((rolling_max - series) / denominator)
        return williams.fillna(-50).replace([np.inf, -np.inf], -50).values

    except Exception as e:
        return np.full_like(arr, -50.0)


def momentum(arr, period=10):
    """Enhanced momentum calculation with validation"""
    try:
        if len(arr) == 0:
            return arr

        period = min(period, len(arr))
        series = pd.Series(arr)
        momentum_val = series.diff(period)
        return momentum_val.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)


def rate_of_change(arr, period=10):
    """Enhanced rate of change with validation"""
    try:
        if len(arr) == 0:
            return arr

        period = min(period, len(arr))
        series = pd.Series(arr)
        roc = series.pct_change(period) * 100
        return roc.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)


def commodity_channel_index(arr, window=20):
    """Enhanced Commodity Channel Index with validation"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)
        typical_price = series
        sma = typical_price.rolling(window=window, min_periods=1).mean()
        mean_deviation = typical_price.rolling(window=window, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - x.mean())) if len(x) > 0 else 1e-10
        )

        # Avoid division by zero
        mean_deviation = mean_deviation.replace(0, 1e-10)
        cci = (typical_price - sma) / (0.015 * mean_deviation)

        return cci.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)


def safe_aroon_calculation(arr, window, direction='up'):
    """Safe Aroon calculation helper"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)
        aroon_vals = np.zeros(len(series))

        for i in range(len(series)):
            if i < window:
                period_data = series[:i + 1]
                if direction == 'up':
                    periods_since_extreme = len(period_data) - 1 - period_data.idxmax()
                else:
                    periods_since_extreme = len(period_data) - 1 - period_data.idxmin()
                aroon_vals[i] = 100 * (len(period_data) - periods_since_extreme) / len(period_data)
            else:
                period_data = series[i - window + 1:i + 1]
                if direction == 'up':
                    periods_since_extreme = len(period_data) - 1 - (period_data.idxmax() - (i - window + 1))
                else:
                    periods_since_extreme = len(period_data) - 1 - (period_data.idxmin() - (i - window + 1))
                aroon_vals[i] = 100 * (window - periods_since_extreme) / window

        return np.clip(aroon_vals, 0, 100)

    except Exception as e:
        return np.full_like(arr, 50.0)


def aroon_up(arr, window=25):
    """Enhanced Aroon Up with validation"""
    return safe_aroon_calculation(arr, window, 'up')


def aroon_down(arr, window=25):
    """Enhanced Aroon Down with validation"""
    return safe_aroon_calculation(arr, window, 'down')


def average_true_range(arr, window=14):
    """Enhanced Average True Range with validation"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)

        # Simplified ATR using rolling high-low
        high_low = series.rolling(window=2, min_periods=1).max() - series.rolling(window=2, min_periods=1).min()
        true_range = high_low.fillna(0)
        atr = true_range.rolling(window=window, min_periods=1).mean()

        return atr.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)


def parabolic_sar(arr, af_step=0.02, af_max=0.2):
    """Enhanced Parabolic SAR with validation"""
    try:
        if len(arr) <= 1:
            return arr.copy()

        series = pd.Series(arr)
        psar = np.zeros(len(series))
        psar[0] = series.iloc[0]

        trend = 1  # 1 for uptrend, -1 for downtrend
        af = af_step
        ep = series.iloc[0]  # extreme point

        for i in range(1, len(series)):
            psar[i] = psar[i - 1] + af * (ep - psar[i - 1])

            if trend == 1:  # uptrend
                if series.iloc[i] > ep:
                    ep = series.iloc[i]
                    af = min(af + af_step, af_max)
                if series.iloc[i] < psar[i]:
                    trend = -1
                    psar[i] = ep
                    af = af_step
                    ep = series.iloc[i]
            else:  # downtrend
                if series.iloc[i] < ep:
                    ep = series.iloc[i]
                    af = min(af + af_step, af_max)
                if series.iloc[i] > psar[i]:
                    trend = 1
                    psar[i] = ep
                    af = af_step
                    ep = series.iloc[i]

        return np.where(np.isfinite(psar), psar, series.values)

    except Exception as e:
        return arr


def safe_fibonacci_retracement(arr, window, level):
    """Safe Fibonacci retracement calculation"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)
        rolling_max = series.rolling(window=window, min_periods=1).max()
        rolling_min = series.rolling(window=window, min_periods=1).min()
        fib_level = rolling_max - level * (rolling_max - rolling_min)

        return fib_level.bfill().replace([np.inf, -np.inf], series.median()).values

    except Exception as e:
        return arr


def fibonacci_retracement_236(arr, window=50):
    """Enhanced 23.6% Fibonacci retracement with validation"""
    return safe_fibonacci_retracement(arr, window, 0.236)


def fibonacci_retracement_382(arr, window=50):
    """Enhanced 38.2% Fibonacci retracement with validation"""
    return safe_fibonacci_retracement(arr, window, 0.382)


def fibonacci_retracement_618(arr, window=50):
    """Enhanced 61.8% Fibonacci retracement with validation"""
    return safe_fibonacci_retracement(arr, window, 0.618)


# Additional advanced time series functions
def autocorrelation(arr, lag=1):
    """Calculate autocorrelation with specified lag"""
    try:
        if len(arr) <= lag:
            return np.zeros_like(arr)

        series = pd.Series(arr)
        autocorr = series.rolling(window=min(20, len(arr)), min_periods=lag + 1).apply(
            lambda x: np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else 0
        )

        return autocorr.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)


def rolling_entropy(arr, window=10):
    """Calculate rolling entropy"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)

        def entropy(x):
            if len(x) == 0:
                return 0
            try:
                # Discretize the data
                hist, _ = np.histogram(x, bins=min(10, len(x)), density=True)
                hist = hist[hist > 0]  # Remove zero entries
                return -np.sum(hist * np.log(hist))
            except:
                return 0

        rolling_ent = series.rolling(window=window, min_periods=1).apply(entropy)
        return rolling_ent.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)


def rolling_variance_ratio(arr, window=10):
    """Calculate rolling variance ratio (variance / mean)"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)
        rolling_var = series.rolling(window=window, min_periods=1).var()
        rolling_mean = series.rolling(window=window, min_periods=1).mean()

        # Avoid division by zero
        rolling_mean = rolling_mean.replace(0, 1e-10)
        var_ratio = rolling_var / np.abs(rolling_mean)

        return var_ratio.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)


def rolling_kurtosis_adjusted(arr, window=10):
    """Calculate rolling excess kurtosis (kurtosis - 3)"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)
        min_periods = min(window, 4)
        rolling_kurt = series.rolling(window=window, min_periods=min_periods).kurt()

        # Excess kurtosis (subtract 3 for normal distribution)
        excess_kurt = rolling_kurt - 3
        return excess_kurt.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)


def trend_strength(arr, window=10):
    """Calculate trend strength using linear regression slope"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)

        def calc_slope(x):
            if len(x) < 2:
                return 0
            try:
                y = np.array(x)
                x_vals = np.arange(len(y))
                slope = np.polyfit(x_vals, y, 1)[0]
                return slope
            except:
                return 0

        trend = series.rolling(window=window, min_periods=2).apply(calc_slope)
        return trend.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)


def mean_reversion_indicator(arr, window=20):
    """Calculate mean reversion indicator"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1e-10)

        # Distance from mean in standard deviations
        mean_reversion = (series - rolling_mean) / rolling_std
        return mean_reversion.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)


# Seasonal decomposition components
def seasonal_trend(arr, period=12):
    """Extract trend component using moving average"""
    try:
        if len(arr) == 0 or period >= len(arr):
            return arr

        series = pd.Series(arr)
        # Use centered moving average for trend extraction
        trend = series.rolling(window=period, center=True, min_periods=1).mean()
        return trend.bfill().ffill().values

    except Exception as e:
        return arr


def seasonal_residual(arr, period=12):
    """Extract residual component after removing trend"""
    try:
        if len(arr) == 0:
            return arr

        trend = seasonal_trend(arr, period)
        residual = arr - trend
        return residual

    except Exception as e:
        return arr


# Frequency domain features
def dominant_frequency(arr, sample_rate=1.0):
    """Find dominant frequency using FFT"""
    try:
        if len(arr) < 4:
            return np.zeros_like(arr)

        # Apply FFT
        fft_vals = np.fft.fft(arr - np.mean(arr))
        freqs = np.fft.fftfreq(len(arr), 1 / sample_rate)

        # Find dominant frequency
        magnitude = np.abs(fft_vals)
        dominant_freq_idx = np.argmax(magnitude[1:len(magnitude) // 2]) + 1
        dominant_freq = freqs[dominant_freq_idx]

        return np.full_like(arr, dominant_freq)

    except Exception as e:
        return np.zeros_like(arr)


def spectral_energy(arr, window=10):
    """Calculate rolling spectral energy"""
    try:
        if len(arr) == 0:
            return arr

        window = min(window, len(arr))
        series = pd.Series(arr)

        def calc_spectral_energy(x):
            if len(x) < 4:
                return 0
            try:
                fft_vals = np.fft.fft(x - np.mean(x))
                energy = np.sum(np.abs(fft_vals) ** 2)
                return energy
            except:
                return 0

        spectral_eng = series.rolling(window=window, min_periods=4).apply(calc_spectral_energy)
        return spectral_eng.fillna(0).replace([np.inf, -np.inf], 0).values

    except Exception as e:
        return np.zeros_like(arr)