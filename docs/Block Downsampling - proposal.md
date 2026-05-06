# **Technical Report: Implementation of Agnostic Block Sampling for Multi-Method Time Series Analysis**

**Module:** BigFeat.fit (Downsampling Logic) & Window Detectors (DFT, ACF, Lomb-Scargle)

## **1\. Executive Summary**

The BigFeat system has evolved to support multiple Time Series Window Detectors (DFT, ACF, and Lomb-Scargle). However, the existing "Random Downsampling" mechanism creates a critical failure point across *all* these methods by destroying temporal continuity.

This report details the transition to **Detector-Agnostic Contiguous Block Sampling**. This approach ensures that regardless of the detection method selected, the underlying data structure remains valid for both window detection and feature generation during the memory-constrained fit phase.

## **2\. Problem Description: The Universal Sampling Conflict**

All implemented time-series methods rely on the relationship between data points at specific time intervals (lags). Random sampling destroys these intervals.

### **2.1 Impact by Detector Type**

While the root cause (loss of continuity) is the same, the failure mode differs for each detector:

| Detector | Mechanism | Failure Mode under Random Sampling |
| :---- | :---- | :---- |
| **DFT** | Fourier Transform | **Aliasing & White Noise:** Randomly sampled data destroys the phase information. The spectrum becomes flat (white noise), causing the detector to miss clear periodic signals or return random high-frequency noise. |
| **ACF** | Autocorrelation | **Broken Lags:** ACF calculates correlation between $X\_t$ and $X\_{t-k}$. In random sampling, if $X\_t$ is preserved, $X\_{t-k}$ is likely missing. The correlation at any lag $k$ drops to near zero. |
| **Lomb-Scargle** | Least-Squares Spectral | **Nyquist Limit Violation:** While robust to *gaps*, LS requires sufficient local point density to fit sinusoids. Random sampling increases the effective sampling interval, making it impossible to detect high-frequency (short-period) patterns. |

### **2.2 Impact on Feature Engineering**

Regardless of the detector used, the **Feature Generators** (e.g., rolling\_mean, macd, rsi) require contiguous data. Applying these operators to a randomly sampled index results in mathematically invalid features, causing the Genetic Algorithm to reject them as "low importance."

### **2.3 Illustrative Example: The Moving Average Failure**

To visualize the mathematical error, consider a simple **3-Day Moving Average** feature on a sequential dataset.

Baseline (Full Data):  
In a correct calculation, the feature for Jan 5th averages the values of Jan 3, 4, and 5\.

| Date | Value | 3-Day Moving Avg Calculation | Result |
| :---- | :---- | :---- | :---- |
| Jan 3 | 15 | \- | \- |
| Jan 4 | 20 | \- | \- |
| **Jan 5** | **25** | $(15 + 20 + 25) / 3$ | **20.00** |

Scenario A: Random Sampling (Current Implementation)  
If random downsampling selects Jan 1, Jan 5, and Jan 8, the rolling window operator is forced to use non-consecutive neighbors.

| Index | Original Date | Value | 3-Day Moving Avg Calculation | Result | Status |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 0 | Jan 1 | 10 | \- | \- | \- |
| **1** | **Jan 5** | **25** | $(10 + 25 + Missing) / 2$ | **17.50** | **INVALID** |
| 2 | Jan 8 | 18 | ... | ... | ... |

* **Error:** The calculated value (17.50) differs from the true feature value (20.00).  
* **Consequence:** The Genetic Algorithm perceives this feature as noisy/uncorrelated and discards it, failing to discover the trend.

Scenario B: Block Sampling (Proposed)  
By sampling a contiguous block (Jan 3–Jan 6), we preserve the local relationships.

| Index | Original Date | Value | 3-Day Moving Avg Calculation | Result | Status |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 0 | Jan 3 | 15 | \- | \- | \- |
| 1 | Jan 4 | 20 | \- | \- | \- |
| **2** | **Jan 5** | **25** | $(15 + 20 + 25) / 3$ | **20.00** | **VALID** |

* **Result:** The feature value matches the baseline perfectly. The Genetic Algorithm can correctly evaluate its importance.

## **3\. Proposed Solution: Agnostic Block Sampling**

We propose a dynamic, detector-agnostic block sampling strategy. This logic queries the active detector for its specific constraints (max\_window\_days) and calculates a safe block size that guarantees valid feature generation.

### **3.1 The "Golden Rule" of Block Sizing**

To ensure valid feature calculation, the sampled block size ($S\_{block}$) must exceed the maximum potential window size ($W\_{max}$) by a safety margin.

$$S\_{block} \\ge W\_{max} \\times SafetyFactor$$

* **Safety Factor:** Fixed at **3.0**. (e.g., if checking a 30-day window, we sample 90-day blocks).  
* **Result:** The first $1/3$ of the block serves as the "warm-up" for the rolling window, leaving $2/3$ of the block as valid training data for the Genetic Algorithm.

### **3.2 Algorithm Logic**

The new \_calculate\_block\_params method dynamically adapts to the configuration:

1. **Resolve Max Window:** Check the active window\_detector instance.  
   * If using **ACF**, retrieve ACF-specific max window.  
   * If using **Lomb-Scargle**, retrieve LS-specific max window.  
   * Fallback to default config if no detector is active.  
2. **Calculate Minimum Viable Block:** $MinSize = MaxWindow \\times 3$.  
3. **Optimize Diversity:** Attempt to fit **10 distinct blocks** within the memory limit.  
4. **Constraint Handling:** If memory is too tight for 10 blocks, reduce the block count ($N$) to maintain the integrity of the Block Size ($S$).

## **4\. Implementation Specification**

### **4.1 New Helper Method**

Add to BigFeat class:
```python
def _calculate_block_params(self, total_limit, max_window_size=None):  
    """  
    Determines optimal block count and size, agnostic of the detector used.  
    """  
    # 1. dynamic parameter resolution  
    if max_window_size is None:  
        if hasattr(self, 'window_detector') and hasattr(self.window_detector, 'max_window_days'):  
            max_window_size = self.window_detector.max_window_days  
        elif hasattr(self, 'dft_max_window_days'):  
            max_window_size = self.dft_max_window_days  
        else:  
            max_window_size = 365 

    # 2. Safety Calculation  
    min_block_size = max_window_size * 3  
      
    # 3. Memory Constraint Logic  
    if total_limit < min_block_size:  
        min_block_size = int(max_window_size * 1.1) # Emergency fallback

    # 4. Diversity Optimization  
    ideal_n_blocks = 10  
    tentative_size = total_limit // ideal_n_blocks  
      
    if tentative_size >= min_block_size:  
        return ideal_n_blocks, tentative_size  
    else:  
        # Prioritize Size over Count  
        return max(1, total_limit // min_block_size), min_block_size
```
### **4.2 Integration into fit()**

The fit method will utilize this helper to generate indices:
```python
if len(X_features) > limit:  
    # ... verbose logging ...  
      
    # Calculate params using the agnostic helper  
    n_blocks, block_size = self._calculate_block_params(limit)  
      
    # Generate Block Indices  
    downsample_rng = np.random.RandomState(seed=self.downsampling_random_state)  
    max_start = len(X_features) - block_size  
    start_indices = downsample_rng.choice(max_start, n_blocks, replace=False)  
      
    # Flatten into a single index list  
    sample_indices = []  
    for start in start_indices:  
        sample_indices.extend(range(start, start + block_size))  
          
    # ... Apply selection ...
```

## **5\. Conclusion**

This architecture update decouples the sampling strategy from specific algorithms (like DFT). By enforcing **Contiguous Block Sampling**, we ensure:

1. **DFT** retains frequency coherence.  
2. **ACF** retains valid lag relationships.  
3. **Lomb-Scargle** retains sufficient local density.  
4. **Feature Generators** produce valid, non-NaN values for selection.

This enables BigFeat to act as a robust, memory-safe, and method-agnostic tool for automated time-series feature engineering.