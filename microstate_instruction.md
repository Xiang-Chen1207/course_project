I need you to implement a Python pipeline for EEG Microstate Analysis based on the methodology described in the paper "Fine-Tuning Large Language Models Using EEG Microstate Features for Mental Workload Assessment".

The goal is to extract 5 specific features for 4 microstate classes (A, B, C, D) from raw EEG signals.

Please write a Python class `MicrostateFeatureExtractor` using `numpy` and `scipy`. 

### Input Data
The input to the pipeline will be:
1. `data`: A 2D numpy array of shape `(n_channels, n_samples)`.
2. `sfreq`: Sampling frequency (float), e.g., 250 Hz or 1000 Hz.

### Algorithm Steps

**Step 1: Global Field Power (GFP) Calculation**
Calculate the GFP for every time point $t$.
Formula: Standard deviation of all electrodes at time $t$.
$$ GFP(t) = \sqrt{\frac{\sum_{i=1}^{N} (V_i(t) - \bar{V}(t))^2}{N}} $$
Where $V_i(t)$ is voltage at electrode $i$, and $\bar{V}(t)$ is the mean voltage across all electrodes at $t$.

**Step 2: Template Generation (Segmentation)**
1. Extract topographies (maps) only at **GFP Peaks** (local maxima of the GFP curve) to improve signal-to-noise ratio.
2. Perform **Modified K-Means Clustering** on these peak maps to find $K=4$ centroids (Microstate templates).
   * **Crucial:** The clustering must be **polarity invariant**. Map $X$ and map $-X$ are considered identical.
   * Distance metric: $1 - |SpatialCorrelation(X, Y)|$.
   * Or simply ensure that when updating centroids, you align the polarity of the maps to the centroid before averaging.

**Step 3: Backfitting (Winner-Takes-All)**
Assign *every* time point in the original `data` (not just peaks) to one of the 4 centroids.
1. For each time point $t$, calculate the spatial correlation (Pearson correlation) between the instantaneous map $Data_t$ and all 4 centroids.
2. Use absolute correlation values (ignore polarity).
3. Assign label $L_t \in \{0, 1, 2, 3\}$ corresponding to the centroid with the highest absolute correlation.
   $$ Label_t = \operatorname{argmax}_k (|Corr(Data_t, Centroid_k)|) $$

**Step 4: Feature Extraction**
Based on the sequence of labels (e.g., `0, 0, 0, 1, 1, 2...`) and the correlation values, calculate the following 5 features for EACH microstate class (k=0..3):

1. **Mean Duration (meandurs):**
   - Calculate the average duration of continuous segments for this state.
   - Formula: $\frac{\text{Total time in state } k}{\text{Number of distinct segments of state } k}$
   - Unit: Seconds (use `sfreq` to convert samples to seconds).

2. **Occurrence per second (occurrence):**
   - Formula: $\frac{\text{Number of distinct segments of state } k}{\text{Total recording time in seconds}}$
   - Unit: Hz (times per second).

3. **Time Coverage (timecov):**
   - Formula: $\frac{\text{Total samples assigned to state } k}{\text{Total samples in recording}}$
   - Unit: Ratio (0.0 to 1.0) or Seconds (Paper uses seconds, please provide Seconds).

4. **Mean Correlation (mean_corr):**
   - The average spatial correlation between the data and the template for all time points assigned to this state.
   - Formula: Average of $|Corr(Data_t, Centroid_k)|$ for all $t$ where $Label_t = k$.

5. **Global Explained Variance (gev):**
   - Represents the explained variance weighted by GFP.
   - Formula:
     $$ GEV_k = \frac{\sum_{t \in k} (GFP(t) \cdot Corr(Data_t, Centroid_k))^2}{\sum_{all\_t} (GFP(t))^2} $$
   - Total GEV is the sum of GEV for all 4 states.

### Output
{
  "Microstate_0": {"gev": ..., "mean_corr": ..., "timecov": ..., "meandurs": ..., "occurrence": ...},
  "Microstate_1": { ... },
  ...
}