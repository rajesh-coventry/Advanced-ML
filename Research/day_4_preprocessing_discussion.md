# Phase A, Day 4: Preprocessing, Target Definition, and Split

This document details the final stage of the data preparation pipeline for the NEPSE market regime detection assignment. Following the computation of 10 engineered features in Days 2–3, Day 4 focuses on cleaning the data, defining the regression target and classification labels, splitting the data temporally, and scaling the features.

## 1. Dropping NaN Rows

**Action:** We dropped the first 55 rows from the combined dataset.
**Result:** The dataset was reduced from 6,033 to 5,978 rows, with absolutely zero NaNs remaining.

**Rationale:**
Many of our engineered features rely on rolling lookback windows. The longest window required was 50 days (for the MA-50 component of `normalized_trend_strength`), meaning the first 50 rows of the dataset inherently lacked sufficient history to compute a valid signal. By explicitly dropping the first 55 rows, we ensure that every single sample in the final dataset contains a complete, valid vector of features.

## 2. Defining the Target Variable (GPR)

**Action:** We defined the regression target variable $y_{\text{regression}}$ as `rolling_vol_20`.

**Rationale:**
The 20-day rolling standard deviation of log returns is the classic empirical estimator for realized volatility. It satisfies the strict requirements for Gaussian Process Regression:
1. **Continuous & Smooth:** It provides a smooth response surface that a GP kernel can effectively model, unlike the highly noisy daily absolute returns.
2. **Stationary (Mean-Reverting):** It oscillates around a long-term mean and is strictly non-negative, avoiding the non-stationarity of raw price levels.
3. **Interpretability:** Its scale is identical to the standard deviation of returns.

*Reference figure generated:* `day4_target_gpr_volatility.png` plots this variable over the 25-year history, visually confirming that it successfully captures known market turbulence, such as the initial 2008 crash and the 2020 COVID-19 panic.

## 3. Defining Volatility Regime Labels (GPC)

**Action:** We created a categorical target variable $y_{\text{regime}}$ by applying strict percentile thresholds to the regression target (`rolling_vol_20`).

**Thresholds Used:**
- **Quiet (Class 0):** $\le$ 50th percentile
- **Normal (Class 1):** $>$ 50th percentile and $\le$ 85th percentile
- **Turbulent (Class 2):** $>$ 85th percentile

**Class Distribution:**
- **Quiet (0):** 2,989 rows (50.0%)
- **Normal (1):** 2,092 rows (35.0%)
- **Turbulent (2):** 897 rows (15.0%)

**Why Threshold on the Target?**
The coursework assignment specifically mandates that the classification regimes must be defined using the output target variable. We must not use inputs (like trend indicators or volume spikes) to define the ground-truth classes, as that would introduce circular logic into the classifier. Percentile thresholds provide an adaptive, mathematically rigorous way to define regimes regardless of absolute market scaling over the 25-year period.

**Addressing Class Imbalance:**
The resulting distribution (50% / 35% / 15%) is deliberately imbalanced. Financial markets are mostly quiet, occasionally normal, and rarely turbulent. We document this imbalance clearly here because it will influence the choice of evaluation metrics for our GPC models in Phase B (e.g., relying on Macro-F1 rather than raw Accuracy).

*Reference figures generated:*
- `day4_regime_distribution.png`: Bar chart of the class imbalance.
- `day4_volatility_regimes_scatter.png`: Shows the target variable mapped onto the three color-coded regimes.

## 4. Feature Matrix, Temporal Split, and Scaling

**Action:**
- Constructed input matrix `X` using the 9 independent engineered features. (The 10th, `rolling_vol_20`, was removed from `X` to prevent target leakage).
- Split the data **temporally** into 80% Training and 20% Test sets.
- Scaled `X` using `StandardScaler`, fitted **only** on the training set.

**Temporal Splitting Rationale:**
In time-series machine learning, randomly shuffling rows (e.g., via `train_test_split(..., shuffle=True)`) is a critical error. It leaks future information into past predictions. We preserve chronological order:
- **Train Period:** March 2000 to December 2020 (4,782 days)
- **Test Period:** December 2020 to May 2026 (1,196 days)

This strictly simulates the real-world deployment scenario where the model must generalize to a future, unseen timeline.

**Scaling Rationale:**
Gaussian Process kernels (such as RBF or Matern) compute Euclidean distances between feature vectors. If features exist on wildly different scales (e.g., RSI is 0-100, log return is 0-0.1), the large-scale features will artificially dominate the kernel distance metric. `StandardScaler` standardizes all features to mean=0, std=1.
Crucially, the scaler was `.fit()` *only* on the training data. Applying `.fit()` to the entire dataset would leak the global mean and variance of the future test set back into the training process.

## Output Payload

The preprocessed pipeline yields a strictly formatted `.npz` file containing:
- `X_train`, `X_test`: (Scaled, chronological input matrices)
- `y_train_reg`, `y_test_reg`: (Continuous targets for GPR)
- `y_train_clf`, `y_test_clf`: (Categorical labels for GPC)
- `train_dates`, `test_dates`: (Timestamps for plotting)

**Data Path:** `Research/data_splits/day_4_processed_data.npz`

This concludes Phase A (Data Pipeline). The data is now rigorously sanitized and optimally formatted for Phase B: Gaussian Process Modeling.
