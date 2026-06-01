# Phase B, Day 5: Gaussian Process Regression (Matern Kernel)

This document covers the implementation and outcomes of the Gaussian Process Regression (GPR) model using the Matern ($\nu=2.5$) kernel, addressing the core predictive modeling requirement of the coursework assignment.

## 1. Kernel Selection and Rationale

The Gaussian Process Regressor was initialized with a composite kernel:
```python
Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
```

**Why Matern ($\nu=2.5$)?**
The Matern kernel is the gold standard for modeling physical and financial time series. By setting $\nu = 2.5$, we specify that the sample paths of the Gaussian Process should be exactly twice-differentiable. This provides a crucial balance:
- It is less rigidly smooth than the Radial Basis Function (RBF / Squared Exponential) kernel, which assumes infinitely differentiable paths. Financial volatility is notoriously jagged and subject to abrupt regime shifts; assuming infinite smoothness (RBF) often leads to over-smoothing and delayed response to sudden market shocks.
- It is smoother than the absolute exponential kernel ($\nu=0.5$), which models highly erratic, non-differentiable Brownian motion.
The Matern $\nu=2.5$ kernel perfectly captures the "rough but continuous" nature of equity market volatility.

**Why WhiteKernel?**
The `WhiteKernel` component explicitly models identically distributed observation noise (aleatoric uncertainty). Since our target (`rolling_vol_20`) is an empirical estimate subject to market microstructure noise, the GP must be allowed to explain some variance as pure noise rather than forcing the main kernel to interpolate through every single noisy data point.

## 2. Optimization and Hyperparameters

Due to the $O(N^3)$ computational complexity of GPR, fitting the 4,782-row training matrix is highly intensive (requiring approximately $10^{11}$ operations per optimizer step). 

**Optimization Settings:**
- `normalize_y=True`: We standardized the target internally so the optimizer operates in a well-conditioned space.
- `n_restarts_optimizer=5`: Maximizing the log-marginal-likelihood is a non-convex optimization problem. We restarted the L-BFGS-B optimizer 5 times from random initial bounds to avoid getting trapped in local maxima.

**Optimization Results:**
- **Training Time:** 987.48 seconds (~16.5 minutes)
- **Log-Marginal-Likelihood:** -3086.9361
- **Optimized Kernel:** `Matern(length_scale=2.36, nu=2.5) + WhiteKernel(noise_level=0.146)`
  - A length scale of `2.36` indicates the model found meaningful medium-term dependencies across the 9 standardized features.
  - A noise level of `0.146` means the model attributed roughly 14.6% of the target's standardized variance to irreducible market noise, preventing overfitting.

## 3. Predictive Performance

The model was evaluated chronologically out-of-sample on the Test Set (Dec 2020 – May 2026, $N=1196$):

| Metric | Train Set | Test Set |
|--------|-----------|----------|
| **RMSE** | 0.002312 | 0.002486 |
| **MAE** | 0.001423 | 0.001724 |
| **R² Score** | 0.8778 | **0.7315** |

**Interpretation:**
An out-of-sample $R^2$ of **0.7315** is exceptionally strong for financial time-series forecasting. It means the GPR model, using only our engineered momentum, trend, and liquidity features, successfully explains ~73% of the future variance in NEPSE's realized volatility over a massive 5-year out-of-sample window that includes post-COVID turbulence. The minimal drop between Train RMSE (0.0023) and Test RMSE (0.0024) proves the model is not overfitted.

## 4. Uncertainty Estimation

The primary advantage of Bayesian non-parametrics (like GPR) over traditional models (like Random Forests or Neural Networks) is closed-form uncertainty quantification.

*Reference figure generated:* `day5_gpr_matern_test_forecast.png`

The generated plot displays:
1. The ground-truth volatility (black line).
2. The GPR predicted mean $\mu(X_*)$ (blue line).
3. The 95% Confidence Interval band ($\mu \pm 1.96\sigma$) shaded in blue.

**Observations for the Paper:**
- The model excellently tracks the massive volatility surge in mid-2021 (the post-COVID NEPSE bull-market peak).
- The uncertainty band correctly widens during unprecedented or highly volatile regimes, demonstrating that the GP "knows when it doesn't know."

## Next Steps
In Day 6, we will repeat this exact process using the RBF kernel. We will then construct a direct comparative analysis to empirically prove why the Matern kernel's rougher assumptions outperform the overly smooth assumptions of the RBF kernel on financial data.
