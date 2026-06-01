# Phase B, Day 6: Gaussian Process Regression (RBF vs Matern Comparison)

This document covers the results of training the Gaussian Process Regressor using the RBF (Squared Exponential) kernel and provides the crucial side-by-side comparative analysis against the Day 5 Matern ($\nu=2.5$) kernel, as mandated by the assignment brief.

## 1. RBF Kernel Training & Optimization

The model was initialized exactly the same as Day 5, with the only change being the core kernel:
```python
RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
```

**Optimization Results:**
- **Training Time:** 348.02 seconds (~5.8 minutes). *(Noticeably faster than Matern due to the simpler gradient computations of the squared exponential function vs. the modified Bessel functions in Matern).*
- **Log-Marginal-Likelihood:** -3341.6557 *(Worse/lower than Matern's -3086.9361, indicating a poorer fit to the training data).*
- **Optimized Kernel:** `RBF(length_scale=1.96) + WhiteKernel(noise_level=0.177)`

## 2. Kernel Comparison Table (Test Set)

The following table demonstrates the out-of-sample predictive performance of both kernels on the strict hold-out Test Set (Dec 2020 – May 2026):

| Metric | Matern ($\nu=2.5$) | RBF (Squared Exponential) |
|--------|------------------|---------------------------|
| **RMSE** | **0.002486** | 0.002575 |
| **MAE** | **0.001724** | 0.001769 |
| **R² Score** | **0.7315** | 0.7119 |
| **Optimized Length Scale** | 2.3600 | 1.9600 |
| **Optimized Noise Level** | 0.146 | 0.177 |

*Reference figure generated:* `day6_gpr_kernel_comparison.png` overlays the predicted means of both kernels against the actual rolling volatility.

## 3. Analytical Discussion (For the Paper's "Methods/Results" Section)

**Why did Matern outperform RBF?**
The empirical superiority of the Matern kernel ($R^2 = 0.7315$) over the RBF kernel ($R^2 = 0.7119$) aligns perfectly with time-series forecasting theory.

1. **The "Too Smooth" Problem:** The RBF kernel assumes that the underlying function it is modeling is infinitely differentiable. This creates an inherently "smooth" prior. Financial markets, particularly frontier markets like NEPSE, are defined by abrupt structural breaks, panic selling, and sudden liquidity surges. An infinitely smooth kernel will inherently struggle to adapt to these non-differentiable "kinks" in the volatility surface.
2. **The Matern Solution:** The Matern ($\nu=2.5$) kernel assumes the function is only twice-differentiable. This relaxes the smoothness constraint, allowing the GP sample paths to be "rougher". Consequently, the Matern model is more agile and responds faster to regime transitions and sudden volatility spikes.
3. **Noise Attribution:** Notice that the RBF kernel optimized to a higher `noise_level` (0.177) than the Matern kernel (0.146). Because the RBF kernel is too rigid to follow the jagged path of the actual market volatility, it is mathematically forced to dismiss more of the true signal variance as "irreducible white noise". 

**Conclusion for GPR Phase:**
The Matern kernel provides the optimal inductive bias for financial volatility modeling. We will proceed to use the Matern kernel (or its principles) as we transition into the Gaussian Process Classification (GPC) phase to classify the three volatility regimes.
