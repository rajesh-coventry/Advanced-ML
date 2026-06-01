# Phase B, Day 8: Native Gaussian Process Classification (GPC)

This document covers the implementation and evaluation of a native multi-class `GaussianProcessClassifier` to predict market volatility regimes. 

## 1. Methodology

Unlike Day 7 (where we applied a hard mathematical threshold to a continuous regression output), Day 8 trains a pure probabilistic classifier to output regime probabilities directly from the 9 engineered input features.

**Model Configuration:**
- **Algorithm:** `sklearn.gaussian_process.GaussianProcessClassifier`
- **Multi-class Strategy:** `OneVsRest` (The algorithm fits three separate binary GPCs: Quiet vs Rest, Normal vs Rest, Turbulent vs Rest).
- **Posterior Approximation:** Laplace Approximation (since exact inference is intractable for GPC).
- **Kernel:** `Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)`. (The Matern kernel was carried over as the superior choice from Day 6).

## 2. Optimization and Computational Complexity

Because of the `OneVsRest` architecture and the $O(N^3)$ complexity of the Laplace approximation on 4,782 training samples, this was the most computationally expensive step in the pipeline.
- **Training Time:** 3209.05 seconds (~53.5 minutes).
- The optimizer pushed the `WhiteKernel` noise levels to their lower bounds (`1e-05`) across all three classes, as the probit link function in classification naturally absorbs most observation noise.

## 3. Classification Performance (Test Set)

The model was evaluated on the out-of-sample Test Set (Dec 2020 – May 2026, $N=1196$).

**Classification Report:**
```text
               precision    recall  f1-score   support

    Quiet (0)       0.67      0.95      0.79       315
   Normal (1)       0.80      0.74      0.77       637
Turbulent (2)       0.85      0.55      0.67       244

     accuracy                           0.76      1196
    macro avg       0.77      0.75      0.74      1196
 weighted avg       0.78      0.76      0.75      1196
```

## 4. Analytical Comparison: Day 7 (Thresholded GPR) vs Day 8 (Native GPC)

This is a critical finding for the coursework discussion section:
- **Day 7 (Thresholded GPR) Macro-F1:** 0.78
- **Day 8 (Native GPC) Macro-F1:** 0.74

**Why did the regression-based approach outperform the native classifier?**
Market volatility regimes are fundamentally **ordinal and continuous** (Quiet $\rightarrow$ Normal $\rightarrow$ Turbulent). 
1. The GPR (Day 7) models the exact magnitude of volatility. Thresholding this continuous prediction inherently preserves the ordinal relationship between regimes.
2. The GPC (Day 8) uses a `OneVsRest` multi-class strategy. It treats "Quiet", "Normal", and "Turbulent" as distinct, unordered categories (like categorizing apples, bananas, and oranges). By discarding the continuous spatial relationship between the classes, the GPC loses valuable mathematical context, resulting in a slight drop in accuracy (from 80% to 76%) and recall for the minority "Turbulent" class.

## 5. The Value of Probabilistic Output

Despite the slight drop in hard-accuracy, the native GPC provides one massive advantage: **Probabilistic Confidence Bounds**.

*Reference figure generated:* `day8_gpc_probabilities.png`

The bottom subplot in this figure visualizes the exact probability mass assigned to each regime over time. During clear bull/bear markets, the model is highly confident (>90% probability) in a single regime. However, during market transitions (such as the peak of the 2021 NEPSE bull run), we can visibly see the probability mass "smear" across two classes, indicating genuine mathematical uncertainty. This is the hallmark of Bayesian machine learning and a major point to highlight in the paper.

*Other reference figures generated:*
- `day8_gpc_confusion_matrix.png`: Heatmap showing where misclassifications occurred.
- `day8_gpc_regime_timeseries.png`: Overlay of the predicted classification bands on the true volatility curve, with misclassifications marked by 'X'.

## Next Steps
This concludes Phase B (Modeling). We now move to **Phase C (Days 9-10)** to execute structural break analyses (Chow & CUSUM) and consolidate the final paper figures.
