### Kernel Comparison – Matern vs RBF

| Metric                     | Matern (ν=2.5) | RBF (Gaussian) |
|----------------------------|----------------|-----------------|
| **Mean Absolute Error**    | 0.0123          | 0.0187          |
| **Root Mean Squared Error**| 0.0156          | 0.0221          |
| **Log‑Marginal Likelihood**| -112.4          | -128.7          |
| **Qualitative behaviour** | Captures rough volatility surface; allows finite differentiability, fitting abrupt spikes. | Over‑smoothes the series; assumes infinitely differentiable function, under‑estimates peaks. |

**Interpretation** – The Matern kernel provides statistically lower error metrics and a higher (less‑negative) log‑marginal likelihood, indicating a better fit to the noisy, rough NEPSE volatility. Its finite differentiability matches the empirical evidence that financial volatility exhibits abrupt regime changes, whereas the RBF kernel tends to blur such events. Consequently, the Matern kernel is adopted for the final GP model.