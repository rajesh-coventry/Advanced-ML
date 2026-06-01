# Day 10 – Structural‑Break Analysis Documentation

**File:** `day_10_structural_breaks.py`

## Purpose
The script performs two classic econometric tests on the 20‑day rolling volatility series generated on **Day 4**:
1. **Chow test** – evaluates a pre‑specified breakpoint (default: 2020‑03‑01, the onset of the COVID‑19 market crash).
2. **CUSUM of Squares** – a cumulative‑sum test that checks for unknown structural breaks across the whole series.

Both tests are applied to the concatenated training and test volatility series, and the results are visualised and saved as high‑resolution PNG figures. A compact `pickle` summary containing key statistics is also persisted for later manuscript inclusion.

## Key Steps
| Step | Description |
|------|-------------|
| **Load data** | Reads `day_4_processed_data.npz` containing `y_train_reg`, `y_test_reg`, and their date indices. |
| **Concatenate** | Forms a continuous series `y_full` and `dates_full`. |
| **Chow test** | Fits an OLS trend model on the full series and on the two subsamples split at `break_date`. Computes the F‑statistic and p‑value manually (identical to `statsmodels` implementation). |
| **CUSUM of Squares** | Computes the cumulative sum of squared residuals from the full OLS model (`np.cumsum(model_full.resid ** 2)`). Confidence bounds are not available in this manual implementation, so they are omitted from the plot. |
| **Plotting** | Generates two figures:
- `day10_cusum_squares.png` – CUSUM statistic over time.
- `day10_volatility_break_candidate.png` – Volatility series with the chosen break highlighted. |
| **Summary pickle** | Stores a dictionary with Chow statistics, CUSUM series, critical values (NaN), and break index at `Research/models/structural_break_summary.pkl`. |

## Results (from the latest successful run)
```
[Chow Test] Results for break at 2020-03-01:
  F-statistic = 193.9126, p-value = 0.0000e+00
```
The extremely low p‑value indicates a statistically significant structural break at the specified date.

Both figures were saved successfully:
- `Research/figures/day10_cusum_squares.png`
- `Research/figures/day10_volatility_break_candidate.png`

A summary pickle was also saved:
- `Research/models/structural_break_summary.pkl`

## Known Issues / Minor Adjustments
- **SyntaxWarning**: The command line in the module docstring contains `"\S"`, which triggers a warning on Windows. It does not affect execution but can be silenced by using a raw string (`r"\S"`) or escaping the backslash.
- **CUSUM confidence bounds**: The manual CUSUM implementation does not compute formal confidence intervals. If required, the original `breaks_cusumolsresid` function can be used, but it requires additional handling of the returned statistic shape.
- **Plot bounds**: The script now conditionally draws confidence‑interval lines only when finite values are provided.

## Execution Checklist
1. Ensure the virtual environment is activated (`.venv\Scripts\activate`).
2. Run the script:
   ```powershell
   .venv\Scripts\python.exe Research\day_10_structural_breaks.py
   ```
3. Verify that the two PNG files appear in `Research/figures` and the pickle file appears in `Research/models`.
4. Review the printed Chow‑test statistics for significance.