# Days 2–3 — Feature Engineering: 10 Stationary Input Features

This document provides the theoretical background, methodological justifications, statistical interpretations, and scientific write-up for the Days 2–3 stage of **Task 1: Market Regime Detection in NEPSE Using Gaussian Framework**.

---

## 1. Overview: Why Feature Engineering Matters for Gaussian Processes

A Gaussian Process model does not learn feature representations the way a deep neural network does — it learns a **covariance function (kernel)** defined directly over the input feature space. This means the quality, stationarity, and orthogonality of the input features have a disproportionately large effect on GP performance.

For a GP kernel to function correctly:
1. **Features must be stationary:** Non-stationary inputs (e.g., raw closing prices that trend upward over decades) cause the kernel to model trend rather than volatility structure. Stationary features fluctuate around a stable mean, allowing the GP to learn covariance patterns.
2. **Features should be approximately scaled similarly:** The GP's length-scale hyperparameter is learned jointly across all dimensions. If one feature spans [0, 3000] and another spans [0, 1], the optimizer must find a single length-scale that is simultaneously meaningful in both spaces — an impossible compromise without normalization.
3. **Multicollinearity should be minimized:** Near-perfectly correlated features add no information but destabilize the gradient landscape of the marginal likelihood optimization.

All 10 features below are engineered with these three constraints in mind.

---

## 2. Feature-by-Feature Analysis

### Feature 1: `log_return`

**Formula:** $\text{log\_return}_t = \ln\!\left(\dfrac{\text{Close}_t}{\text{Close}_{t-1}}\right)$

**Statistical Properties:**
- The log return is the continuously compounded daily return. For small daily moves (|r| < 5%), log return and arithmetic return are numerically near-identical, but log returns possess critical mathematical advantages:
  - They are **temporally additive**: the multi-day log return from day $t_0$ to $t_n$ equals the sum of daily log returns $\sum_{i=t_0}^{t_n} \text{log\_return}_i$.
  - They are **symmetric**: a log return of $+x$ and $-x$ represent gains and losses of equal geometric magnitude, whereas arithmetic returns are asymmetric (+10% followed by -9.09% returns to the same level).
  - They are **approximately normally distributed** for short time horizons — directly aligning with the GP's prior assumption of Gaussian-distributed observations.

**Why Not `daily_return` (arithmetic %)?**
The `percent_change` column already available in the raw NEPSE data is the arithmetic return: $(Close_t - Close_{t-1}) / Close_{t-1}$. For daily data this is numerically equivalent to `log_return` with a Pearson correlation of $r \approx 0.999$. Including both features would introduce **near-perfect multicollinearity**, which can create an ill-conditioned Gram (kernel) matrix and destabilize hyperparameter optimization. We retain only `log_return`.

---

### Feature 2: `abs_change_norm` *(proxy for hl_spread)*

**Formula:** $\text{abs\_change\_norm}_t = \dfrac{|\text{absolute\_change}_t|}{\text{index\_value}_t}$

**Why Not `hl_spread`?**
The ideal feature here was $\text{hl\_spread} = (\text{High} - \text{Low}) / \text{Close}$, but the NEPSE_Index table does not store High or Low prices — no table in `Indexes.db` does. Every table has the same 6 columns: `sn, index_value, absolute_change, percent_change, turnover, trade_date`. We therefore substitute with the closest available proxy.

**What `abs_change_norm` Captures:**
`absolute_change` is the net close-to-close point change of the NEPSE index. Taking the absolute value removes direction, and normalizing by `index_value` removes the price-scale non-stationarity:

- Large values → big daily price swings → elevated session activity
- Small values → quiet, range-bound sessions
- Zero values → flat sessions (common in NEPSE's early illiquid era)

**Key Properties:**
- Always non-negative (absolute value)
- Approximately equal to $|\text{log\_return}_t|$ for small daily moves (since $\ln(1+r) \approx r$)
- Stationary: expressed as a fraction of the current price level
- **Limitation vs. hl_spread:** Underestimates intraday volatility on sessions with large intraday reversals (opened high, closed near open). This limitation is noted explicitly in the paper.

---

### Feature 3: `normalized_trend_strength`

**Formula:** $\text{NTS}_t = \dfrac{\text{MA}_{20,t} - \text{MA}_{50,t}}{\text{Close}_t}$

**Statistical Properties:**
The difference between a short-term (20-day) and long-term (50-day) simple moving average of closing price is a classic "golden cross / death cross" momentum indicator.

**Why Normalize by Close?**
The raw difference $\text{MA}_{20} - \text{MA}_{50}$ is directly proportional to the price level. On a 200-point NEPSE index (early 2000s), a ±5-point difference represents 2.5% momentum. On a 3000-point index (2021), a ±80-point difference represents the same 2.67% momentum but looks 16× larger in absolute terms. The raw difference is therefore **non-stationary** — its variance grows with the price level.

Dividing by Close converts the difference into a percentage-like momentum metric with stable variance over time, satisfying the GP's stationarity requirement.

**Interpretation:**
- $\text{NTS} > 0$: Upward momentum (MA20 above MA50 → bullish trend alignment)
- $\text{NTS} < 0$: Downward momentum (MA20 below MA50 → bearish trend alignment)
- $\text{NTS} \approx 0$: Trend convergence → often precedes regime transitions

> [!NOTE]
> MA50 requires 50 calendar trading days to compute. The first 49 rows of `normalized_trend_strength` will be NaN. These are removed during Day-4 preprocessing, where the first 55 rows are dropped globally.

---

### Feature 4: `rolling_vol_20` ⟵ **GPR Target Variable**

**Formula:** $\text{rolling\_vol\_20}_t = \text{std}\!\left(\text{log\_return}_{t-19},\, \ldots,\, \text{log\_return}_t\right)$

**This is the primary output variable (target) for Gaussian Process Regression.**

**Statistical Properties:**
Rolling realized volatility is the empirical standard deviation of log returns over a trailing 20-day (approximately 1 calendar month) window. It is the most widely used proxy for short-term market risk in academic finance.

**Why is it the Right GPR Target?**

| Property | Raw Closing Price | Rolling Volatility |
|---|---|---|
| Stationarity | ❌ Non-stationary (trends upward) | ✅ Stationary (mean-reverting) |
| GP suitability | ❌ GP cannot extrapolate long-term trends | ✅ GP models covariance structure around mean |
| Interpretability | Low (dimensioned in index points) | High (annualized risk proxy) |
| Regime relevance | Indirect | Direct (high vol = turbulent regime) |

**Why 20 Days?**
- 20 trading days ≈ 1 calendar month.
- Too short (< 10 days): Estimates dominated by individual extreme days, producing noisy target.
- Too long (> 40 days): Over-smoothed signal; fails to capture regime shifts within a quarter.
- The 20-day standard is used by the CBOE VIX methodology (Whaley, 2009) and GARCH(1,1) benchmarks (Bollerslev, 1986).

**Volatility Clustering (ARCH Effect):**
The `rolling_vol_20` time series exhibits visible **volatility clustering** — high-volatility periods are followed by more high-volatility periods. This is the ARCH effect (Engle, 1982), well-documented in financial returns. GPR's kernel structure (particularly the Matérn kernel) can model this by learning that nearby points in time have high covariance, capturing the autocorrelation structure of volatility clusters.

---

### Feature 5: `rolling_abs_change_14` *(proxy for atr_14)*

**Formula:**
$$\text{abs\_norm}_t = \dfrac{|\text{absolute\_change}_t|}{\text{index\_value}_t}$$
$$\text{rolling\_abs\_change\_14}_t = \text{RollingMean}(\text{abs\_norm},\; 14\text{ days})$$

**Why Not `atr_14`?**
ATR-14 (Wilder, 1978) requires daily High and Low prices to compute the True Range:
$$\text{TrueRange}_t = \max(H_t - L_t,\;|H_t - C_{t-1}|,\;|L_t - C_{t-1}|)$$
Since no table in `Indexes.db` contains High or Low columns, ATR cannot be computed.

**What `rolling_abs_change_14` Captures:**
This is the 14-day rolling mean of the normalized absolute daily close-to-close move — a smoothed multi-session swing magnitude measure:

- High values → large average daily moves over the last 14 sessions → elevated volatility regime
- Low values → compressed daily moves → quiet / low-volatility regime

**Relationship to ATR:**
In markets with infrequent overnight gaps, ATR $\approx$ rolling mean of $|\text{close}_t - \text{close}_{t-1}|$ — exactly what this feature computes. For NEPSE, a daily settlement market with limited overnight gap risk, this close-to-close approximation is valid and documented in the paper.

**Complementarity with `abs_change_norm`:**
- `abs_change_norm` (Feature 2) is the raw single-day normalized swing — high-frequency, noisy.
- `rolling_abs_change_14` is its 14-day smoothed version — low-frequency, regime-aware.

---

### Feature 6: `bb_width`

**Formula:**
$$\text{bb\_width}_t = \frac{\text{Upper Band}_t - \text{Lower Band}_t}{\text{Middle Band}_t} = \frac{4 \times \sigma_{20,t}}{\text{MA}_{20,t}}$$

where $\sigma_{20}$ is the 20-day rolling standard deviation of Close, and bands are set at ±2σ around the 20-day MA.

**Statistical Properties:**
The Bollinger Band Width (Bollinger, 1992) is a normalized measure of relative volatility with respect to the current price trend:

- **Low BBW (Volatility Squeeze):** The market is consolidating. Bollinger himself identified that prolonged squeezes frequently precede large breakout moves (the "Squeeze Play"). In terms of market regimes, this corresponds to a **pre-transition state** — the regime is about to change but has not yet done so.
- **High BBW (Band Expansion):** The market is experiencing elevated directional volatility. This corresponds to the **Turbulent regime**.

**Distinction from `rolling_vol_20`:**
- `rolling_vol_20` captures **absolute** volatility in log-return units.
- `bb_width` captures **relative** volatility — how large the current volatility is compared to the price level and recent price trend.

These two features are positively correlated (both measure volatility) but not redundant, as they normalize differently and respond to different time horizons.

---

### Feature 7: `rsi_14`

**Formula (Wilder's RSI):**
$$\text{RS}_t = \frac{\text{EMA}_{13}(\text{gain}_t)}{\text{EMA}_{13}(\text{loss}_t)}, \qquad \text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}$$

where $\text{gain}_t = \max(\Delta\text{Close}_t, 0)$, $\text{loss}_t = \max(-\Delta\text{Close}_t, 0)$, and EMA₁₃ denotes the exponential moving average with $\alpha = 1/14$ (Wilder's smoothing equivalent to `com=13`).

**Statistical Properties:**
RSI is a **bounded oscillator** with range [0, 100], making it one of the most stationary features in the set regardless of the market era. Key regime-level interpretations:

| RSI Range | Momentum Signal | Regime Association |
|---|---|---|
| > 70 | Overbought: Sustained buying pressure | Normal → Turbulent transition |
| 40–60 | Neutral: Balanced buying/selling | Quiet / Normal regime |
| < 30 | Oversold: Sustained selling pressure | Turbulent (crash) regime |

**NEPSE-specific Relevance:**
Given NEPSE's high proportion of flat days (43.81% zero-return days, documented in Day 1), RSI will frequently hover near 50 during the early, illiquid period of the dataset. During the 2020–2021 bull market, RSI sustained above-70 readings for extended periods before the correction, making it a strong pre-transition signal.

---

### Feature 8: `macd_signal` (MACD Histogram)

**Formula:**
$$\text{MACD}_t = \text{EMA}_{12}(\text{Close}_t) - \text{EMA}_{26}(\text{Close}_t)$$
$$\text{Signal Line}_t = \text{EMA}_{9}(\text{MACD}_t)$$
$$\text{macd\_signal}_t = \dfrac{\text{MACD}_t - \text{Signal Line}_t}{\text{Close}_t}$$

**Statistical Properties:**
The MACD histogram (MACD line minus signal line, normalized by Close) captures the **acceleration of momentum** — the rate of change of the momentum itself, rather than momentum per se.

**Why the Histogram Over the MACD Line?**
The raw MACD line is the difference between two non-stationary EMAs. For a 3000-point index, MACD values may reach ±100, while for a 300-point index they may reach ±10. The raw MACD is therefore non-stationary across the dataset's history. Two steps are applied to achieve stationarity:
1. **Differencing via histogram:** The histogram is the MACD line minus its own 9-day EMA — a differencing operation that removes the local trend of the MACD line.
2. **Normalization by Close:** Divides by the current price level, suppressing the absolute-scale dependence.

The resulting `macd_signal` is a zero-centered, stationary momentum acceleration metric that naturally highlights regime transitions (crossover points) and momentum exhaustion.

---

### Feature 9: `turnover_spike` *(proxy for volume_spike)*

**Formula:** $\text{turnover\_spike}_t = \dfrac{\text{turnover}_t}{\text{RollingMean}(\text{turnover},\; 20)_t}$

**Why Not `volume_spike`?**
The NEPSE_Index table contains no share-volume column. However, `turnover` (total NPR value of all trades) is available. Since Turnover = Price × Shares, it is a value-weighted proxy for trading volume capturing both quantity and price — arguably a richer liquidity signal than raw share count.

**What `turnover_spike` Captures:**
Exactly the same semantics as a volume spike ratio — how unusual the current session's capital flow is relative to the recent 20-day baseline:

| Ratio | Interpretation | Regime Association |
|---|---|---|
| >> 1 | Extreme capital inflow/outflow | Regime transition marker |
| ≈ 1 | Normal participation | Normal / current regime stable |
| < 1 | Below-average activity | Quiet regime |

**Handling Zero-Turnover Early Data:**
NEPSE had zero turnover in many sessions before ~2008. A rolling mean of zeros creates division-by-zero (producing NaN). A floor of 1 NPR is applied to the denominator before dividing — correctly yielding a ratio of 0 (no capital flow) for genuinely inactive sessions rather than inf/NaN.

---

### Feature 10: `turnover_growth` *(zero-safe pct_change)*

**Formula:**
$$\text{turnover\_growth}_t = \dfrac{\text{Turnover}_t - \text{Turnover}_{t-1}}{\max(\text{Turnover}_{t-1},\; 1\text{ NPR})}$$

**Statistical Properties:**
`turnover_growth` captures the day-over-day percentage change in total capital flow — a stationary, zero-mean liquidity growth signal:
- Positive → rising capital inflow, often accompanying bullish regime transitions
- Negative → capital outflow, common in early bear markets
- Near-zero → stable liquidity, characteristic of the Normal regime

**Critical Fix — Zero-Safe Denominator:**
Using standard `pct_change()` on a series with many zero values (NEPSE's pre-2008 era) produces inf and NaN when the denominator is zero. In the first run, this left **3,804 of 6,033 rows as NaN**, making the feature nearly unusable.

The fix: shift the prior turnover and replace zero denominators with 1 NPR before dividing:
```python
prior_safe = prior_turnover.where(prior_turnover != 0, other=1.0)
turnover_growth = (turnover - prior_turnover) / prior_safe
```
This yields 0.0 growth for genuinely zero-to-zero sessions (correct — no change occurred) and recovers **6,032 valid rows** from the original 2,229.

**Outlier Clipping:**
Transitions from near-zero to measurable turnover (frontier market activation events) still produce extreme ratios. Values beyond ±500% (5×) are winsorized and reported in the paper.

---

## 3. Feature Summary Table

| # | Feature Name | Formula Summary | Type | Stationary? | NaN Rows |
|---|---|---|---|---|---|
| 1 | `log_return` | ln(Close_t / Close_{t-1}) | Return | ✅ | 1 |
| 2 | `abs_change_norm` | \|absolute_change\| / index_value | Volatility proxy | ✅ | 0 |
| 3 | `normalized_trend_strength` | (MA20 − MA50) / Close | Momentum/Trend | ✅ | 49 |
| 4 | `rolling_vol_20` | std(log_return, 20d) | **GPR Target** | ✅ | 20 |
| 5 | `rolling_abs_change_14` | 14d mean(\|abs_change\| / Close) | Volatility proxy (ATR) | ✅ | 13 |
| 6 | `bb_width` | (Upper − Lower) / Middle Band | Volatility | ✅ | 19 |
| 7 | `rsi_14` | Wilder's RSI, 14-period | Momentum | ✅ | 1 |
| 8 | `macd_signal` | (MACD − Signal) / Close | Momentum | ✅ | 0 |
| 9 | `turnover_spike` | Turnover / 20d Mean(Turnover) | Liquidity proxy | ✅ | 19 |
| 10 | `turnover_growth` | zero-safe pct_change(Turnover) | Liquidity | ✅ | 1 |

> [!NOTE]
> **Database schema reality:** The NEPSE `Indexes.db` contains only 6 columns per table: `sn, index_value, absolute_change, percent_change, turnover, trade_date`. No High, Low, or share-volume data is present. Features 2, 5, and 9 are the closest computable proxies from available data; this is acknowledged in the paper's Experimental Setup section.

> [!IMPORTANT]
> The maximum NaN window across all features is **50 rows** (`normalized_trend_strength` requires MA50). Day-4 preprocessing drops the first 55 rows to guarantee all features are non-NaN for every training and test sample.

---

## 4. Correlation Analysis & Multicollinearity Assessment

The correlation heatmap (`feature_correlation_heatmap.png`) is produced at the end of the feature engineering step and saved to `Research/figures/`.

**Expected Correlation Clusters:**

**Observed Correlation Clusters (from actual run):**

1. **Volatility Cluster** (`rolling_vol_20`, `bb_width`, `abs_change_norm`, `rolling_abs_change_14`):
   As expected, all four measure realized volatility and show moderate positive correlation:
   - `abs_change_norm` ↔ `rolling_abs_change_14`: r = **0.68** (smoothed vs. raw version of same signal)
   - `rolling_vol_20` ↔ `bb_width`: r = **0.66** (log-return std vs. price-normalized std)
   - `rolling_vol_20` ↔ `rolling_abs_change_14`: r = **0.52** (complementary lookback windows)
   None exceeds the 0.90 multicollinearity threshold.

2. **Momentum Cluster** (`normalized_trend_strength`, `rsi_14`, `macd_signal`):
   - `normalized_trend_strength` ↔ `rsi_14`: r = **0.60** (both capture directional momentum)
   - `rsi_14` ↔ `macd_signal`: r = **0.51** (bounded oscillator vs. MACD acceleration)

3. **Liquidity Cluster** (`turnover_spike`, `turnover_growth`):
   Both derived from `turnover`. Mild correlation expected but different time aggregations keep them complementary.

4. **Return** (`log_return`): Low correlation with smoothed features — correctly unsmoothed single-day signal.

**Multicollinearity Verdict:** No feature pair exceeds $|r| > 0.90$. The highest observed correlation is 0.68 (`abs_change_norm` ↔ `rolling_abs_change_14`), well within acceptable bounds for GP optimization stability.

**Multicollinearity Threshold:**
Any feature pair with $|r| > 0.90$ would be considered candidates for removal. Based on the feature construction above, no pair is expected to exceed this threshold (the `daily_return` / `log_return` pair at r ≈ 0.999 was already preemptively resolved by excluding `daily_return`).

---

## 5. Draft Text for the Scientific Paper

*The following can be adapted for the **Dataset and Problem Description** and **Experimental Setup** sections of the 6-page paper.*

---

### Section: Feature Engineering

> "Ten stationary input features were engineered from the raw NEPSE index data to capture four distinct dimensions of market behavior: return, volatility, momentum, and liquidity. The features are: (1) log return — the continuously compounded daily price change; (2) normalized absolute daily change — the magnitude of net close-to-close movement as a fraction of the index level (a proxy for intraday range, as the database does not provide High/Low prices); (3) normalized trend strength — the 20-day minus 50-day moving average difference normalized by closing price, a stationary momentum indicator; (4) 20-day rolling volatility of log returns, designated as the GPR regression target; (5) a 14-day rolling mean of the normalized absolute price change, serving as an ATR proxy for smoothed swing magnitude; (6) Bollinger Band Width, normalizing realized volatility by the current price trend; (7) RSI-14, a bounded momentum oscillator; (8) MACD histogram normalized by closing price; (9) turnover spike ratio — daily NPR turnover divided by its 20-day rolling mean, serving as a liquidity event detector in the absence of share-volume data; and (10) zero-safe turnover growth, the percentage change in daily turnover with a safe denominator to handle the zero-turnover frontier-market era prior to 2008. The 20-day rolling standard deviation of log returns (`rolling_vol_20`) was designated as the GPR target variable. Multicollinearity was controlled by excluding the arithmetic daily return (Pearson r ≈ 0.999 with log return). All features were normalized by the contemporaneous index level to ensure stationarity across the full 25-year sample period spanning both frontier-market and active-market eras of NEPSE."

---

## 6. Figures Produced (Days 2–3)

All figures are saved to `Research/figures/` at 150 DPI.

| Figure File | Content |
|---|---|
| `feature_01_log_return.png` | Daily log return time series — shows volatility clustering |
| `feature_02_abs_change_norm.png` | Normalized absolute daily price swing (hl_spread proxy) |
| `feature_03_normalized_trend_strength.png` | MA20/MA50 on price + NTS oscillator below |
| `feature_04_rolling_vol_20_TARGET.png` | GPR target: rolling volatility with NEPSE price overlay |
| `feature_05_rolling_abs_change_14.png` | 14-day smoothed swing magnitude (ATR-14 proxy) |
| `feature_06_bb_width.png` | Bollinger Bands on price + BBW oscillator below |
| `feature_07_rsi_14.png` | RSI-14 with overbought/oversold reference lines |
| `feature_08_macd_signal.png` | MACD components + histogram bar chart |
| `feature_09_turnover_spike.png` | Turnover spike ratio relative to 20-day baseline (volume_spike proxy) |
| `feature_10_turnover_growth.png` | Zero-safe turnover growth (capital flow change) |
| `feature_correlation_heatmap.png` | 10×10 Pearson correlation matrix — paper-quality figure |

---

## 7. What Comes Next (Day 4)

After the 10 features are computed, the Day-4 preprocessing step performs:
1. **Drop first 55 rows** (removes all NaN rows introduced by rolling calculations — MA50 requires the longest window of 50 days; 55 gives 5 rows of buffer)
2. **Define GPR target:** `y_regression = rolling_vol_20`
3. **Define volatility regime classes:** 3-class GPC labels using 50th and 85th percentile thresholds on `rolling_vol_20` (training set only, to prevent leakage)
4. **Define feature matrix X:** All 9 input features (all features except `rolling_vol_20` which is the target — plus the class label column)
5. **Temporal train/test split:** First 80% → train, last 20% → test (no shuffling)
6. **Fit `StandardScaler` on `X_train` only**, then transform both `X_train` and `X_test`
