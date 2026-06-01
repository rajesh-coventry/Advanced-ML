# Dataset Load, Chronological Sorting, and Exploratory Data Analysis (EDA):

This document provides the theoretical background, methodological justifications, statistical interpretations, and scientific write-up for the Day 1 stage of **Task 1: Market Regime Detection in NEPSE Using Gaussian Framework**.

## 1. What We Did & Rationale:

### A. SQLite Database Loading:

* **Action:** Queried the `NEPSE_Index` table from the local SQLite database `Indexes.db` using Python's built-in `sqlite3` driver combined with `pandas.read_sql_query()`.

* **Why We Do This:** 
  - Storing structured historical financial indices in an SQLite database is a standard data engineering practice that ensures data integrity and high-speed local retrieval.
  
  - Loading via pandas allows us to manipulate the table directly as a structured DataFrame, aligning it with downstream machine learning APIs (such as `scikit-learn`'s Gaussian Process models).

### B. Chronological Sorting & Datetime Parsing:

* **Action:** Converted the `trade_date` text column to ISO-formatted pandas datetime objects using `pd.to_datetime()`. Sorted the DataFrame by `trade_date` in ascending order (earliest records to latest records) and reset the index.

* **Why We Do This (Critical to prevent Data Leakage):**
  - The SQLite database stor es records in reverse-chronological order (newest date first).
  
  - Financial time-series models use lag operators (e.g., $Close_{t-1}$) and rolling statistics (e.g., standard deviation over the last 20 days: $[t-19, t]$).
  
  - If we calculate rolling features on reverse-chronological data, `.shift(1)` would fetch the price of $t+1$ (the next calendar day). This represents **look-ahead bias (future data leakage)**. The model would train on future information that is unavailable in real-world deployment, yielding artificially inflated performance metrics but failing in production.
  
  - Ascending sorting ensures that calculations represent true historical moving windows.

## 2. Statistical Analysis & Market Dynamics:

Based on the 6,033 trading days in the dataset, the following properties were discovered:

### A. Basic Statistical Indicators:

* **Data Scale:** 6,033 trading days (approx. 24–25 years of daily financial activity).

* **Index Range:** The index closing price reaches a wide range, starting from below 1,000 points to a peak exceeding 3,000 points.

* **Returns Distribution:** The daily percentage changes (`percent_change`) hover closely around a mean of ~0.0%, exhibiting characteristic financial properties such as heavy tails (excess kurtosis) and volatility clustering.

### B. Ratio of Up/Down Days:

An analysis of daily price trends reveals a unique statistical footprint of the NEPSE index:
- **Up Days (Positive return):** 27.12% (1,636 days)

- **Down Days (Negative return):** 29.07% (1,754 days)

- **Flat Days (Zero return):** 43.81% (2,643 days)

**Why are there so many Flat Days? (Key Scientific Discovery):**
- In modern liquid markets (e.g., S&P 500), flat trading days are practically non-existent.

- In the NEPSE dataset, the high proportion of flat days (43.81%) is a historical artifact of **frontier market illiquidity**, especially during the 2000–2010 decade. During this period, trading was semi-manual, listing rules were basic, and order-matching was sparse, resulting in many consecutive days with zero index movement.

- **Implication for Modelling:** This indicates extreme zero-inflation in early daily returns. It suggests that a rolling volatility target (such as standard deviation over a 20-day window) is far superior to raw daily returns. Volatility calculations smooth out these zero-movement days, enabling the Gaussian Process models to learn consistent regime signals rather than getting disrupted by long periods of artificial zeroes.


This close balance between up and down days is typical of long-term stock indices and justifies the use of a volatility-based regime framework rather than a simple direction-based binary classification.

## 3. Financial and Scientific Interpretations for the Paper:

The exploratory visualizations generated during this stage reveal crucial market structures that must be documented in the paper:

### A. The Evolution of Market Size (Index Value vs. Turnover):

- **Visual Trend:** In the early parts of the series, the daily turnover (value traded in NPR) is extremely low (flat lining near the bottom of the axis), even when the NEPSE index experienced minor percentage adjustments.

- **The Liquidity Explosion:** In the latter half of the series (especially post-2020), there is an exponential surge in daily turnover coinciding with the rapid growth of the NEPSE index.

- **Scientific Value:** This proves that the NEPSE market has transitioned from a illiquid, frontier market to a much larger, highly active, retail-dominated market. The absolute volatility scales differ significantly across these eras, reinforcing why we normalized our trend feature `(ma_20 - ma_50) / Close` and why we used percentile-based volatility thresholds rather than fixed points for defining market regimes.

### B. High-Volatility Shock Clusters (Structural Breaks):

- The price chart displays distinct, sharp drawdowns and rapid exponential climbs (e.g., the massive bull market of 2020–2021 followed by a severe bear market).

- Volatility is not distributed uniformly over time. It exhibits **volatility clustering**—high-volatility days are followed by high-volatility days, and low-volatility days are followed by low-volatility days.

- This clustering is the primary scientific justification for using **Gaussian Process Regression (GPR)**: GPs can capture non-linear, time-varying uncertainty structures, widening their confidence intervals (error bands) during high-volatility clusters.

## 4. Draft Text for the Scientific Paper:

*This section can be copied directly into the **Dataset and Problem Description** or **Experimental Setup** sections of your 6-page draft.*

### Section: Dataset and Preprocessing:

> "The primary dataset comprises daily closing index values, trading volume, and turnover of the Nepal Stock Exchange (NEPSE), spanning 6,033 trading days. The raw data was obtained from a localized relational SQLite database structure. 
> 
> Due to the data storage configuration containing records in reverse-chronological order, a preprocessing pipeline was implemented to sort the time series in ascending order ($t_0 \to t_N$). This step is mathematically critical; applying rolling estimators or lag operators directly on reverse-ordered data introduces look-ahead bias by exposing future parameters to the current prediction step. The parsed temporal features reveal structural shifts in liquidity, where trading turnover experienced an exponential increase post-2020. To ensure stability across these distinct market eras, all engineered trend features were scale-normalized against the closing index price, preventing parameter drift in the Gaussian covariance optimization."