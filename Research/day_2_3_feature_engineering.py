"""
===================
FEATURE ENGINEERING:
===================
Purpose:
-------
This module engineers 10 stationary input features from the raw NEPSE index
data. Every feature is:mCalculated on the chronologically-sorted DataFrame.

The 10 features are:
  1.  log_return              — Log-differenced returns (stationary proxy for price)
  2.  hl_spread               — Intraday High-Low range normalized by Close
  3.  normalized_trend_strength — (MA20 - MA50) / Close  (stationary momentum)
  4.  rolling_vol_20          — 20-day rolling std of log_return  (GPR TARGET)
  5.  atr_14                  — Average True Range over 14 days (normalized)
  6.  bb_width                — Bollinger Band width (20-day)
  7.  rsi_14                  — Relative Strength Index over 14 days
  8.  macd_signal             — MACD histogram (12/26/9 EMA crossover)
  9.  volume_spike            — Daily volume / 20-day rolling mean of volume
  10. turnover_growth         — Percentage change in daily turnover

Dependencies: 
------------
    day_1_eda.run_day_1_eda() must be called first to produce the sorted
    DataFrame. This script accepts that DataFrame as input.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ---------------------------------------------------------------------------
# Shared Plot Aesthetics — identical style parameters for consistency
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family'       : 'sans-serif',
    'font.sans-serif'   : ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'axes.edgecolor'    : '#cccccc',
    'axes.linewidth'    : 0.8,
    'xtick.color'       : '#333333',
    'ytick.color'       : '#333333',
    'grid.color'        : '#eeeeee',
    'grid.linestyle'    : '--',
    'figure.titlesize'  : 14,
    'axes.labelsize'    : 11,
    'axes.titlesize'    : 12,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
})


# ---------------------------------------------------------------------------
# Helper — resolve the figures output directory relative to this script
# ---------------------------------------------------------------------------
def _get_figures_dir() -> str:
    """Return the absolute path to Research/figures/, creating it if needed."""
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


# -----------------------------------------------------
# Helper — save a matplotlib figure and report its path
# -----------------------------------------------------
def _save_fig(fig: plt.Figure, filename: str) -> None:
    """Save *fig* to Research/figures/<filename> at 150 dpi."""
    path = os.path.join(_get_figures_dir(), filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [Saved] {path}")
    plt.close(fig)


# ======================
# LOG RETURN
# ======================
def compute_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 1: log_return
    ----------------------
    Formula : log_return_t = ln( Close_t / Close_{t-1} )

    Statistical Rationale
    ---------------------
    Log returns (continuously compounded returns) are preferred over arithmetic
    returns for the following reasons:
      - **Symmetry:** A +10% and a -10% log return are symmetric around zero,
        whereas arithmetic returns are asymmetric (+10% vs. -9.09% equivalent).
      - **Temporal Additivity:** Multi-period log returns sum correctly, making
        them suitable for rolling calculations.
      - **Approximate Normality:** For small daily moves, log returns are
        approximately normally distributed, aligning with the Gaussian Process
        prior assumption of Gaussian-distributed noise.
      - **Stationarity:** Unlike raw closing prices (which are non-stationary),
        log returns oscillate around a near-zero mean, satisfying the
        stationarity requirement of the GP kernel.

    Note: `daily_return` (arithmetic %) is NOT included as a feature because it
    correlates with log_return at r ≈ 0.999 — including both would introduce
    near-perfect multicollinearity, destabilizing the GP hyperparameter
    optimization (marginal likelihood maximization).

    The first row will be NaN because there is no prior day to reference.
    """
    print("\n[Feature 1] Computing log_return ...")
    df = df.copy()

    # ln(Close_t / Close_{t-1})  — pandas .shift(1) fetches the PREVIOUS row
    # because the DataFrame is sorted in ascending chronological order (Day 1).
    df['log_return'] = np.log(df['index_value'] / df['index_value'].shift(1))

    # ---- Verification printout ----
    print(f"  log_return — first 5 valid values:\n{df['log_return'].dropna().head()}\n")
    print(f"  Descriptive stats:\n{df['log_return'].describe()}\n")

    # ---- Visualization ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['trade_date'], df['log_return'], linewidth=0.7,
            color='#1f77b4', alpha=0.85)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_title('Feature 1: Daily Log Return  [ ln(Close_t / Close_{t-1}) ]',
                 fontweight='bold')
    ax.set_xlabel('Trade Date', fontweight='bold')
    ax.set_ylabel('Log Return', fontweight='bold')
    ax.annotate('Note: volatility clustering clearly visible — '
                'high-vol periods follow high-vol periods',
                xy=(0.02, 0.92), xycoords='axes fraction',
                fontsize=8, color='gray')
    fig.tight_layout()
    _save_fig(fig, 'feature_01_log_return.png')

    return df


# ===============================================
# ABSOLUTE CHANGE (NORMALIZED DAILY PRICE SWING)
# ===============================================
def compute_abs_change_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 2: abs_change_norm  (replaces hl_spread)
    --------------------------------------------------
    Formula : abs_change_norm_t = |absolute_change_t| / index_value_t

    Context — Why Not hl_spread?
    -----------------------------
    The ideal feature here was hl_spread = (High - Low) / Close, which
    captures intraday realized volatility. However, the NEPSE_Index table
    in Indexes.db only stores: index_value, absolute_change, percent_change,
    turnover, trade_date. No High or Low columns are present anywhere in
    the database. Therefore, we substitute with the closest available proxy.

    What abs_change_norm Captures
    ------------------------------
    `absolute_change` is the net point change of the NEPSE index from the
    prior session's close to the current session's close:
        absolute_change_t = index_value_t - index_value_{t-1}

    Taking the absolute value gives the **magnitude** of the daily move
    (regardless of direction), and normalizing by the current index level
    removes the price-scale non-stationarity:

      - Large values → big price swings → elevated market activity
      - Small values → quiet, range-bound sessions

    Note: Unlike hl_spread (which captures the intraday High-to-Low range),
    abs_change_norm captures the net close-to-close move. It therefore
    underestimates intraday volatility on days where the index reverses
    strongly (opened high, closed low). This limitation is noted in the paper.

    Key Properties:
      - Always non-negative (absolute value of change)
      - Approximately equivalent to |log_return_t| for small daily moves
        (since log(1+r) ≈ r for small r)
      - Stationary: expressed as a fraction of the current price level
    """
    print("[Feature 2] Computing abs_change_norm (|absolute_change| / index_value) ...")
    df = df.copy()

    df['abs_change_norm'] = df['absolute_change'].abs() / df['index_value']

    print(f"  abs_change_norm — first 5 valid values:\n{df['abs_change_norm'].head()}\n")
    print(f"  Descriptive stats:\n{df['abs_change_norm'].describe()}\n")

    # ---- Visualization ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(df['trade_date'], df['abs_change_norm'],
                    color='#d62728', alpha=0.30, label='|Daily Change| / Index')
    ax.plot(df['trade_date'], df['abs_change_norm'], color='#d62728',
            linewidth=0.6, alpha=0.8)
    ax.set_title(
        'Feature 2: Normalized Absolute Daily Price Swing\n'
        '[ |absolute_change| / index_value ]  (proxy for intraday range)',
        fontweight='bold'
    )
    ax.set_xlabel('Trade Date', fontweight='bold')
    ax.set_ylabel('|Change| / Index (fraction)', fontweight='bold')
    ax.legend(loc='upper left')
    fig.tight_layout()
    _save_fig(fig, 'feature_02_abs_change_norm.png')

    return df


# ======================================
# NORMALIZED TREND STRENGTH
# ======================================
def compute_normalized_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 3: normalized_trend_strength
    --------------------------------------
    Formula : nts_t = (MA20_t - MA50_t) / Close_t
      where MA20 and MA50 are simple moving averages of Close over 20 and 50
      trading days respectively.

    Statistical Rationale
    ---------------------
    The difference between a short-term (MA20) and a long-term (MA50) moving
    average is a classic momentum signal:
      - **Positive NTS:** Short-term average is above long-term → upward trend.
      - **Negative NTS:** Short-term average is below long-term → downward trend
        or transition to bear market.

    Why Normalize by Close?
      The raw MA difference (MA20 - MA50) grows proportionally with the price
      level — on a 200-point index it may be ±5 points, while on a 3000-point
      index it may be ±80 points for an equivalent momentum signal. Without
      normalization, this feature would be non-stationary, confounding the GP's
      covariance estimation across different market eras.
      Dividing by Close produces a percentage-like quantity that is stationary
      and directly interpretable (e.g., 0.02 = MA20 is 2% above MA50).

    NaN at Start:
      MA50 requires 50 days to compute, so the first 49 rows will be NaN.
      These are removed during the Day-4 preprocessing step.
    """
    print("[Feature 3] Computing normalized_trend_strength ...")
    df = df.copy()

    ma_20 = df['index_value'].rolling(window=20, min_periods=20).mean()
    ma_50 = df['index_value'].rolling(window=50, min_periods=50).mean()

    df['normalized_trend_strength'] = (ma_20 - ma_50) / df['index_value']

    # Store intermediate MAs for visualization reference (not final features)
    df['_ma_20'] = ma_20
    df['_ma_50'] = ma_50

    print(f"  normalized_trend_strength — first 5 valid values:\n"
          f"{df['normalized_trend_strength'].dropna().head()}\n")
    print(f"  Descriptive stats:\n{df['normalized_trend_strength'].describe()}\n")

    # ---- Visualization: dual-panel — MAs on price, NTS below ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(df['trade_date'], df['index_value'], color='#1f77b4',
             linewidth=0.9, label='NEPSE Close', alpha=0.9)
    ax1.plot(df['trade_date'], df['_ma_20'], color='#ff7f0e',
             linewidth=1.2, label='MA-20', alpha=0.85)
    ax1.plot(df['trade_date'], df['_ma_50'], color='#2ca02c',
             linewidth=1.2, label='MA-50', alpha=0.85)
    ax1.set_title('NEPSE Index with MA-20 and MA-50 Overlaid', fontweight='bold')
    ax1.set_ylabel('Index Value (Points)', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)

    ax2.plot(df['trade_date'], df['normalized_trend_strength'],
             color='#9467bd', linewidth=0.9, alpha=0.85)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.fill_between(df['trade_date'], df['normalized_trend_strength'],
                     where=(df['normalized_trend_strength'] > 0),
                     color='#2ca02c', alpha=0.2, label='Bullish Momentum')
    ax2.fill_between(df['trade_date'], df['normalized_trend_strength'],
                     where=(df['normalized_trend_strength'] < 0),
                     color='#d62728', alpha=0.2, label='Bearish Momentum')
    ax2.set_title('Feature 3: Normalized Trend Strength  [ (MA20 - MA50) / Close ]',
                  fontweight='bold')
    ax2.set_xlabel('Trade Date', fontweight='bold')
    ax2.set_ylabel('NTS (fraction)', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    _save_fig(fig, 'feature_03_normalized_trend_strength.png')

    # Drop temporary helper columns (keep the DataFrame clean)
    df.drop(columns=['_ma_20', '_ma_50'], inplace=True)

    return df


# ============================================
# ROLLING VOLATILITY (GPR TARGET)
# ============================================
def compute_rolling_vol_20(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 4 / GPR Target: rolling_vol_20
    ----------------------------------------
    Formula : rolling_vol_20_t = std( log_return_{t-19} ... log_return_t )
              i.e., the 20-day rolling standard deviation of log returns.

    THIS IS THE TARGET VARIABLE FOR GAUSSIAN PROCESS REGRESSION.

    Statistical Rationale
    ---------------------
    Rolling volatility is a realized-volatility estimator — the empirical
    standard deviation of log returns over a backward-looking window. It is:

      - **Stationary:** Unlike price levels, rolling volatility fluctuates around
        a long-run mean (mean-reverting behavior), satisfying the GP stationarity
        assumption.
      - **Directly Interpretable:** Expressed in the same units as log returns,
        a value of 0.01 means daily returns have a standard deviation of 1% over
        the last 20 trading days.
      - **Continuous and Smooth Enough for GPR:** The 20-day window smooths
        extreme single-day outliers while still capturing regime-level shifts
        within a month — the ideal signal for a GP to fit.
      - **Ecologically Valid:** Rolling volatility is widely used in academic
        and practitioner finance as the standard proxy for market risk.

    Why 20 Days?
      20 trading days ≈ 1 calendar month. This window size is a balance:
        - Too short (< 10 days): Unstable estimates dominated by individual
          extreme days.
        - Too long (> 40 days): Over-smoothed; fails to respond quickly to new
          volatility regimes.
      The 20-day window is the standard used in the VIX methodology and
      Bollerslev (1986) GARCH benchmarks.

    NaN: The first 19 rows will be NaN (need 20 data points for std).

    IMPORTANT: log_return must be computed before calling this function.
    """
    print("[Feature 4 — GPR TARGET] Computing rolling_vol_20 ...")

    if 'log_return' not in df.columns:
        raise ValueError("log_return must be computed before rolling_vol_20. "
                         "Call compute_log_return() first.")

    df = df.copy()
    df['rolling_vol_20'] = (
        df['log_return']
        .rolling(window=20, min_periods=20)
        .std()
    )

    print(f"  rolling_vol_20 — first 5 valid values:\n"
          f"{df['rolling_vol_20'].dropna().head()}\n")
    print(f"  Descriptive stats:\n{df['rolling_vol_20'].describe()}\n")

    # ---- Visualization: volatility time-series with NEPSE index overlay ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1.2]})

    ax1.plot(df['trade_date'], df['index_value'], color='#1f77b4',
             linewidth=0.9, alpha=0.85, label='NEPSE Index Close')
    ax1.set_title('NEPSE Index Value Over Time', fontweight='bold')
    ax1.set_ylabel('Index Value (Points)', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)

    ax2.fill_between(df['trade_date'], df['rolling_vol_20'],
                     color='#e377c2', alpha=0.4, label='20-Day Rolling Volatility')
    ax2.plot(df['trade_date'], df['rolling_vol_20'], color='#e377c2',
             linewidth=0.8)
    ax2.set_title(
        'Feature 4 (GPR TARGET): 20-Day Rolling Volatility  [ std(log_return, 20d) ]',
        fontweight='bold'
    )
    ax2.set_xlabel('Trade Date', fontweight='bold')
    ax2.set_ylabel('Volatility (Log Return Std)', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    _save_fig(fig, 'feature_04_rolling_vol_20_TARGET.png')

    return df


# =================================================
# ROLLING ABSOLUTE CHANGE (SMOOTHED SWING MEASURE)
# =================================================
def compute_rolling_abs_change_14(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 5: rolling_abs_change_14  (replaces atr_14)
    -----------------------------------------------------
    Formula:
      abs_change_norm_t   = |absolute_change_t| / index_value_t
      rolling_abs_change_14_t = rolling_mean(abs_change_norm, 14 days)

    Context — Why Not ATR-14?
    --------------------------
    The Average True Range (Wilder, 1978) requires daily High and Low prices
    to measure intraday ranges and overnight gaps:
        TrueRange_t = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    Since the NEPSE_Index table contains only index_value and absolute_change
    (not High and Low), ATR cannot be computed directly.

    What rolling_abs_change_14 Captures
    ------------------------------------
    This feature is a 14-day rolling mean of the normalized absolute daily
    price change — a smoothed, multi-session swing magnitude measure that
    closely approximates what ATR-14 would capture from the close-to-close
    perspective:

      - High values → the market has been making large daily moves over
        the last 14 sessions → elevated volatility regime.
      - Low values → tight, compressed daily moves over 14 sessions
        → quiet / low-volatility regime.

    Relationship to ATR:
      In liquid markets with rare overnight gaps, ATR ≈ rolling mean of
      |close_t - close_{t-1}|, which is exactly what this feature computes
      (using index_value as the close proxy). For NEPSE, which is a daily
      settlement market with limited gap risk, this approximation is valid.

    NaN: The first row of abs_change_norm is 0 (no prior close). Rolling
    mean introduces NaN for the first 13 rows (needs 14 values). So the
    first ~14 rows will have NaN — far fewer than the 50-row global drop.
    """
    print("[Feature 5] Computing rolling_abs_change_14 (14-day smoothed price swing) ...")
    df = df.copy()

    # Step 1: Compute single-day normalized absolute change
    abs_norm = df['absolute_change'].abs() / df['index_value']

    # Step 2: Smooth over 14 days — equivalent to ATR's rolling mean of TrueRange
    df['rolling_abs_change_14'] = abs_norm.rolling(window=14, min_periods=14).mean()

    print(f"  rolling_abs_change_14 — first 5 valid values:\n"
          f"{df['rolling_abs_change_14'].dropna().head()}\n")
    print(f"  Descriptive stats:\n{df['rolling_abs_change_14'].describe()}\n")

    # ---- Visualization ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['trade_date'], df['rolling_abs_change_14'],
            color='#8c564b', linewidth=0.8, alpha=0.85)
    ax.fill_between(df['trade_date'], df['rolling_abs_change_14'],
                    color='#8c564b', alpha=0.2)
    ax.set_title(
        'Feature 5: 14-Day Rolling Absolute Change (ATR Proxy)\n'
        '[ 14-day mean of |absolute_change| / index_value ]',
        fontweight='bold'
    )
    ax.set_xlabel('Trade Date', fontweight='bold')
    ax.set_ylabel('Rolling |Change| / Index (fraction)', fontweight='bold')
    fig.tight_layout()
    _save_fig(fig, 'feature_05_rolling_abs_change_14.png')

    return df


# ====================
# BOLLINGER BAND WIDTH
# ====================
def compute_bb_width(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 6: bb_width
    --------------------
    Formula:
      middle_band_t   = MA20_t                    (20-day simple moving average)
      upper_band_t    = middle_band_t + 2 × std20_t
      lower_band_t    = middle_band_t - 2 × std20_t
      bb_width_t      = (upper_band - lower_band) / middle_band
                      = 4 × std20 / MA20

    Statistical Rationale
    ---------------------
    The Bollinger Band Width (BBW) was introduced by John Bollinger (1992) as a
    normalized measure of the band's expansion and contraction. It directly
    quantifies the ratio between recent realized volatility (std over 20 days)
    and the trend level (MA20):

      - **Volatility Squeeze (Low BBW):** When BBW drops to historic lows, the
        market is compressing. This often precedes a large breakout move — either
        bullish or bearish. In regime detection, squeezes mark pre-regime-shift
        states.
      - **Band Expansion (High BBW):** High BBW indicates that the market is
        experiencing high volatility and large directional moves. This corresponds
        to the "Turbulent" regime.

    Complementarity with Other Features:
      - `rolling_vol_20` captures absolute volatility magnitude.
      - `bb_width` captures relative volatility compared to the price trend,
        providing a normalized, trend-adjusted volatility signal that is
        particularly effective when prices are at extreme levels.

    NaN: First 19 rows are NaN (requires 20 points for MA and std).
    """
    print("[Feature 6] Computing bb_width ...")
    df = df.copy()

    middle_band = df['index_value'].rolling(window=20, min_periods=20).mean()
    std_20      = df['index_value'].rolling(window=20, min_periods=20).std()

    upper_band  = middle_band + 2 * std_20
    lower_band  = middle_band - 2 * std_20

    # Normalized bandwidth: (upper - lower) / middle = 4 * std20 / MA20
    df['bb_width'] = (upper_band - lower_band) / middle_band

    print(f"  bb_width — first 5 valid values:\n{df['bb_width'].dropna().head()}\n")
    print(f"  Descriptive stats:\n{df['bb_width'].describe()}\n")

    # ---- Visualization: Bollinger bands on price, bb_width below ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(df['trade_date'], df['index_value'], color='#1f77b4',
             linewidth=0.9, label='NEPSE Close', alpha=0.85)
    ax1.plot(df['trade_date'], middle_band, color='#ff7f0e',
             linewidth=1.0, label='MA-20 (Middle Band)', alpha=0.85)
    ax1.fill_between(df['trade_date'], upper_band, lower_band,
                     alpha=0.15, color='#aec7e8', label='Bollinger Band (±2σ)')
    ax1.set_title('NEPSE Index with Bollinger Bands (20-day, ±2σ)', fontweight='bold')
    ax1.set_ylabel('Index Value (Points)', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)

    ax2.plot(df['trade_date'], df['bb_width'], color='#17becf',
             linewidth=0.9, alpha=0.85)
    ax2.fill_between(df['trade_date'], df['bb_width'],
                     color='#17becf', alpha=0.2)
    ax2.set_title('Feature 6: Bollinger Band Width  [ (Upper - Lower) / Middle ]',
                  fontweight='bold')
    ax2.set_xlabel('Trade Date', fontweight='bold')
    ax2.set_ylabel('BB Width (fraction)', fontweight='bold')

    fig.tight_layout()
    _save_fig(fig, 'feature_06_bb_width.png')

    return df


# ============================================
# RELATIVE STRENGTH INDEX (RSI-14)
# ============================================
def compute_rsi_14(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 7: rsi_14
    ------------------
    Formula (Wilder's smoothed RSI):
      gain_t  = max(Close_t - Close_{t-1}, 0)
      loss_t  = max(Close_{t-1} - Close_t, 0)
      avg_gain_t = Wilder's smoothed average of gain over 14 periods
      avg_loss_t = Wilder's smoothed average of loss over 14 periods
      RS_t       = avg_gain_t / avg_loss_t
      RSI_t      = 100 - ( 100 / (1 + RS_t) )

    Wilder's smoothing is equivalent to an Exponential Moving Average (EMA)
    with alpha = 1/14, implemented here via pandas `.ewm(com=13, adjust=False)`.

    Statistical Rationale
    ---------------------
    RSI is a bounded [0, 100] momentum oscillator that captures the relative
    speed and magnitude of upward vs. downward price moves:

      - **RSI > 70:** Overbought territory — the market has risen sharply
        over 14 days. Often precedes a correction or regime transition.
      - **RSI < 30:** Oversold territory — sustained selling pressure, often
        following a market crash or panic (e.g., COVID-19 March 2020 in NEPSE).
      - **RSI ~ 50:** Neutral momentum, typical of the "Normal" regime.

    For GP modelling, RSI provides a bounded, non-price feature that captures
    medium-term momentum — a dimension not captured by volatility-based features
    (`rolling_vol_20`, `atr_14`, `bb_width`) or trend-based features
    (`normalized_trend_strength`).

    NaN: First 13 rows will have NaN (Wilder's smoothing requires 14 periods
    to initialize the exponential moving average).
    """
    print("[Feature 7] Computing rsi_14 ...")
    df = df.copy()

    delta = df['index_value'].diff()

    # Separate gains (positive deltas) and losses (positive magnitude)
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothed average: EMA with span = 2*N - 1 = 27, or com = N-1 = 13
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()

    rs            = avg_gain / avg_loss
    df['rsi_14']  = 100 - (100 / (1 + rs))

    print(f"  rsi_14 — first 5 valid values:\n{df['rsi_14'].dropna().head()}\n")
    print(f"  Descriptive stats:\n{df['rsi_14'].describe()}\n")

    # ---- Visualization: RSI with overbought/oversold reference lines ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['trade_date'], df['rsi_14'], color='#bcbd22',
            linewidth=0.9, alpha=0.85, label='RSI-14')
    ax.axhline(70, color='#d62728', linestyle='--', linewidth=0.9,
               alpha=0.7, label='Overbought (70)')
    ax.axhline(50, color='gray', linestyle='--', linewidth=0.7,
               alpha=0.5, label='Neutral (50)')
    ax.axhline(30, color='#2ca02c', linestyle='--', linewidth=0.9,
               alpha=0.7, label='Oversold (30)')
    ax.fill_between(df['trade_date'], df['rsi_14'], 70,
                    where=(df['rsi_14'] > 70),
                    color='#d62728', alpha=0.15, label='Overbought Zone')
    ax.fill_between(df['trade_date'], df['rsi_14'], 30,
                    where=(df['rsi_14'] < 30),
                    color='#2ca02c', alpha=0.15, label='Oversold Zone')
    ax.set_ylim(0, 100)
    ax.set_title('Feature 7: Relative Strength Index (RSI-14)\n'
                 'Wilder\'s EMA-smoothed Momentum Oscillator',
                 fontweight='bold')
    ax.set_xlabel('Trade Date', fontweight='bold')
    ax.set_ylabel('RSI (0–100)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=3)
    fig.tight_layout()
    _save_fig(fig, 'feature_07_rsi_14.png')

    return df


# ===================================
# MACD SIGNAL (HISTOGRAM)
# ===================================
def compute_macd_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 8: macd_signal  (MACD Histogram)
    -----------------------------------------
    Formula:
      EMA_12_t      = EMA( Close, span=12 )
      EMA_26_t      = EMA( Close, span=26 )
      MACD_t        = EMA_12_t - EMA_26_t         (MACD line)
      Signal_line_t = EMA( MACD, span=9 )         (Signal line)
      macd_signal_t = MACD_t - Signal_line_t       (Histogram / feature)

    Statistical Rationale
    ---------------------
    The Moving Average Convergence Divergence (Gerald Appel, 1979) histogram
    is the difference between the MACD line (short minus long EMA) and its own
    9-day exponential smoothing (the signal line). This captures the rate of
    change of momentum:

      - **Positive Histogram:** The MACD line is rising faster than its signal
        line — accelerating upward momentum. Common at the start of bull markets.
      - **Negative Histogram:** MACD line is falling — decelerating or reversing
        momentum. Common at the start of bear markets or corrections.
      - **Histogram crossing zero:** This is the classic momentum crossover
        signal — a leading indicator of regime transitions.

    Why the Histogram Instead of the MACD Line Itself?
      The MACD line alone is non-stationary for long price series (it trends
      with the price scale). The histogram (second derivative of price trend)
      oscillates around zero, is mean-reverting, and is stationary — making it
      more suitable as a GP feature. We normalize by Close to further suppress
      any residual scale dependence.

    NaN: First 25 rows will have significant NaN from EMA-26 warm-up.
    """
    print("[Feature 8] Computing macd_signal (MACD histogram) ...")
    df = df.copy()

    ema_12     = df['index_value'].ewm(span=12, adjust=False).mean()
    ema_26     = df['index_value'].ewm(span=26, adjust=False).mean()
    macd_line  = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    # Store histogram — normalized by Close for scale-invariance
    df['macd_signal'] = (macd_line - signal_line) / df['index_value']

    print(f"  macd_signal — first 5 valid values:\n{df['macd_signal'].dropna().head()}\n")
    print(f"  Descriptive stats:\n{df['macd_signal'].describe()}\n")

    # ---- Visualization: MACD components + histogram ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(df['trade_date'], macd_line  / df['index_value'],
             color='#1f77b4', linewidth=0.9, label='MACD Line (normalized)', alpha=0.85)
    ax1.plot(df['trade_date'], signal_line / df['index_value'],
             color='#ff7f0e', linewidth=1.0, label='Signal Line (normalized)', alpha=0.85)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax1.set_title('MACD Line vs Signal Line (normalized by Close)', fontweight='bold')
    ax1.set_ylabel('Value (fraction)', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)

    colors = ['#2ca02c' if v >= 0 else '#d62728'
              for v in df['macd_signal']]
    ax2.bar(df['trade_date'], df['macd_signal'],
            color=colors, alpha=0.7, width=1.5, label='MACD Histogram (feature)')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax2.set_title('Feature 8: MACD Signal (Histogram)  [ MACD - Signal / Close ]',
                  fontweight='bold')
    ax2.set_xlabel('Trade Date', fontweight='bold')
    ax2.set_ylabel('Histogram (fraction)', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    _save_fig(fig, 'feature_08_macd_signal.png')

    return df


# ====================================================
# TURNOVER SPIKE (LIQUIDITY EVENT DETECTOR)
# ====================================================
def compute_turnover_spike(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 9: turnover_spike  (replaces volume_spike)
    ----------------------------------------------------
    Formula : turnover_spike_t = turnover_t / rolling_mean(turnover, 20)_t

    Context — Why Not volume_spike?
    --------------------------------
    The NEPSE_Index table does not include a share-volume column (shares
    traded per session). Only `turnover` (total NPR value traded) is
    available. Turnover = price × shares, so it is a value-weighted proxy
    for trading volume that captures both the quantity traded and the
    price level at which it was executed.

    What turnover_spike Captures
    -----------------------------
    This ratio compares the current session's turnover to the 20-day rolling
    mean of turnover — exactly the same logic as a volume spike ratio:

      - **ratio >> 1:** Unusually high capital flow → major market event,
        institutional activity, or panic/euphoria. Frequently marks the
        beginning or end of a volatility regime.
      - **ratio ≈ 1:** Normal participation level for the current era.
      - **ratio < 1:** Below-normal activity → quiet, low-liquidity session
        characteristic of the Quiet regime.

    Why Ratio Over Z-Score?
      Dividing by the rolling mean produces a ratio that is bounded below
      at 0 (turnover cannot be negative) and is intuitive: 2.0 = double
      the recent average. A Z-score would produce negative values for
      below-average sessions, which is harder to interpret for liquidity.

    Handling Zero-Turnover Early Data:
      NEPSE had near-zero turnover in many early sessions (2000–2006),
      with the rolling mean also near zero in those windows. To prevent
      division-by-zero, we apply a floor of 1 NPR on the rolling mean
      before dividing. This converts the ratio to 0 (rather than inf/NaN)
      for those early sessions, which is a correct representation of the
      quiet frontier-market era.

    NaN: First 19 rows will have NaN from the 20-day rolling window.
    """
    print("[Feature 9] Computing turnover_spike (turnover / 20-day rolling mean) ...")

    if 'turnover' not in df.columns:
        print("  WARNING: 'turnover' column not found. Setting turnover_spike = NaN.\n")
        df['turnover_spike'] = np.nan
        return df

    df = df.copy()

    # Floor the rolling mean at 1 NPR to avoid division by zero
    # (valid because turnover cannot physically be less than 1 NPR per session)
    rolling_mean = df['turnover'].rolling(window=20, min_periods=20).mean()
    rolling_mean_safe = rolling_mean.clip(lower=1.0)

    df['turnover_spike'] = df['turnover'] / rolling_mean_safe

    print(f"  turnover_spike — first 5 valid values:\n"
          f"{df['turnover_spike'].dropna().head()}\n")
    print(f"  Descriptive stats:\n{df['turnover_spike'].describe()}\n")

    # ---- Visualization ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['trade_date'], df['turnover_spike'], color='#7f7f7f',
            linewidth=0.7, alpha=0.8, label='Turnover Spike Ratio')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.7,
               label='Baseline (ratio = 1.0)')
    ax.fill_between(df['trade_date'], df['turnover_spike'], 1.0,
                    where=(df['turnover_spike'] > 1.0),
                    color='#1f77b4', alpha=0.2, label='Above-Average Turnover')
    ax.set_title(
        'Feature 9: Turnover Spike Ratio  [ Turnover / 20-day Rolling Mean Turnover ]',
        fontweight='bold'
    )
    ax.set_xlabel('Trade Date', fontweight='bold')
    ax.set_ylabel('Spike Ratio (dimensionless)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    _save_fig(fig, 'feature_09_turnover_spike.png')

    return df


# ===============
# TURNOVER GROWTH
# ===============
def compute_turnover_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 10: turnover_growth
    ----------------------------
    Formula : turnover_growth_t = (Turnover_t - Turnover_{t-1}) / Turnover_{t-1}
                                = Turnover.pct_change()  [with zero-safe denominator]

    Statistical Rationale
    ---------------------
    Turnover (total NPR value of shares traded) measures market liquidity and
    capital flow intensity. Taking the percentage change transforms the
    non-stationary, exponentially growing turnover series into a stationary,
    zero-mean growth rate:
      - Positive growth → rising capital inflow; often precedes or accompanies
        the beginning of a bullish regime.
      - Negative growth → falling participation; common in early bear markets
        or Quiet regime periods.
      - Near-zero growth → stable liquidity; characteristic of the Normal regime.

    Handling Zero-Turnover Early Data (Critical Fix)
    -------------------------------------------------
    The Day-1 EDA revealed that NEPSE had zero (or near-zero) turnover
    in many sessions before approximately 2008. Standard pct_change() computes:
        (new - old) / old
    When old = 0, this produces inf (division by zero), and when both are 0,
    it produces NaN. This caused 3,804 of 6,033 rows to be NaN — rendering
    the feature unusable without correction.

    Fix: We use a zero-safe denominator by replacing 0 with 1 NPR before
    dividing. When turnover is genuinely 0 (no trading), the growth rate is
    0 — a correct representation of no liquidity change. This approach:
      - Preserves all 6,032 valid rows (vs. only 2,229 before the fix)
      - Correctly represents the zero-liquidity frontier market period
      - Avoids the need to drop entire early-period data

    Outlier Clipping:
      Transitions from near-zero to measurable turnover (e.g., 1 NPR → 50M NPR)
      still produce extreme ratios. Values beyond ±500% (5x) are winsorized.
      We report this clipping decision in the paper.
    """
    print("[Feature 10] Computing turnover_growth (zero-safe pct_change) ...")
    df = df.copy()

    if 'turnover' not in df.columns:
        print("  WARNING: 'turnover' column not found. Setting turnover_growth = NaN.\n")
        df['turnover_growth'] = np.nan
        return df

    # Zero-safe percentage change:
    # Replace 0 in the denominator (prior turnover) with 1 NPR to avoid inf/NaN
    prior_turnover = df['turnover'].shift(1)
    prior_turnover_safe = prior_turnover.where(prior_turnover != 0, other=1.0)
    df['turnover_growth'] = (df['turnover'] - prior_turnover) / prior_turnover_safe

    # Clip extreme outliers (frontier-market transition spikes)
    clip_threshold = 5.0   # 500% = 5× change
    n_clipped = (df['turnover_growth'].abs() > clip_threshold).sum()
    if n_clipped > 0:
        df['turnover_growth'] = df['turnover_growth'].clip(
            lower=-clip_threshold, upper=clip_threshold
        )
        print(f"  NOTE: {n_clipped} extreme outlier rows clipped to "
              f"±{clip_threshold} (500%) for numerical stability.")

    valid_count = df['turnover_growth'].notna().sum()
    print(f"  turnover_growth — {valid_count:,} valid rows (was 2,229 before zero-safe fix)")
    print(f"  First 5 valid values:\n{df['turnover_growth'].dropna().head()}\n")
    print(f"  Descriptive stats:\n{df['turnover_growth'].describe()}\n")

    # ---- Visualization ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['trade_date'], df['turnover_growth'], color='#2ca02c',
            linewidth=0.7, alpha=0.8)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.fill_between(df['trade_date'], df['turnover_growth'],
                    where=(df['turnover_growth'] > 0),
                    color='#2ca02c', alpha=0.2, label='Positive Growth')
    ax.fill_between(df['trade_date'], df['turnover_growth'],
                    where=(df['turnover_growth'] < 0),
                    color='#d62728', alpha=0.2, label='Negative Growth')
    ax.set_title(
        'Feature 10: Turnover Growth  [ zero-safe pct_change(Turnover) ]',
        fontweight='bold'
    )
    ax.set_xlabel('Trade Date', fontweight='bold')
    ax.set_ylabel('Turnover % Change (fraction)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    _save_fig(fig, 'feature_10_turnover_growth.png')

    return df


# ======================================
# FEATURE SUMMARY & CORRELATION HEATMAP
# =====================================
def generate_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Final Step: Feature Summary and Correlation Heatmap
    ----------------------------------------------------
    After all 10 features are computed, this function produces:
      1. A summary table showing NaN counts per feature (verifying that
         rolling lookback NaN patterns are as expected).
      2. A high-quality correlation heatmap of all 10 features using
         Pearson correlation coefficients.

    Scientific Rationale for the Heatmap
    -------------------------------------
    The heatmap serves two purposes in the paper:

    A) **Multicollinearity Detection:**
       Features with |r| > 0.90 are near-perfectly collinear. Including both
       in a GP model would not add information but would destabilize the
       kernel hyperparameter optimization (gradient directions can conflict in
       a highly collinear feature space). We explicitly dropped `daily_return`
       for this reason (r ≈ 0.999 with `log_return`).

    B) **Complementarity Verification:**
       Ideally, the 10 features should span multiple dimensions of market
       information. If they cluster into distinct correlation groups (e.g.,
       volatility features correlated with each other; momentum features
       correlated with each other), that is evidence that our feature set
       captures multiple market aspects — desirable for GP prediction.

    The heatmap figure is Paper-Quality: it goes directly into the
    "Experimental Setup" section of the scientific paper.
    """
    print("\n[Summary] Feature NaN counts:")
    feature_cols = [
        'log_return', 'abs_change_norm', 'normalized_trend_strength',
        'rolling_vol_20', 'rolling_abs_change_14', 'bb_width', 'rsi_14',
        'macd_signal', 'turnover_spike', 'turnover_growth'
    ]

    # Only include columns that were successfully created
    available_features = [c for c in feature_cols if c in df.columns]
    missing_features   = set(feature_cols) - set(available_features)
    if missing_features:
        print(f"  WARNING: The following features are missing and will be "
              f"excluded from heatmap: {missing_features}")

    nan_report = df[available_features].isnull().sum()
    print(nan_report.to_string())
    print(f"\n  Total rows in DataFrame: {len(df)}")

    # ---- Correlation Heatmap ----
    # Compute Pearson correlation on rows with no NaN (drop NaN rows for calculation)
    corr_matrix = df[available_features].dropna().corr(method='pearson')

    # Create annotation array formatted to 2 decimal places
    annot_matrix = corr_matrix.round(2)

    fig, ax = plt.subplots(figsize=(13, 10))

    mask = np.zeros_like(corr_matrix, dtype=bool)  # No masking — show full matrix
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        annot=annot_matrix,
        fmt='.2f',
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'shrink': 0.75, 'label': 'Pearson Correlation Coefficient'},
        ax=ax
    )

    ax.set_title(
        'Pearson Correlation Matrix — 10 Engineered Features\n'
        '(NEPSE Market Regime Detection, Days 2–3)',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.tight_layout()
    _save_fig(fig, 'feature_correlation_heatmap.png')

    print("\n  Correlation heatmap saved.")
    print("\n  Top 5 strongest positive correlations among features:")
    corr_pairs = (
        corr_matrix
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    print(corr_pairs.head())

    print("\n  Top 5 strongest negative correlations among features:")
    print(corr_pairs.tail())


# =============================================
# MAIN ORCHESTRATOR — run_feature_engineering()
# =============================================
def run_feature_engineering(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Master function for Days 2–3 Feature Engineering.

    Accepts the chronologically sorted DataFrame produced by
    `day_1_eda.run_day_1_eda()` and sequentially applies all 10 feature
    computations in the correct dependency order:
      - log_return MUST be computed before rolling_vol_20 (depends on it).
      - All other features only require the raw OHLCV columns.

    Parameters
    ----------
    df_sorted : pd.DataFrame
        The cleaned, sorted DataFrame from Day 1. Must include columns:
        'trade_date', 'index_value', 'turnover', and ideally 'high', 'low',
        'volume' or 'shares_traded' (volume-related features gracefully degrade
        to NaN if the column is absent).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with 10 new feature columns appended.
        No rows are dropped — NaN handling is deferred to Day 4.
    """
    print("\n" + "=" * 60)
    print("  PHASE A — DAYS 2-3: FEATURE ENGINEERING")
    print("=" * 60)

    print(f"\n  Input DataFrame: {df_sorted.shape[0]} rows × "
          f"{df_sorted.shape[1]} columns")
    print(f"  Available columns: {list(df_sorted.columns)}\n")

    # -----------------------------------------------------------------------
    # Apply features in dependency order
    # log_return → rolling_vol_20 (rolling_vol_20 depends on log_return)
    # All others are independent of each other
    # -----------------------------------------------------------------------
    df = df_sorted.copy()

    df = compute_log_return(df)                   # Feature 1 — must be FIRST
    df = compute_abs_change_norm(df)              # Feature 2 (proxy for hl_spread)
    df = compute_normalized_trend_strength(df)    # Feature 3
    df = compute_rolling_vol_20(df)               # Feature 4 — GPR TARGET (depends on 1)
    df = compute_rolling_abs_change_14(df)        # Feature 5 (proxy for atr_14)
    df = compute_bb_width(df)                     # Feature 6
    df = compute_rsi_14(df)                       # Feature 7
    df = compute_macd_signal(df)                  # Feature 8
    df = compute_turnover_spike(df)               # Feature 9 (proxy for volume_spike)
    df = compute_turnover_growth(df)              # Feature 10 (with zero-safe fix)

    # -----------------------------------------------------------------------
    # Post-engineering summary and correlation heatmap
    # -----------------------------------------------------------------------
    generate_correlation_heatmap(df)

    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING COMPLETE")
    print(f"  Output DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
    print("  New feature columns added:")
    new_cols = [
        'log_return', 'abs_change_norm', 'normalized_trend_strength',
        'rolling_vol_20', 'rolling_abs_change_14', 'bb_width', 'rsi_14',
        'macd_signal', 'turnover_spike', 'turnover_growth'
    ]
    for i, col in enumerate(new_cols, 1):
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"    {i:2d}. {col:<35} [{non_null:,} valid rows]")
    print("=" * 60)
    print("\n  -> Proceed to Day 4: Preprocessing, Scaling, and Train/Test Split")

    return df


# =======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    # Import Day-1 module to obtain the sorted DataFrame
    # Adjust the sys.path so Python can find day_1_eda.py in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from day_1_eda import run_day_1_eda  # noqa: E402

    # Step 1: Run Day-1 EDA to get the cleaned, sorted DataFrame
    df_day1 = run_day_1_eda()

    # Step 2: Run Days 2–3 Feature Engineering
    df_featured = run_feature_engineering(df_day1)

    print("\nFinal DataFrame preview (last 5 rows):")
    print(df_featured.tail())
    
    # Save the dataframe to a CSV file
    csv_path = os.path.join(script_dir, "nepse_featured_data.csv")
    df_featured.to_csv(csv_path, index=False)
    print(f"\n[Saved] Engineered feature dataset saved to: {csv_path}")
