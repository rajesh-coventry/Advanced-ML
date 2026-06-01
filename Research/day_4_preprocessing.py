import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def _save_fig(fig, filename):
    """Helper to save figures into the Research/figures directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    filepath = os.path.join(fig_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  [Saved] {filepath}")

def run_day_4_preprocessing(csv_path: str):
    """
    Day 4: Preprocessing, target definition, and train/test split.
    
    Parameters
    ----------
    csv_path : str
        Path to the 'nepse_featured_data.csv' generated in Days 2-3.
    """
    print("\n" + "=" * 60)
    print("  PHASE A — DAY 4: PREPROCESSING & SPLIT")
    print("=" * 60)

    # Load the data
    df = pd.read_csv(csv_path)
    # Ensure trade_date is datetime
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    print(f"\n[Loaded Data] {df.shape[0]} rows, {df.shape[1]} columns")

    # ==============
    # DROP NaN ROWS
    # ==============
    print("\n[Cell 1] Dropping NaN rows...")
    # The max lookback window was 50 days (MA50 for normalized_trend_strength).
    # We drop the first 55 rows to ensure a completely clean start without NaNs.
    df_clean = df.iloc[55:].copy()
    
    # Verify no NaNs remain in the features
    feature_cols = [
        'log_return', 'abs_change_norm', 'normalized_trend_strength',
        'rolling_abs_change_14', 'bb_width', 'rsi_14',
        'macd_signal', 'turnover_spike', 'turnover_growth'
    ]
    
    nan_count = df_clean[feature_cols + ['rolling_vol_20']].isna().sum().sum()
    print(f"  Rows after dropping first 55: {df_clean.shape[0]}")
    print(f"  Total NaNs remaining in feature/target columns: {nan_count}")
    
    if nan_count > 0:
        print("  WARNING: NaNs still present. Dropping remaining NaN rows.")
        df_clean.dropna(subset=feature_cols + ['rolling_vol_20'], inplace=True)
        print(f"  Rows after full dropna: {df_clean.shape[0]}")

    # ===================
    # DEFINE GPR TARGET
    # ===================
    print("\n[Cell 2] Defining GPR Target...")
    df_clean['y_regression'] = df_clean['rolling_vol_20']
    
    # Verification Plot: Target over time
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_clean['trade_date'], df_clean['y_regression'], color='#d62728', linewidth=0.8)
    ax.set_title('GPR Target Variable: 20-Day Rolling Volatility', fontweight='bold')
    ax.set_xlabel('Trade Date', fontweight='bold')
    ax.set_ylabel('Volatility (Std of Log Returns)', fontweight='bold')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, 'day4_target_gpr_volatility.png')

    # ========================================
    # DEFINE VOLATILITY REGIME LABELS (FOR GPC)
    # ========================================
    print("\n[Cell 3] Defining Volatility Regime Labels...")
    p50 = df_clean['y_regression'].quantile(0.50)
    p85 = df_clean['y_regression'].quantile(0.85)
    
    print(f"  Thresholds based on rolling_vol_20:")
    print(f"    50th Percentile: {p50:.6f}")
    print(f"    85th Percentile: {p85:.6f}")
    
    def assign_regime(vol):
        if vol <= p50:
            return 0  # Quiet
        elif vol <= p85:
            return 1  # Normal
        else:
            return 2  # Turbulent
            
    df_clean['y_regime'] = df_clean['y_regression'].apply(assign_regime)
    
    # Print class distribution
    counts = df_clean['y_regime'].value_counts().sort_index()
    percentages = (counts / len(df_clean)) * 100
    print("\n  Class Distribution:")
    for regime, count in counts.items():
        name = "Quiet (0)" if regime == 0 else "Normal (1)" if regime == 1 else "Turbulent (2)"
        print(f"    {name:<15}: {count} rows ({percentages[regime]:.1f}%)")

    # Plot regime distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_clean, x='y_regime', palette=['#2ca02c', '#1f77b4', '#d62728'], ax=ax)
    ax.set_title('Volatility Regime Class Distribution', fontweight='bold')
    ax.set_xlabel('Regime (0: Quiet, 1: Normal, 2: Turbulent)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    fig.tight_layout()
    _save_fig(fig, 'day4_regime_distribution.png')

    # Plot Volatility Target with Regime Colored
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_clean['trade_date'], df_clean['y_regression'], color='gray', linewidth=0.5, zorder=1)
    scatter = ax.scatter(df_clean['trade_date'], df_clean['y_regression'], 
                         c=df_clean['y_regime'], cmap='coolwarm', s=5, zorder=2)
    
    # Adding horizontal lines for thresholds
    ax.axhline(p50, color='blue', linestyle='--', linewidth=1, label=f'50th Pct ({p50:.4f})')
    ax.axhline(p85, color='red', linestyle='--', linewidth=1, label=f'85th Pct ({p85:.4f})')
    
    ax.set_title('Volatility Regimes Mapped onto Target Variable', fontweight='bold')
    ax.set_xlabel('Trade Date', fontweight='bold')
    ax.set_ylabel('Rolling Volatility', fontweight='bold')
    ax.legend(loc='upper left')
    fig.tight_layout()
    _save_fig(fig, 'day4_volatility_regimes_scatter.png')

    # ===================================================
    # CELL 4 — FEATURE MATRIX, SCALING, AND TEMPORAL SPLIT
    # ===================================================
    print("\n[Cell 4] Temporal Train/Test Split & Scaling...")
    
    X = df_clean[feature_cols].copy()
    y_reg = df_clean['y_regression'].values
    y_clf = df_clean['y_regime'].values
    
    # Temporal split 80/20
    split_idx = int(len(X) * 0.8)
    
    X_train_raw = X.iloc[:split_idx]
    X_test_raw  = X.iloc[split_idx:]
    
    y_train_reg = y_reg[:split_idx]
    y_test_reg  = y_reg[split_idx:]
    
    y_train_clf = y_clf[:split_idx]
    y_test_clf  = y_clf[split_idx:]
    
    train_dates = df_clean['trade_date'].iloc[:split_idx]
    test_dates  = df_clean['trade_date'].iloc[split_idx:]
    
    print(f"  Training set size: {len(X_train_raw)} rows (80%)")
    print(f"  Test set size:     {len(X_test_raw)} rows (20%)")
    print(f"  Train period:      {train_dates.min().date()} to {train_dates.max().date()}")
    print(f"  Test period:       {test_dates.min().date()} to {test_dates.max().date()}")
    
    # Standard Scaling
    scaler = StandardScaler()
    # Fit only on training set to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)
    
    # Convert back to DataFrame for easy saving
    X_train = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train_raw.index)
    X_test  = pd.DataFrame(X_test_scaled,  columns=feature_cols, index=X_test_raw.index)
    
    # Save the processed data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, 'data_splits')
    os.makedirs(processed_dir, exist_ok=True)
    
    # We will save everything in a single .npz file for easy loading in Days 5-8
    out_path = os.path.join(processed_dir, 'day_4_processed_data.npz')
    np.savez(
        out_path,
        X_train=X_train.values,
        X_test=X_test.values,
        y_train_reg=y_train_reg,
        y_test_reg=y_test_reg,
        y_train_clf=y_train_clf,
        y_test_clf=y_test_clf,
        feature_names=np.array(feature_cols),
        train_dates=train_dates.values,
        test_dates=test_dates.values
    )
    print(f"\n[Saved] Train/Test splits saved to: {out_path}")
    print("  Access variables using np.load(path)['X_train'], etc.")
    
    print("\n" + "=" * 60)
    print("  DAY 4 PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg,
        'y_train_clf': y_train_clf,
        'y_test_clf': y_test_clf,
        'train_dates': train_dates,
        'test_dates': test_dates
    }

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "nepse_featured_data.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run day_2_3_feature_engineering.py first.")
        sys.exit(1)
        
    run_day_4_preprocessing(csv_file)
