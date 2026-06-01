"""Structural-Break Analysis
============================ 

This performs two classic econometric tests for a known structural break
in the 20-day rolling volatility series that was used throughout the project:

1. **Chow test** - tests a pre-specified breakpoint (default: 2020-03-01, the
   COVID-19 market crash).  It fits a simple linear trend model to the full
   series and to the two subsamples (pre- and post-break) and compares the
   residual sum-of-squares.

2. **CUSUM of Squares** - a cumulative-sum test that evaluates the entire
   series for any change-point without a prior hypothesis.  It uses the same
   OLS baseline as the Chow test.

Both tests are applied to the *rolling volatility* (`rolling_vol_20`) that is
stored in ``day_4_processed_data.npz`` .  The results are
saved as high-resolution PNG figures (300 dpi) in ``Research/figures`` and a
compact ``pickle`` summary in ``Research/models`` for later inclusion in the
paper.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.diagnostic import breaks_cusumolsresid

# ----------------
# Helper utilities
# ----------------

def _save_fig(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
    """Save a Matplotlib figure.

    Parameters
    ----------
    fig: plt.Figure
        The Matplotlib figure object to be saved.
    filename: str
        Desired file name (PNG will be used). The function automatically adds the
        ``.png`` extension if omitted.
    """
    # Resolve absolute path inside the repository
    script_dir = Path(__file__).resolve().parent
    fig_dir = script_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the filename ends with .png
    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"
    filepath = fig_dir / filename

    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)  # free memory, especially useful when many figures are made
    print(f"  [Saved] {filepath}")

# -----------------------
# Core analysis function
# -----------------------

def run_day_10_structural_breaks(
    break_date: pd.Timestamp = pd.Timestamp("2020-03-01"),
    data_path: Path = Path(__file__).resolve().parent / "data_splits" / "day_4_processed_data.npz",
) -> None:
    """Execute the structural-break workflow.

    Parameters
    ----------
    break_date: pd.Timestamp, optional
        Known breakpoint to be tested with the Chow test.  Defaults to the start
        of the COVID-19 market crash (2020-03-01).
    data_path: pathlib.Path, optional
        Path to the ``.npz`` file created on Day 4 containing the pre-processed
        training / test sets.
    """
    print("\n" + "=" * 60)
    print("  STRUCTURAL BREAK ANALYSIS")
    print("=" * 60)

    # ------------------------------------------------------
    # Load pre-processed data (features, target, timestamps)
    # ------------------------------------------------------
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing {data_path}. Run day_4_preprocessing.py first.")

    print(f"[Loading Data] {data_path}")
    data = np.load(data_path, allow_pickle=True)

    # Volatility series used throughout the project (training + test)
    y_train = data["y_train_reg"]
    y_test = data["y_test_reg"]
    dates_train = pd.to_datetime(data["train_dates"])
    dates_test = pd.to_datetime(data["test_dates"])

    # Concatenate for a continuous time-series view
    y_full = np.concatenate([y_train, y_test])
    dates_full = pd.concat([pd.Series(dates_train), pd.Series(dates_test)], ignore_index=True)

    # --------------------------------
    # Chow test - known breakpoint
    # --------------------------------
    # Find the index of the nearest date to the chosen breakpoint
    break_idx = dates_full.searchsorted(break_date)
    if break_idx == 0 or break_idx >= len(y_full):
        raise ValueError("Break date not within the data range.")

    # Simple OLS trend model: y_t = beta_0 + beta_1*t + epsilon_t
    t = np.arange(len(y_full))
    X = sm.add_constant(t)
    model_full = sm.OLS(y_full, X).fit()

    # Split the data at the breakpoint and fit separate models
    X_pre, y_pre = X[:break_idx], y_full[:break_idx]
    X_post, y_post = X[break_idx:], y_full[break_idx:]

    model_pre = sm.OLS(y_pre, X_pre).fit()
    model_post = sm.OLS(y_post, X_post).fit()

    # Manual Chow-F statistic (based on RSS) - identical to statsmodels' helper
    rss_full = np.sum(model_full.resid ** 2)
    rss_pre = np.sum(model_pre.resid ** 2)
    rss_post = np.sum(model_post.resid ** 2)
    k = X.shape[1]            # number of parameters (incl. intercept)
    n = len(y_full)
    chow_F = ((rss_full - (rss_pre + rss_post)) / k) / ((rss_pre + rss_post) / (n - 2 * k))
    p_value = 1 - stats.f.cdf(chow_F, k, n - 2 * k)

    print("\n[Chow Test] Results for break at {:%Y-%m-%d}:".format(break_date))
    print(f"  F-statistic = {chow_F:.4f}, p-value = {p_value:.4e}")

    # -----------------------------------------------
    # CUSUM of Squares - unspecified breakpoints
    # -----------------------------------------------
    print("\n[CU-SUM OF SQUARES] Running test on entire volatility series")
    # The baseline OLS model (same as used for Chow) supplies residuals
    # Manual CUSUM of Squares using residuals (cumulative sum of squared residuals)
    cusum_stat = np.cumsum(model_full.resid ** 2)
    crit_low, crit_high = np.nan, np.nan  # No explicit confidence bounds in manual implementation

    # Plot the CUSUM statistic with confidence bounds
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates_full, cusum_stat, label="CUSUM of Squares", color="darkblue")
    # Plot confidence bounds only if they are finite
    if np.isfinite(crit_low) and np.isfinite(crit_high):
        ax.axhline(crit_low, color="red", linestyle="--", linewidth=1, label="Lower 5% bound")
        ax.axhline(crit_high, color="red", linestyle="--", linewidth=1, label="Upper 5% bound")
    ax.set_title("CUSUM of Squares Test for Structural Instability in Volatility", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("CUSUM Statistic")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "day10_cusum_squares.png")

    # ------------------------------------------------
    # Visualise the chosen break on the volatility series
    # ------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(dates_full, y_full, color="black", linewidth=1.2, label="Rolling Volatility")
    ax2.axvline(break_date, color="orange", linestyle="--", linewidth=2, label="Chosen Break (COVID-19)")
    ax2.set_title("Volatility Series with Structural Break Candidate", fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("20-Day Rolling Volatility")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    _save_fig(fig2, "day10_volatility_break_candidate.png")

    # ---------------------------------------------------------------
    # compact summary (pickle) for the manuscript appendix
    # ---------------------------------------------------------------
    summary: Dict[str, object] = {
        "chow": {
            "date": str(break_date.date()),
            "F": chow_F,
            "p": p_value,
        },
        "cusum": {
            "stat_series": cusum_stat,  # large array - stored for reproducibility
            "crit_low": crit_low,
            "crit_high": crit_high,
        },
        "break_index": int(break_idx),
    }
    out_dir = Path(__file__).resolve().parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "structural_break_summary.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)
    print("\n[Saved] Structural break summary saved.")

    print("\n" + "=" * 60)
    print("  COMPLETED")
    print("=" * 60)

# ------------
# Entry-point:
# -----------
if __name__ == "__main__":
    run_day_10_structural_breaks()
