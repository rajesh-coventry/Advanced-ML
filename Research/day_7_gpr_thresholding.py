# d:/Disk-E/Advanced-ML/Research/day_7_gpr_thresholding.py

"""GPR Thresholding & Gaussian Process Classification (GPC)

This uses Matern GPR model. It:
1. Loads the pre‑processed feature matrix and the volatility target.
2. Derives discrete regime labels from the continuous volatility using the
   same percentile thresholds defined in Day 4 (15th & 85th percentiles).
3. Trains a `GaussianProcessClassifier` on the training set.
4. Evaluates classification performance on the test set (accuracy, confusion
   matrix, classification report).
5. Produces two key figures:
   * Figure6 - Confusion matrix heat‑map.
   * Figure7-  Predicted regime timeline (classification) over the test
     period, overlaid with the true volatility regime.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# ----------------
# Helper utilities
# ----------------

def _save_fig(fig, filename):
    """Save a Matplotlib figure to the project's ``figures`` folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [Saved] {path}")

def _regime_labels(vol_series, low_q=0.15, high_q=0.85):
    """Convert a continuous volatility series into three discrete regimes.

    * 0 – Low volatility (<= low percentile)
    * 1 – Medium volatility (between low and high percentiles)
    * 2 – High volatility (>= high percentile)
    """
    low_thr = np.quantile(vol_series, low_q)
    high_thr = np.quantile(vol_series, high_q)
    labels = np.where(vol_series <= low_thr, 0,
                     np.where(vol_series >= high_thr, 2, 1))
    return labels, low_thr, high_thr

# ---------
# workflow
# ---------

def run_day_7_gpc():
    print("\n" + "=" * 60)
    print("  GPR THRESHOLDING & GPC CLASSIFICATION")
    print("=" * 60)

    # 1. Load pre‑processed data (same as Day 5/6)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data_splits", "day_4_processed_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run day_4_preprocessing.py first.")
    print(f"[Loading Data] {data_path}")
    data = np.load(data_path, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train_vol = data["y_train_reg"]  # continuous volatility
    y_test_vol = data["y_test_reg"]
    train_dates = pd.to_datetime(data["train_dates"])
    test_dates = pd.to_datetime(data["test_dates"])
    feature_names = data["feature_names"]

    # 2. Derive discrete regime labels from volatility
    y_train_cls, low_thr, high_thr = _regime_labels(y_train_vol)
    y_test_cls, _, _ = _regime_labels(y_test_vol, low_q=0.15, high_q=0.85)
    print(f"[Regime Thresholds] low={low_thr:.5f}, high={high_thr:.5f}")
    print(f"  Class distribution (train): {np.bincount(y_train_cls)}")
    print(f"  Class distribution (test) : {np.bincount(y_test_cls)}")

    # 3. Optional: standardise features (helps GPC convergence)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # 4. Define GPC kernel – we reuse a simple RBF + WhiteKernel
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    kernel += WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    gpc = GaussianProcessClassifier(
        kernel=kernel,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=5,
        random_state=42,
        max_iter_predict=1000,
    )

    print("\n[Training GPC] Starting optimisation...")
    start = time.time()
    gpc.fit(X_train_std, y_train_cls)
    fit_time = time.time() - start
    print(f"  Training completed in {fit_time:.2f} seconds.")
    print(f"  Optimized kernel: {gpc.kernel_}")

    # 5. Predict on test set
    print("\n[Predicting] Generating class predictions...")
    y_test_pred = gpc.predict(X_test_std)
    y_test_proba = gpc.predict_proba(X_test_std)

    # 6. Evaluation metrics
    acc = accuracy_score(y_test_cls, y_test_pred)
    cm = confusion_matrix(y_test_cls, y_test_pred)
    report = classification_report(y_test_cls, y_test_pred, target_names=["Low", "Medium", "High"], output_dict=True)
    print(f"  Test Accuracy: {acc:.4f}")

    # 7. Figure 6 – Confusion matrix heat‑map
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"], ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title("Figure 6 – Confusion Matrix (GPC)")
    _save_fig(fig_cm, "day7_gpc_confusion_matrix.png")
    plt.close(fig_cm)

    # 8. Predicted regime timeline vs true volatility regime
    # Convert true volatility to regime for visual overlay (same thresholds)
    true_regime = y_test_cls
    pred_regime = y_test_pred
    fig_ts, ax_ts = plt.subplots(figsize=(14, 4))
    # Plot true regime as stepped line
    ax_ts.step(test_dates, true_regime, where="post", label="True Regime", linewidth=1.5)
    # Plot predicted regime
    ax_ts.step(test_dates, pred_regime, where="post", label="Predicted Regime", linewidth=1.5, alpha=0.8)
    ax_ts.set_yticks([0, 1, 2])
    ax_ts.set_yticklabels(["Low", "Medium", "High"])
    ax_ts.set_xlabel("Date")
    ax_ts.set_title("Figure 7 – Predicted Regime Timeline (Test Set)")
    ax_ts.legend(loc="upper right")
    ax_ts.grid(alpha=0.3)
    _save_fig(fig_ts, "day7_gpc_regime_timeline.png")
    plt.close(fig_ts)

    # 9. Persist results (compact dict, no heavy objects)
    out_dir = os.path.join(script_dir, "models")
    os.makedirs(out_dir, exist_ok=True)
    results = {
        "kernel": str(gpc.kernel_),
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "fit_time_seconds": fit_time,
        "thresholds": {"low": low_thr, "high": high_thr},
    }
    results_path = os.path.join(out_dir, "gpc_regime_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n[Saved] GPC results saved to {results_path}")

    print("\n" + "=" * 60)
    print("  TASK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_day_7_gpc()
