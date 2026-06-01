import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.patches import Patch

def _save_fig(fig, filename):
    """Helper to save figures into the Research/figures directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    filepath = os.path.join(fig_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  [Saved] {filepath}")

def run_day_8_gpc():
    """
    Native Gaussian Process Classification (GPC).
    
    This phase trains a proper probabilistic GPC using the Laplace approximation
    for the posterior, using the best kernel identified in the GPR phase (Matern).
    """
    print("\n" + "=" * 60)
    print("  GAUSSIAN PROCESS CLASSIFICATION (GPC)")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Day 4 Preprocessed Data
    data_path = os.path.join(script_dir, 'data_splits', 'day_4_processed_data.npz')
    print(f"\n[Loading Data] {data_path}")
    data = np.load(data_path, allow_pickle=True)
    
    X_train = data['X_train']
    X_test  = data['X_test']
    y_train_clf = data['y_train_clf']
    y_test_clf  = data['y_test_clf']
    y_test_reg  = data['y_test_reg'] # For plotting the overlay
    test_dates  = pd.to_datetime(data['test_dates'])
    
    # 2. Define the Matern Kernel
    # In GPC, the noise is absorbed by the probit/logit link function, 
    # so we often do not need a WhiteKernel. But for consistency with the GPR
    # and to handle extreme non-separability, we'll keep a tiny WhiteKernel.
    kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) \
             + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
             
    # 3. Define the GPC
    # multi_class='one_vs_rest' is used internally by sklearn for >2 classes.
    # This means it will fit 3 separate binary GPCs, which is very computationally heavy.
    gpc = GaussianProcessClassifier(
        kernel=kernel,
        optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=2,  # Reduced from 5 to manage multiclass O(N^3) time
        random_state=42,
        multi_class='one_vs_rest',
        n_jobs=-1  # Parallelize the 3 OVR models across CPU cores
    )
    
    print("\n[Training GPC Model] Starting optimization (Laplace approximation)...")
    print("  (This requires fitting 3 separate O(N^3) models. Please be patient.)")
    
    start_time = time.time()
    gpc.fit(X_train, y_train_clf)
    fit_time = time.time() - start_time
    
    print(f"  Training completed in {fit_time:.2f} seconds.")
    print(f"  Optimized Kernels (One-vs-Rest):")
    for i, k in enumerate(gpc.kernel_.kernels):
        print(f"    Class {i}: {k}")
        
    # 4. Predict on Test
    print("\n[Predicting] Generating hard classes and probabilities...")
    y_test_pred = gpc.predict(X_test)
    y_test_prob = gpc.predict_proba(X_test)
    
    # 5. Evaluate Metrics
    print("\n[Evaluation] GPC Classification Report (Test Set)")
    target_names = ['Quiet (0)', 'Normal (1)', 'Turbulent (2)']
    print(classification_report(y_test_clf, y_test_pred, target_names=target_names))
    
    # 6. Visualizations
    
    # Plot 1: Confusion Matrix Heatmap
    print("\n[Visualizing] Generating Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_test_clf, y_test_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Quiet', 'Normal', 'Turbulent'], 
                yticklabels=['Quiet', 'Normal', 'Turbulent'], ax=ax)
                
    ax.set_title('GPC Confusion Matrix (Test Set)', fontweight='bold')
    ax.set_xlabel('Predicted Regime', fontweight='bold')
    ax.set_ylabel('Actual Regime', fontweight='bold')
    fig.tight_layout()
    _save_fig(fig, 'day8_gpc_confusion_matrix.png')
    
    # Plot 2: Time Series Overlay (Actual vs Predicted Regimes)
    print("              Generating Time-Series Regime Overlay...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(test_dates, y_test_reg, color='black', linewidth=1, label='Actual Volatility', zorder=2)
    
    colors = {0: '#2ca02c', 1: '#1f77b4', 2: '#d62728'}
    
    # Shade backgrounds based on predicted regime
    for i in range(len(test_dates) - 1):
        ax.axvspan(test_dates[i], test_dates[i+1], 
                   color=colors[y_test_pred[i]], alpha=0.25, linewidth=0, zorder=1)
                   
    # Plot misclassifications as markers
    errors = y_test_pred != y_test_clf
    ax.scatter(test_dates[errors], y_test_reg[errors], 
               color='black', marker='x', s=20, label='Misclassification', zorder=3)
                   
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=1, label='Actual Volatility'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markeredgecolor='black', markersize=8, label='Misclassification'),
        Patch(facecolor=colors[0], alpha=0.3, label='Predicted: Quiet'),
        Patch(facecolor=colors[1], alpha=0.3, label='Predicted: Normal'),
        Patch(facecolor=colors[2], alpha=0.3, label='Predicted: Turbulent')
    ]
    
    ax.set_title('Native GPC Predicted Regimes Overlaid on Actual Volatility (Test Set)', fontweight='bold')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Rolling Volatility', fontweight='bold')
    ax.legend(handles=legend_elements, loc='upper right')
    
    fig.tight_layout()
    _save_fig(fig, 'day8_gpc_regime_timeseries.png')
    
    # Plot 3: Probabilistic Output for an interesting sub-period (e.g., 2021 Bull Run / Crash)
    # The probabilistic output is scientifically important
    print("              Generating Probabilistic Confidence Plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})
    
    # Subplot 1: Volatility
    ax1.plot(test_dates, y_test_reg, color='black', linewidth=1.2)
    ax1.set_title('Actual Volatility vs GPC Class Probabilities (Test Set)', fontweight='bold')
    ax1.set_ylabel('Rolling Volatility', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Subplot 2: Stacked probabilities
    ax2.stackplot(test_dates, 
                  y_test_prob[:, 0], y_test_prob[:, 1], y_test_prob[:, 2], 
                  labels=['Quiet (Prob)', 'Normal (Prob)', 'Turbulent (Prob)'],
                  colors=['#2ca02c', '#1f77b4', '#d62728'], alpha=0.7)
    
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Probability', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 1)
    
    fig.tight_layout()
    _save_fig(fig, 'day8_gpc_probabilities.png')
    
    # Save results
    out_dir = os.path.join(script_dir, 'models')
    results = {
        'metrics': classification_report(y_test_clf, y_test_pred, output_dict=True),
        'predictions': y_test_pred,
        'probabilities': y_test_prob,
        'fit_time_seconds': fit_time
    }
    with open(os.path.join(out_dir, 'gpc_matern_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
        
    print("\n" + "=" * 60)
    print("  TASK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_day_8_gpc()
