import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def _save_fig(fig, filename):
    """Helper to save figures into figures directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    filepath = os.path.join(fig_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  [Saved] {filepath}")

def run_day_6_gpr_rbf():
    print("\n" + "=" * 60)
    print("  GPR WITH RBF KERNEL")
    print("=" * 60)
    
    # 1. Load the preprocessed data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data_splits', 'day_4_processed_data.npz')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run day_4_preprocessing.py first.")
        
    print(f"\n[Loading Data] {data_path}")
    data = np.load(data_path, allow_pickle=True)
    
    X_train = data['X_train']
    X_test  = data['X_test']
    y_train = data['y_train_reg']
    y_test  = data['y_test_reg']
    test_dates  = pd.to_datetime(data['test_dates'])
    
    # 2. Define the RBF Kernel + White Kernel
    # RBF assumes infinitely differentiable (smooth) functions.
    kernel_rbf = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
               + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
             
    # 3. Define the GPR
    gpr_rbf = GaussianProcessRegressor(
        kernel=kernel_rbf, 
        alpha=0.0, 
        optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=5, 
        normalize_y=True,
        random_state=42
    )
    
    print("\n[Training GPR RBF Model] Starting optimization...")
    print("  (This will take around 15-20 minutes or more...")
    
    start_time = time.time()
    gpr_rbf.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    print(f"  Training completed in {fit_time:.2f} seconds.")
    print(f"  Optimized Kernel: {gpr_rbf.kernel_}")
    print(f"  Log-Marginal-Likelihood: {gpr_rbf.log_marginal_likelihood_value_:.4f}")
    
    # 4. Predict on Train and Test
    print("\n[Predicting] Generating mean and uncertainty estimates...")
    y_train_pred, y_train_std = gpr_rbf.predict(X_train, return_std=True)
    y_test_pred, y_test_std   = gpr_rbf.predict(X_test, return_std=True)
    
    # 5. Evaluate Metrics
    def calc_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return rmse, mae, r2
        
    train_rmse, train_mae, train_r2 = calc_metrics(y_train, y_train_pred)
    test_rmse, test_mae, test_r2 = calc_metrics(y_test, y_test_pred)
    
    # 6. Load Matern Results for Comparison
    matern_results_path = os.path.join(script_dir, 'models', 'gpr_matern_results.pkl')
    with open(matern_results_path, 'rb') as f:
        matern_results = pickle.load(f)
        
    matern_test_metrics = matern_results['metrics']['test']
    matern_test_pred = matern_results['predictions']['y_test_pred']
    
    # Extract optimized length scales from strings
    def extract_length_scale(kernel_str):
        # Example format: Matern(length_scale=2.36, nu=2.5) + WhiteKernel(noise_level=0.146)
        import re
        match = re.search(r"length_scale=([\d\.]+)", kernel_str)
        return float(match.group(1)) if match else None

    rbf_ls = extract_length_scale(str(gpr_rbf.kernel_))
    matern_ls = extract_length_scale(matern_results['kernel_str'])
    
    # 7. Print Comparison Table
    print("\n[Comparison] Matern vs RBF (Test Set)")
    print("-" * 60)
    print(f"| {'Metric':<25} | {'Matern (nu=2.5)':<15} | {'RBF':<10} |")
    print("-" * 60)
    print(f"| {'RMSE':<25} | {matern_test_metrics['rmse']:<15.6f} | {test_rmse:<10.6f} |")
    print(f"| {'MAE':<25} | {matern_test_metrics['mae']:<15.6f} | {test_mae:<10.6f} |")
    print(f"| {'R^2':<25} | {matern_test_metrics['r2']:<15.4f} | {test_r2:<10.4f} |")
    print(f"| {'Optimized length scale':<25} | {matern_ls:<15.4f} | {rbf_ls:<10.4f} |")
    print("-" * 60)
    
    # 8. Visualization: Side-by-Side overlay plot
    print("\n[Visualizing] Generating Kernel Comparison overlay plot...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(test_dates, y_test, color='black', label='Actual Volatility', linewidth=1.5, alpha=0.8)
    
    # Plot Matern Mean
    ax.plot(test_dates, matern_test_pred, color='blue', label='Matern ($\\nu=2.5$) Prediction', linewidth=1.0, alpha=0.8)
    
    # Plot RBF Mean
    ax.plot(test_dates, y_test_pred, color='red', label='RBF Prediction', linewidth=1.0, alpha=0.8)
    
    ax.set_title('GPR Kernel Comparison: Matern vs RBF (Test Set Out-of-Sample)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('20-Day Rolling Volatility', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    _save_fig(fig, 'day6_gpr_kernel_comparison.png')
    
    # ----------------------
    # 9. Persist RBF results
    # ----------------------
    out_dir = os.path.join(script_dir, 'models')
    os.makedirs(out_dir, exist_ok=True)

    results_rbf = {
        'kernel_str'      : str(gpr_rbf.kernel_),
        'metrics': {
            'train': {'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2},
            'test' : {'rmse': test_rmse,  'mae': test_mae,  'r2': test_r2},
        },
        'predictions': {
            'y_test_pred' : y_test_pred,
            'y_test_std'  : y_test_std,
        },
        'fit_time_seconds': fit_time,
    }
    rbf_pkl = os.path.join(out_dir, 'gpr_rbf_results.pkl')
    with open(rbf_pkl, 'wb') as f:
        pickle.dump(results_rbf, f)
    print(f"\n[Saved] RBF results saved to {rbf_pkl}")
    
    print("\n" + "=" * 60)
    print("  TASK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_day_6_gpr_rbf()
