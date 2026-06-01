import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def _save_fig(fig, filename):
    """Helper to save figures into the Research/figures directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    filepath = os.path.join(fig_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  [Saved] {filepath}")

def run_day_5_gpr_matern():
    print("\n" + "=" * 60)
    print("  GPR WITH MATERN KERNEL")
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
    train_dates = pd.to_datetime(data['train_dates'])
    test_dates  = pd.to_datetime(data['test_dates'])
    feature_names = data['feature_names']
    
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")
    
    # Optional: If GPR is too slow, we can sub-sample, but we'll try the full dataset first.
    # Scikit-learn GPR is O(N^3). N=4782 might take a few minutes to fit.
    
    # 2. Define the Matern Kernel + White Kernel
    # The Matern kernel is initialized with nu=2.5, which implies once-differentiable 
    # sample paths, providing a good balance between smoothness and roughness for financial data.
    kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) \
             + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
             
    # 3. Define the GPR
    gpr_matern = GaussianProcessRegressor(
        kernel=kernel, 
        alpha=0.0, # Alpha is 0 because we're using WhiteKernel for noise
        optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=5, 
        normalize_y=True,
        random_state=42
    )
    
    print("\n[Training GPR Matern Model] Starting optimization...")
    print("  (This may take several minutes due to O(N^3) complexity on 4782 rows)")
    
    start_time = time.time()
    gpr_matern.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    print(f"  Training completed in {fit_time:.2f} seconds.")
    print(f"  Optimized Kernel: {gpr_matern.kernel_}")
    print(f"  Log-Marginal-Likelihood: {gpr_matern.log_marginal_likelihood_value_:.4f}")
    
    # 4. Predict on Train and Test
    print("\n[Predicting] Generating mean and uncertainty estimates...")
    y_train_pred, y_train_std = gpr_matern.predict(X_train, return_std=True)
    y_test_pred, y_test_std   = gpr_matern.predict(X_test, return_std=True)
    
    # 5. Evaluate Metrics
    def print_metrics(name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        print(f"  {name} Metrics:")
        print(f"    RMSE : {rmse:.6f}")
        print(f"    MAE  : {mae:.6f}")
        print(f"    R^2  : {r2:.4f}")
        return rmse, mae, r2
        
    train_metrics = print_metrics("Train", y_train, y_train_pred)
    test_metrics  = print_metrics("Test", y_test, y_test_pred)
    
    # 6. Visualization: The "Most Important Figure"
    print("\n[Visualizing] Generating confidence interval plot on Test set...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot True vs Pred on Test set
    ax.plot(test_dates, y_test, color='black', label='Actual Volatility', linewidth=1.2, alpha=0.8)
    ax.plot(test_dates, y_test_pred, color='blue', label='Predicted Mean', linewidth=1.5)
    
    # 95% Confidence Interval (1.96 * std)
    ax.fill_between(
        test_dates, 
        y_test_pred - 1.96 * y_test_std, 
        y_test_pred + 1.96 * y_test_std, 
        alpha=0.3, color='blue', label='95% Confidence Interval'
    )
    
    ax.set_title('GPR (Matern $\\nu=2.5$) Predicted Volatility vs Actual (Test Set)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('20-Day Rolling Volatility', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    _save_fig(fig, 'day5_gpr_matern_test_forecast.png')
    
    # 7. Save Model & Results for comparison
    out_dir = os.path.join(script_dir, 'models')
    os.makedirs(out_dir, exist_ok=True)
    
    results = {
        'kernel_str': str(gpr_matern.kernel_),
        'metrics': {
            'train': {'rmse': train_metrics[0], 'mae': train_metrics[1], 'r2': train_metrics[2]},
            'test':  {'rmse': test_metrics[0],  'mae': test_metrics[1],  'r2': test_metrics[2]}
        },
        'predictions': {
            'y_test_pred': y_test_pred,
            'y_test_std': y_test_std
        },
        'fit_time_seconds': fit_time
    }
    
    # We save the results dict instead of the full model object to save space and avoid pickling issues
    results_path = os.path.join(out_dir, 'gpr_matern_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n[Saved] Matern results saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("  TASK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_day_5_gpr_matern()
