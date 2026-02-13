import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sys
import os
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dgps.tree_friendly import TreeFriendlyDGP
from src.dgps.plr_ccddhnr2018 import PLRCCDDHNR2018DGP

def verify_fallacy():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'figure.figsize': (14, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'lines.linewidth': 2.5
    })

    fig, (ax1, ax2) = plt.subplots(1, 2)
    n = 2000
    
    # --- Tree Friendly ---
    tf_dgp = TreeFriendlyDGP(n_features=11, confounding_strength=0.2, noise_std=1.0, include_collider=True, theta=1.0)
    D_tf, Y_tf, W_tf = tf_dgp.sample(n)
    X_tf = W_tf
    
    model_tf = LinearRegression()
    model_tf.fit(X_tf[:, 0].reshape(-1, 1), D_tf)
    
    model_dml = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42)
    model_dml.fit(X_tf, D_tf)
    
    x_grid_tf = np.linspace(-3, 3, 200)
    g_grid_true = 0.2 * (0.1 * x_grid_tf**3)
    g_grid_ols = model_tf.predict(x_grid_tf.reshape(-1, 1))
    
    # Predict DML with other covariates at 0 (mean)
    n_w_cols = X_tf.shape[1]
    X_grid_full = np.zeros((200, n_w_cols))
    X_grid_full[:, 0] = x_grid_tf
    g_grid_dml = model_dml.predict(X_grid_full)

    # Check Symmetry
    true_confounding_signal = 0.2 * (0.1 * X_tf[:, 0]**3 + 0.5 * np.sin(X_tf[:, 1]))
    print(f"Tree-Friendly Confounder Mean: {np.mean(true_confounding_signal):.4f}")
    print(f"Tree-Friendly Confounder correlation with D: {np.corrcoef(true_confounding_signal, D_tf)[0,1]:.4f}")

    ax1.scatter(X_tf[:, 0], D_tf, color='gray', alpha=0.15, s=15, label='Observed Data')
    ax1.plot(x_grid_tf, g_grid_true, color='#C0392B', label='True Confounder', linewidth=3)
    ax1.plot(x_grid_tf, g_grid_ols, color='#2980B9', linestyle='--', label='OLS Fit', linewidth=2.5)
    ax1.plot(x_grid_tf, g_grid_dml, color='green', linestyle=':', label='DML (RF) Fit', linewidth=2.5)

    ax1.set_title('TreeFriendly: Bias "Hidden by Noise"')
    ax1.set_xlabel('Covariate X0 (Primary Non-Linear Feature)')
    ax1.set_ylabel('Treatment D')
    ax1.set_ylim(-3, 3)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, alpha=0.3)

    # --- PLR ---
    plr_dgp = PLRCCDDHNR2018DGP(n_features=11, noise_std=1.0, include_collider=True, theta=1.0)
    D_plr, Y_plr, W_plr = plr_dgp.sample(n)
    X_plr = W_plr
    
    true_confounder_values = 1.0 * np.exp(X_plr[:, 0]) / (1 + np.exp(X_plr[:, 0]))
    
    model_plr = LinearRegression()
    model_plr.fit(X_plr[:, 0].reshape(-1, 1), true_confounder_values)
    
    x_grid = np.linspace(-3, 3, 200)
    y_grid_true = 1.0 * np.exp(x_grid) / (1 + np.exp(x_grid))
    y_grid_ols = model_plr.predict(x_grid.reshape(-1, 1))

    ax2_density = ax2.twinx()
    ax2_density.hist(X_plr[:, 0], bins=40, color='gray', alpha=0.15, density=True)
    ax2_density.set_yticks([]) 
    
    ax2.plot(x_grid, y_grid_true, color='#C0392B', label='True Confounder', linewidth=3)
    ax2.plot(x_grid, y_grid_ols, color='#2980B9', linestyle='--', label='OLS Fit', linewidth=2.5)

    ax2.set_title('PLR: "Lucky Linearity"')
    ax2.set_xlabel('Covariate X0 (Sigmoid Driver)')
    ax2.set_ylabel('Confounding Function g(X)')
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'fallacy_verification.png')
    plt.savefig(output_path, dpi=200)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    verify_fallacy()
