import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from doubleml import DoubleMLData, DoubleMLPLR
from src.dgps.tree_friendly import TreeFriendlyDGP
from joblib import Parallel, delayed

# --- Tuning Configuration ---
n_sim = 100 # High certainty
theta = 0.0 # Naive case

# Grid
grid_n_obs = [2000]
grid_leaf = [10, 5, 3, 1]
grid_features = [0.33, 0.5, 0.7, 0.9, 1.0] 
grid_trees = [100, 200, 500]

print(f"--- Coverage Tuning (Legacy cubic_sin, N=2000, Sim={n_sim}) ---") 

def run_simulation(seed, n_obs, leaf, features, trees, theta):
    # Generate Data
    dgp = TreeFriendlyDGP(
        n_features=4, include_collider=False, theta=theta, 
        alpha_u=0.0, gamma_u=0.0,
        confounding_strength=0.2, noise_std=1.0
    )
    D, Y, W = dgp.sample(n_obs, seed=seed)
    
    df = pd.DataFrame(W, columns=[f'W{j}' for j in range(W.shape[1])])
    df['D'] = D; df['Y'] = Y
    dml_data = DoubleMLData(df, 'Y', 'D', list(df.columns[:-2]))
    
    # RF Params
    rf_params = {
        'n_estimators': trees,
        'max_depth': None,
        'max_features': features,
        'min_samples_leaf': leaf,
        'min_samples_split': 10,
        'n_jobs': 1, # inner jobs 1 to avoid contention with outer parallel loop
        'random_state': 42 # Deterministic RF split behavior
    }
    
    ml_l = RandomForestRegressor(**rf_params)
    ml_m = RandomForestRegressor(**rf_params)
    
    # Fit DML
    try:
        dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=5, n_rep=1)
        dml_plr.fit()
        
        tau_hat = dml_plr.coef[0]
        ci = dml_plr.confint().iloc[0]
        covered = (ci['2.5 %'] <= 1.0) and (ci['97.5 %'] >= 1.0)
        mse = (tau_hat - 1.0)**2
        bias = (tau_hat - 1.0)
        return covered, mse, bias
    except Exception as e:
        print(f"Error in simulation: {e}")
        return False, 999.0, 999.0

results = []

# Iterate through Grid
for n_obs in grid_n_obs:
    for leaf in grid_leaf:
        for features in grid_features:
            for trees in grid_trees:
                
                # Parallel Execution of Simulations
                # n_jobs=-1 uses all available cores
                sim_results = Parallel(n_jobs=-1, verbose=0)(
                    delayed(run_simulation)(
                        seed=42+i, 
                        n_obs=n_obs, 
                        leaf=leaf, 
                        features=features, 
                        trees=trees, 
                        theta=theta
                    ) for i in range(n_sim)
                )
                
                # Unzip results
                coverage_list, mse_list, bias_list = zip(*sim_results)
                
                avg_cov = np.mean(coverage_list)
                avg_mse = np.mean(mse_list)
                avg_bias = np.mean(bias_list)
                
                print(f"N={n_obs:<4} | Leaf={leaf:<2} | Feat={features:<4} | Trees={trees:<3} || Cov={avg_cov:.2f} | MSE={avg_mse:.4f} | Bias={avg_bias:.4f}")
                
                results.append({
                    'n_obs': n_obs,
                    'leaf': leaf,
                    'features': features,
                    'trees': trees,
                    'coverage': avg_cov,
                    'mse': avg_mse,
                    'bias': avg_bias
                })

print("\n--- Top 10 Configs by MSE ---")
df_res = pd.DataFrame(results)
print(df_res.sort_values(by='mse').head(10))

# Save results
df_res.to_csv('results/tuning_results_n2000_parallel.csv', index=False)
print("\nSaved full results to results/tuning_results_n2000_parallel.csv")

print("\n--- Configs with > 0.94 Coverage (sorted by bias ratio) ---")

valid_cov = df_res[df_res['coverage'] >= 0.94]
if not valid_cov.empty:
    print(valid_cov.sort_values(by=['mse']))
else:
    print("No configs hit > 0.94 coverage.")