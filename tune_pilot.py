"""Pilot Tuning Script for Hyperparameter Optimization.

Performs empirical tuning of Random Forest hyperparameters using Cross-Validation
on the nuisance loss (MSE), without access to true treatment effects.
"""

import numpy as np
from src.dgps.tree_friendly import TreeFriendlyDGP
from src.dgps.plr_ccddhnr2018 import PLRCCDDHNR2018DGP
from src.estimators.dml import DoubleMLEstimator


def run_pilot_tuning():
    """Runs a pilot tuning session to determine optimal Random Forest parameters.

    Uses Cross-Validation on nuisance models (Y~X, D~X), respecting empirical
    constraints.
    """
    np.random.seed(42)

    print("Starting Pilot Tuning Phase...")

    print("\n--- Tuning for TreeFriendlyDGP ---")
    dgp_tree = TreeFriendlyDGP(theta=0.0)
    d, y, w = dgp_tree.sample(n_obs=2000, seed=387)

    def rf_param_space(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_categorical('max_depth', [5, 10, 20, None]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', [0.3, 0.5, 0.8, 'sqrt', 'log2']),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        }

    ml_param_space = {
        'ml_l': rf_param_space,
        'ml_m': rf_param_space
    }

    est = DoubleMLEstimator(random_state=387, n_jobs=-1)

    try:
        best_params = est.tune(
            d, y, w,
            param_space=ml_param_space,
            n_trials=20,
            show_progress=False
        )
        print("Best Params (TreeFriendly):")
        print("ml_l:", best_params['ml_l']['D'][0][0])
        print("ml_m:", best_params['ml_m']['D'][0][0])
    except Exception as e:
        print(f"Tuning failed for TreeFriendly: {e}")

    print("\n--- Tuning for PLR DGP ---")
    dgp_plr = PLRCCDDHNR2018DGP(theta=0.0)
    d, y, w = dgp_plr.sample(n_obs=2000, seed=42)

    try:
        best_params_plr = est.tune(
            d, y, w,
            param_space=ml_param_space,
            n_trials=20,
            show_progress=False
        )
        print("Best Params (PLR):")
        print("ml_l:", best_params_plr['ml_l']['D'][0][0])
        print("ml_m:", best_params_plr['ml_m']['D'][0][0])
    except Exception as e:
        print(f"Tuning failed for PLR: {e}")


if __name__ == "__main__":
    run_pilot_tuning()
