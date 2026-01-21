from dataclasses import dataclass
from typing import Any, Dict, List, Type
import numpy as np
from src.dgps.tree_friendly import TreeFriendlyDGP
from src.dgps.plr_ccddhnr2018 import PLRCCDDHNR2018DGP
from src.estimators.dml import DoubleMLEstimator
from src.estimators.econml import EconMLEstimator
from src.estimators.ols import OLSEstimator


@dataclass
class ScenarioConfig:
    name: str
    dgp_class: Type
    dgp_params: Dict[str, Any]
    estimator_class: Type
    estimator_params: Dict[str, Any]
    sample_size: int
    n_simulations: int
    first_seed: int


class ReproducibleDoubleMLEstimator(DoubleMLEstimator):
    """Wrapper to ensure global randon seed is set before DoubleML execution."""
    def fit(self, D, Y, W):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        super().fit(D, Y, W)


class ReproducibleEconMLEstimator(EconMLEstimator):
    """Wrapper to ensure global randon seed is set before EconML execution."""
    def fit(self, D, Y, W):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        super().fit(D, Y, W)

class ReproducibleOLSEstimator(OLSEstimator):
    """Wrapper to ensure global random seed is set before OLS execution (mostly for symmetry)."""
    def fit(self, D, Y, W):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        super().fit(D, Y, W)


def get_rf_search_space():
    """Defines the hyperparameter search space for Optuna tuning."""
    def ml_l_params(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 0.2, 0.33, 0.4, 0.5, 0.6, 0.8]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10, 20]),
            'max_depth': trial.suggest_categorical('max_depth', [10, 20, 30, None]),
        }
    return {'ml_l': ml_l_params, 'ml_m': ml_l_params}


def get_scenarios(n_sim: int = 100) -> List[ScenarioConfig]:
    """Generates a list of ScenarioConfig objects for the simulation."""
    scenarios = []

    RF_PARAMS = {
        'TreeFriendly': {'n_jobs': -1, 'n_estimators': 500, 'max_features': 0.3, 'min_samples_leaf': 6, 'min_samples_split': 5, 'max_depth': 5},
        'PLR': {'n_jobs': -1, 'n_estimators': 400, 'max_features': 0.8, 'min_samples_leaf': 15, 'min_samples_split': 7, 'max_depth': None}
    }

    DGP_CONFIGS = [
        ('TreeFriendly', TreeFriendlyDGP, {'n_features': 4, 'alpha_u': 0.0, 'gamma_u': 0.0}, 
         {'n_folds': 5, 'n_trees': 400, 'n_rep': 1}, {'n_estimators': 400}),
        ('PLR', PLRCCDDHNR2018DGP, {'n_features': 20, 'tau': 1.0},
         {'n_folds': 5, 'n_trees': 500, 'n_rep': 1}, {'n_estimators': 300})
    ]

    VARIANTS = [
        ('Naive', {'include_collider': False}),
        ('BadControl', {'include_collider': True}),
        ('LinearCollider', {'include_collider': False, 'include_linear_collider': True})
    ]

    def add_scenario(prefix, variant, dgp_cls, dgp_p, est_cls, est_p):
        est_name = est_cls.__name__.replace('Reproducible', '').replace('Estimator', '')
        scenarios.append(ScenarioConfig(
            name=f"{prefix}_{variant}_{est_name}_theta_{dgp_p['theta']}",
            dgp_class=dgp_cls, dgp_params=dgp_p, estimator_class=est_cls,
            estimator_params=est_p, sample_size=2000, n_simulations=n_sim, first_seed=42
        ))

    for theta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for prefix, dgp_cls, base_params, dml_params, econml_params in DGP_CONFIGS:
            for variant_name, variant_params in VARIANTS:
                if variant_name == "Naive" and theta != 0.0:
                    continue
                dgp_p = {**base_params, **variant_params, 'theta': theta}
                add_scenario(prefix, variant_name, dgp_cls, dgp_p, ReproducibleDoubleMLEstimator, {**dml_params, 'rf_params': RF_PARAMS[prefix]})
                add_scenario(prefix, variant_name, dgp_cls, dgp_p, ReproducibleEconMLEstimator, {**econml_params, 'rf_params': RF_PARAMS[prefix]})
                add_scenario(prefix, variant_name, dgp_cls, dgp_p, ReproducibleOLSEstimator,{"add_intercept": True, "robust_se": "HC3"})

    return scenarios


def get_microscope_scenario(theta: float = 1.0, seed: int = 42) -> ScenarioConfig:
    """Returns a specific configuration for the 'Microscope View' diagnostic."""
    return ScenarioConfig(
        name="Microscope_Diagnostic",
        dgp_class=TreeFriendlyDGP,
        dgp_params={'n_features': 4, 'include_collider': True, 'theta': theta},
        estimator_class=ReproducibleEconMLEstimator,
        estimator_params={'n_estimators': 200, 'random_state': seed, 'rf_params': {'n_estimators': 200, 'min_samples_leaf': 1, 'max_features': 0.9, 'n_jobs': -1}},
        sample_size=2000,
        n_simulations=1,
        first_seed=seed
    )