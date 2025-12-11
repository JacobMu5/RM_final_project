from dataclasses import dataclass
from typing import Any, Dict, List, Type
from src.dgps.tree_friendly import TreeFriendlyDGP
from src.estimators.dml import DoubleMLEstimator
from src.estimators.econml import EconMLEstimator

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

def get_scenarios(n_sim: int = 100) -> List[ScenarioConfig]:
    """
    Generates a list of ScenarioConfig objects for the simulation.
    """
    scenarios = []
    thetas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Constants
    BASE_RF_PARAMS = {
        'n_estimators': 200, 
        'min_samples_leaf': 1, 
        'max_features': 0.9,
        'min_samples_split': 10,
        'max_depth': None, 
        'n_jobs': -1,
        'random_state': 42 
    }
    
    BASE_DGP_PARAMS = {'n_features': 4, 'alpha_u': 0.0, 'gamma_u': 0.0}
    SAMPLE_SIZE = 2000
    
    for theta in thetas:
        # Configuration for Naive (No Collider) vs Bad Control (Includes Collider)
        dgp_params_naive = {**BASE_DGP_PARAMS, 'include_collider': False, 'theta': theta}
        dgp_params_bad   = {**BASE_DGP_PARAMS, 'include_collider': True,  'theta': theta}

        # --- DML Scenarios ---
        
        # Naive DML (Correct Specification: Collider ignored)
        scenarios.append(ScenarioConfig(
            name=f"TreeFriendly_Naive_DML_theta_{theta}",
            dgp_class=TreeFriendlyDGP,
            dgp_params=dgp_params_naive,
            estimator_class=DoubleMLEstimator,
            estimator_params={'n_folds': 5, 'n_trees': 200, 'n_rep': 1, 'rf_params': BASE_RF_PARAMS},
            sample_size=SAMPLE_SIZE,
            n_simulations=n_sim,
            first_seed=42 
        ))

        # Bad Control DML (Misspecified: Collider included in W)
        scenarios.append(ScenarioConfig(
            name=f"TreeFriendly_BadControl_DML_theta_{theta}",
            dgp_class=TreeFriendlyDGP,
            dgp_params=dgp_params_bad,
            estimator_class=DoubleMLEstimator,
            estimator_params={'n_folds': 5, 'n_trees': 200, 'n_rep': 1, 'rf_params': BASE_RF_PARAMS},
            sample_size=SAMPLE_SIZE,
            n_simulations=n_sim,
            first_seed=42
        ))

        # --- EconML Scenarios ---

        # Bad Control EconML (Misspecified)
        scenarios.append(ScenarioConfig(
            name=f"TreeFriendly_BadControl_EconML_theta_{theta}",
            dgp_class=TreeFriendlyDGP,
            dgp_params=dgp_params_bad,
            estimator_class=EconMLEstimator,
            estimator_params={'n_estimators': 200, 'rf_params': BASE_RF_PARAMS},
            sample_size=SAMPLE_SIZE,
            n_simulations=n_sim,
            first_seed=42
        ))

        # Naive EconML (Correct Specification)
        scenarios.append(ScenarioConfig(
            name=f"TreeFriendly_Naive_EconML_theta_{theta}",
            dgp_class=TreeFriendlyDGP,
            dgp_params=dgp_params_naive,
            estimator_class=EconMLEstimator,
            estimator_params={'n_estimators': 200, 'rf_params': BASE_RF_PARAMS},
            sample_size=SAMPLE_SIZE,
            n_simulations=n_sim,
            first_seed=42
        ))
            
    return scenarios

def get_microscope_scenario(theta: float = 1.0, seed: int = 42) -> ScenarioConfig:
    """
    Returns a specific configuration for the 'Microscope View' diagnostic.
    """
    rf_params = {
        'n_estimators': 200, 
        'min_samples_leaf': 1, 
        'max_features': 0.9, 
        'n_jobs': -1, 
        'random_state': seed
    }
    
    return ScenarioConfig(
        name="Microscope_Diagnostic",
        dgp_class=TreeFriendlyDGP,
        dgp_params={'n_features': 4, 'include_collider': True, 'theta': theta},
        estimator_class=EconMLEstimator,
        estimator_params={'n_estimators': 200, 'random_state': seed, 'rf_params': rf_params},
        sample_size=2000,
        n_simulations=1,
        first_seed=seed
    )