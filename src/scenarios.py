from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, Optional
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
    
    Args:
        n_sim: Number of simulations per scenario.
    
    Returns:
        List[ScenarioConfig]: A list of configured scenarios.
    """
    scenarios = []
    thetas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    

    # --- Constants for Configuration ---
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
        # Common DGP params for this theta
        dgp_params_naive = {**BASE_DGP_PARAMS, 'include_collider': False, 'theta': theta}
        dgp_params_bad   = {**BASE_DGP_PARAMS, 'include_collider': True,  'theta': theta}

        # --- Tree Friendly Scenarios (Correct Specification) ---
        
        # 1. Naive DML (No OVB)
        # "Naive" means we do NOT include the "bad control" (Collider) in our dataset.
        # This is the CORRECT approach when OVB is absent (alpha_u=gamma_u=0).
        # We run this for all thetas to show that even if a collider exists generating data,
        # IGNORING it (Naive) should yield the correct zero bias.
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

        # 2. Bad Control DML
        # "Bad Control" means we INCLUDE the collider in our dataset W.
        # DML will control for it, which inadvertently opens a backdoor path, causing bias.
        # As theta increases, this bias should get worse.
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

        # 3. Bad Control EconML (No OVB)
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

        # 4. Naive EconML (No OVB)
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