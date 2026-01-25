import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List
from pathlib import Path
from src.orchestration.runner import run_single_simulation, run_raw_simulation
from src.scenarios import get_microscope_scenario

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.scenarios import ScenarioConfig

def run_all_scenarios(scenarios: List['ScenarioConfig']) -> pd.DataFrame:
    """
    Runs simulations for each scenario in parallel.
    Utilizes joblib to maximize CPU core usage.
    """
    all_results = []
    
    for config in tqdm(scenarios, desc="Overall Progress"):
        # run_single_simulation handles the logic for one run
        # n_jobs=-1 uses all available cores for this batch
        results = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)(
                config=config,
                seed=config.first_seed + i,
                sim_id=i
            ) for i in tqdm(range(config.n_simulations), desc=f"Running {config.name}", leave=False)
        )
        all_results.extend(results)
        pd.DataFrame(all_results).to_csv('results/final_results.csv', index=False)
    
    return pd.DataFrame(all_results)

def run_microscope_diagnostic(theta: float = 1.0, n_obs: int = 2000, seed: int = 42):
    """
    Runs a single targeted simulation for diagnostic purposes.
    Returns the raw DGP and Estimator objects instead of metrics.
    
    This fits logic into the orchestration layer because it involves setting up
    and running a specialized simulation flow.
    """
    print(f"Running Microscope Diagnostic (Theta={theta})...")
    
    # 1. Get Config from Scenarios
    # Note: We override n_obs/seed if passed, but the default helper matches defaults
    config = get_microscope_scenario(theta=theta, seed=seed)
    
    # Ensure sample size matches request (helper defaults to 2000)
    config.sample_size = n_obs

    # 2. Run Raw Simulation using Runner
    dgp, est = run_raw_simulation(config, seed=seed)
    
    return dgp, est