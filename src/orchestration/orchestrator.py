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
    results_path = Path('results/final_results.csv')
    all_results = []
    completed_sims = set()
    
    if results_path.exists():
        try:
            existing_df = pd.read_csv(results_path)
            all_results = existing_df.to_dict('records')
            for index, row in existing_df.iterrows():
                completed_sims.add((row['scenario'], int(row['sim_id'])))
            print(f"Resuming execution: Found {len(all_results)} completed simulations.")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}. Starting from scratch.")
    
    for config in tqdm(scenarios, desc="Overall Progress"):
        sims_to_run = [i for i in range(config.n_simulations) 
                       if (config.name, i) not in completed_sims]
        
        if not sims_to_run:
            continue

        results = Parallel(n_jobs=7)(
            delayed(run_single_simulation)(
                config=config,
                seed=config.first_seed + i,
                sim_id=i
            ) for i in tqdm(sims_to_run, desc=f"Running {config.name}", leave=False)
        )
        all_results.extend(results)
        
        pd.DataFrame(all_results).to_csv(results_path, index=False)
    
    return pd.DataFrame(all_results)

def run_microscope_diagnostic(theta: float = 1.0, n_obs: int = 2000, seed: int = 42):
    """
    Runs a single targeted simulation for diagnostic purposes.
    Returns the raw DGP and Estimator objects instead of metrics.
    
    This fits logic into the orchestration layer because it involves setting up
    and running a specialized simulation flow.
    """
    print(f"Running Microscope Diagnostic (Theta={theta})...")
    
    config = get_microscope_scenario(theta=theta, seed=seed)
    
    config.sample_size = n_obs

    dgp, est = run_raw_simulation(config, seed=seed)
    
    return dgp, est