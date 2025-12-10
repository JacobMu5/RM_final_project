import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List
from src.orchestration.runner import run_single_simulation

# Type hint trick
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.scenarios import ScenarioConfig

def run_all_scenarios(scenarios: List['ScenarioConfig']) -> pd.DataFrame:
    """
    Runs simulations for each scenario in parallel.
    Uses joblib to utilize all CPU cores.
    """
    all_results = []
    
    # Iterate over scenarios
    for config in tqdm(scenarios, desc="Overall Progress"):
        
        # Run simulations in parallel (n_jobs=-1 uses all cores)
        results = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)(
                config=config,
                seed=config.first_seed + i,
                sim_id=i
            ) for i in tqdm(range(config.n_simulations), desc=f"Running {config.name}", leave=False)
        )
        all_results.extend(results)
    
    return pd.DataFrame(all_results)