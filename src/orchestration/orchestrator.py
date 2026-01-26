import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List, Dict, Tuple
from pathlib import Path

from src.orchestration.runner import run_single_simulation, run_raw_simulation
from src.scenarios import get_microscope_scenario

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.scenarios import ScenarioConfig


def run_all_scenarios(scenarios: List["ScenarioConfig"]) -> pd.DataFrame:
    """
    Runs simulations for each scenario in parallel.
    Utilizes joblib to maximize CPU core usage.
    """
    results_path = Path("results/final_results.csv")
    all_results = []
    completed_sims = set()

    if results_path.exists():
        try:
            existing_df = pd.read_csv(results_path)
            all_results = existing_df.to_dict("records")
            for _, row in existing_df.iterrows():
                completed_sims.add((row["scenario"], int(row["sim_id"])))
            print(f"Resuming execution: Found {len(all_results)} completed simulations.")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}. Starting from scratch.")

    for config in tqdm(scenarios, desc="Overall Progress"):
        sims_to_run = [
            i for i in range(config.n_simulations)
            if (config.name, i) not in completed_sims
        ]

        if not sims_to_run:
            continue

        results = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)(
                config=config,
                seed=config.first_seed + i,
                sim_id=i
            )
            for i in tqdm(sims_to_run, desc=f"Running {config.name}", leave=False)
        )

        all_results.extend(results)
        pd.DataFrame(all_results).to_csv(results_path, index=False)

    return pd.DataFrame(all_results)


def run_microscope_diagnostic(
    theta: float = 1.0,
    n_obs: int = 2000,
    seed: int = 42,
    dgps: List[str] = None
) -> Dict[str, Tuple[object, object]]:
    """
    Runs targeted diagnostic simulations for multiple DGPs.
    Returns a dict: {dgp_name: (dgp_obj, est_obj)}.
    """
    if dgps is None:
        dgps = ["TreeFriendly", "PLR", "WGAN"]

    print(f"Running Microscope Diagnostic (Theta={theta}) for DGPS={dgps} ...")

    outputs: Dict[str, Tuple[object, object]] = {}

    for dgp_name in dgps:
        config = get_microscope_scenario(theta=theta, seed=seed, dgp=dgp_name)
        config.sample_size = n_obs

        dgp_obj, est_obj = run_raw_simulation(config, seed=seed)
        outputs[dgp_name] = (dgp_obj, est_obj)

    return outputs
