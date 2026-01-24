import pandas as pd
from pathlib import Path
from src.scenarios import get_scenarios
from src.orchestration.orchestrator import run_all_scenarios
from src.dgps.wgan import WGANDGP

def main():
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    results_path = results_dir / "final_results.csv"
    
    n_sim = 100
    
    print(f"Preparing scenarios (N={n_sim})...")

    print("Training WGAN models...")
    WGANDGP() 

    scenarios = get_scenarios(n_sim=n_sim)
    
    print(f"Starting execution of {len(scenarios)} scenarios...")
    
    results_df = run_all_scenarios(scenarios)
    
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved {len(results_df)} rows to {results_path}")

if __name__ == "__main__":
    main()