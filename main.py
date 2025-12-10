import pandas as pd
from pathlib import Path
from src.scenarios import get_scenarios
from src.orchestration.orchestrator import run_all_scenarios

def main():
    # --- Configuration ---
    # Create results folder if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Define file path
    results_path = results_dir / "final_results.csv"
    
    # Simulation Settings
    n_sim = 100
    
    # --- 1. Get Scenarios ---
    print(f"Preparing scenarios (N={n_sim})...")
    scenarios = get_scenarios(n_sim=n_sim)
    
    # --- 2. Run Simulations ---
    print(f"Starting execution of {len(scenarios)} scenarios...")
    
    # This runs all simulations in parallel and gives us a DataFrame back
    results_df = run_all_scenarios(scenarios)
    
    # --- 3. Save Results ---
    results_df.to_csv(results_path, index=False)
    print(f"\nSuccess! Saved {len(results_df)} rows to {results_path}")

if __name__ == "__main__":
    main()