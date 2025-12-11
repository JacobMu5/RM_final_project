import sys
import pandas as pd
import matplotlib
from pathlib import Path

# Use Agg backend to prevent display errors on headless environments
matplotlib.use('Agg')

# Ensure local modules are discoverable
sys.path.append(str(Path.cwd()))

try:
    from src.evaluation import calculate_metrics
    from src.plotting import (
        plot_standard_metrics, 
        plot_bias_distribution, 
        plot_bias_variance,
        plot_microscope_view
    )
    from src.orchestration.orchestrator import run_microscope_diagnostic
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    # --- Performance Metrics and Standard Plots ---
    results_path = Path('results/final_results.csv')
    
    if results_path.exists():
        print("Processing standard metrics...")
        df = pd.read_csv(results_path)
        
        summary, df = calculate_metrics(df)
        summary.to_csv('results/detailed_summary_metrics.csv', index=False)
        
        plots_dir = Path('results/plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_standard_metrics(df, summary, plots_dir)
        plot_bias_distribution(df, plots_dir)
        plot_bias_variance(df, plots_dir)
    else:
        print(f"Warning: Results file not found at {results_path}. Skipping metrics.")

    # --- Microscope Diagnostic ---
    # Generates the paradox visualization for Theta=1.0
    results_basedir = Path('results')
    dgp, est = run_microscope_diagnostic(theta=1.0)

    plot_microscope_view(dgp, est, theta=1.0, output_dir=results_basedir)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()