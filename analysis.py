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
        plot_microscope_view,
        plot_bias_comparison,
        plot_coverage_comparison, 
        plot_bias_variance_grid,
        plot_tau_distribution_1x3_by_dgp
    )
    from src.orchestration.orchestrator import run_microscope_diagnostic
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def main():
    # --- Paths ---
    results_path = Path('results/final_results.csv')

    results_basedir = Path('results')
    results_basedir.mkdir(parents=True, exist_ok=True)

    plots_dir = results_basedir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Performance Metrics and Standard Plots ---
    if results_path.exists():
        print("Processing standard metrics...")
        df = pd.read_csv(results_path)

        summary, df = calculate_metrics(df)
        summary.to_csv(results_basedir / 'detailed_summary_metrics.csv', index=False)

        plot_bias_comparison(df, plots_dir)
        plot_coverage_comparison(summary, plots_dir)
        plot_bias_variance_grid(df, plots_dir)
        plot_tau_distribution_1x3_by_dgp(df, plots_dir)


    else:
        print(f"Warning: Results file not found at {results_path}. Skipping metrics.")

    # --- Microscope Diagnostic ---
    # Generates the paradox visualization for Theta=1.0
    outs = run_microscope_diagnostic(theta=1.0)

    for dgp_name, (dgp, est) in outs.items():
        plot_microscope_view(
            dgp,
            est,
            theta=1.0,
            output_dir=results_basedir,
            filename_suffix=f"_{dgp_name}"
        )
        print("Analysis complete.")


if __name__ == "__main__":
    main()
