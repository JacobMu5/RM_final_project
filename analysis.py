import sys
import pandas as pd
import matplotlib
from pathlib import Path


matplotlib.use('Agg')


sys.path.append(str(Path.cwd()))

from src.evaluation import calculate_metrics
from src.plotting import (
    plot_standard_metrics, 
    plot_bias_distribution, 
    plot_bias_variance,
    plot_microscope_view
)
from src.orchestration.orchestrator import run_microscope_diagnostic
from src.dgps.wgan import WGANDGP
from src.validation.wgan_validation import generate_wgan_validation_plots

def main():

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


    results_basedir = Path('results')
    dgp, est = run_microscope_diagnostic(theta=1.0)

    plot_microscope_view(dgp, est, theta=1.0, output_dir=results_basedir)
    
    print("Running WGAN Validation...")
    wgan = WGANDGP(theta=0.0) 
    generate_wgan_validation_plots(wgan, results_basedir)

    print("Analysis complete.")

if __name__ == "__main__":
    main()