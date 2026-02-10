import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import matplotlib

matplotlib.use("Agg")

sys.path.append(str(Path.cwd()))

try:
    from src.evaluation import calculate_metrics
    from src.plotting import (
        plot_microscope_view,
        plot_bias_comparison,
        plot_coverage_comparison,
        plot_bias_variance_grid,
        plot_tau_distribution_1x3_by_dgp,
    )
    from src.orchestration.orchestrator import run_microscope_diagnostic
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PathConfig:
    """Configuration for file paths used in analysis."""
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.results_file = self.base_dir / "final_results.csv"
        self.plots_dir = self.base_dir / "plots"
        self.summary_file = self.base_dir / "detailed_summary_metrics.csv"
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


def load_results(results_path: Path) -> Optional[pd.DataFrame]:
    """Load results from CSV file.

    Args:
        results_path: Path to the results CSV file.

    Returns:
        DataFrame containing results, or None if file doesn't exist.
    """
    if not results_path.exists():
        logger.warning(f"Results file not found at {results_path}")
        return None
    
    logger.info(f"Loading results from {results_path}")
    return pd.read_csv(results_path)


def generate_standard_plots(df: pd.DataFrame, summary: pd.DataFrame, 
                           plots_dir: Path) -> None:
    """Generate all standard analysis plots.

    Args:
        df: DataFrame with individual simulation results including tau_hat.
        summary: DataFrame with aggregated metrics including Coverage.
        plots_dir: Directory where plots will be saved.
    """
    plot_functions = [
        ("Bias Comparison", lambda: plot_bias_comparison(df, plots_dir)),
        ("Coverage Comparison", lambda: plot_coverage_comparison(summary, plots_dir)),
        ("Bias-Variance Grid", lambda: plot_bias_variance_grid(df, plots_dir)),
        ("Tau Distribution", lambda: plot_tau_distribution_1x3_by_dgp(df, plots_dir)),
    ]
    
    for name, plot_func in plot_functions:
        try:
            logger.info(f"Generating {name}...")
            plot_func()
        except Exception as e:
            logger.error(f"Failed to generate {name}: {e}")


def generate_microscope_diagnostics(paths: PathConfig, theta: float = 1.0) -> None:
    """Generate microscope diagnostic plots for all DGPs.

    Args:
        paths: PathConfig object containing output paths.
        theta: Theta value for microscope diagnostic.
    """
    logger.info(f"Running microscope diagnostic (theta={theta})...")
    
    try:
        outputs = run_microscope_diagnostic(theta=theta)
    except Exception as e:
        logger.error(f"Failed to run microscope diagnostic: {e}")
        return
    
    for dgp_name, (dgp, est) in outputs.items():
        try:
            logger.info(f"Generating microscope view for {dgp_name}")
            plot_microscope_view(
                dgp=dgp,
                est=est,
                theta=theta,
                output_dir=paths.base_dir,
                filename_suffix=f"_{dgp_name}",
            )
        except Exception as e:
            logger.error(f"Failed to generate microscope view for {dgp_name}: {e}")


def process_metrics_and_plots(paths: PathConfig) -> None:
    """Process performance metrics and generate plots.

    Args:
        paths: PathConfig object containing file paths.
    """
    df = load_results(paths.results_file)
    
    if df is None:
        logger.warning("Skipping metrics and standard plots (no results file)")
        return
    
    logger.info("Calculating metrics...")
    try:
        summary, df = calculate_metrics(df)
        summary.to_csv(paths.summary_file, index=False)
        logger.info(f"Saved summary metrics to {paths.summary_file}")
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        return
    
    generate_standard_plots(df, summary, paths.plots_dir)


def main() -> None:
    """Main entry point for analysis pipeline."""
    logger.info("Starting analysis pipeline...")
    
    paths = PathConfig(base_dir="results")
    paths.ensure_directories()
    
    process_metrics_and_plots(paths)
    generate_microscope_diagnostics(paths, theta=1.0)
    
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()