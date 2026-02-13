# Causal Inference Project

This project simulates how well different methods (DoubleML, EconML, OLS) estimate treatment effects. It uses three different ways to generate data: TreeFriendly, PLR, and WGAN.

## Research Question

**To what extent does the inclusion of collider variables induce coverage collapse in Double Machine Learning and Causal Forests, causing confidence intervals to systematically fail to reflect the true bias in treatment effect estimates?**

## Setup

To run the code, you need Python **3.11.9**.

1.  Clone this folder.

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Reproducibility

The project is set up to give the exact same results every time.
-   **Random Seeds**: All random numbers are controlled by a fixed seed.
-   **WGAN**: The WGAN model uses saved weights so it always generates the same data.

## Running the Code

### 1. Run Simulations
To run all simulations and save the results:
```bash
python main.py
```
This will create `results/final_results.csv`.

### 2. Generate Plots
To analyze the results and create plots:
```bash
python analysis.py
```
The plots will be saved in the `results/plots` folder.

## Project Organization

```
.
├── analysis.py                     # Main script for data analysis and plotting
├── main.py                         # Main entry point for running simulations
├── requirements.txt                # Project dependencies
├── tune_pilot.py                   # Script for hyperparameter tuning
├── appendix/                       # Supplementary material and verification scripts
│   ├── comparison_table.tex        # Generated LaTeX comparison table
│   ├── fallacy_verification.png    # Verification plot for collider fallacy
│   ├── generate_table.py           # Script to generate comparison tables
│   └── verify_fallacy.py           # Script to verify collider fallacy logic
├── results/                        # Simulation results and output plots
│   ├── detailed_summary_metrics.csv
│   ├── final_results.csv
│   └── plots/
├── src/                            
│   ├── dgps/                       # Data Generating Processes
│   │   ├── plr_ccddhnr2018.py      # Partially Linear Regression DGP
│   │   ├── tree_friendly.py        # Tree-friendly DGP
│   │   └── wgan.py                 # Wasserstein GAN DGP
│   ├── estimators/                 # Causal Estimators
│   │   ├── dml.py                  # Double Matchine Learning Estimator
│   │   ├── econml.py               # Causal Forest Estimator
│   │   └── ols.py                  # OLS Estimator
│   ├── orchestration/              # Simulation orchestration
│   │   ├── orchestrator.py         # Parallel execution manager
│   │   └── runner.py               # Single simulation runner
│   ├── validation/                 # Validation scripts and resources
│   │   ├── data/                   # Validation data
│   │   ├── trained_models/         # Pre-trained WGAN models
│   │   └── wgan_validation.py      # WGAN validation script
│   ├── evaluation.py               # Metrics calculation logic
│   ├── plotting.py                 # Plotting 
│   ├── protocols.py                # Type protocols/interfaces
│   └── scenarios.py                # Simulation configuration scenarios
```