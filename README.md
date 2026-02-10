# Causal Inference Project

This project simulates how well different methods (DoubleML, EconML, OLS) estimate treatment effects. It uses three different ways to generate data: TreeFriendly, PLR, and WGAN.

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