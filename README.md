# Research Methods Final Project: Collider Bias & Causal Forests

This repository contains the simulation code for our final project investigating spurious heterogeneity.

### Research Question
*"How does the functional form of a collider (Linear, Multiplicative, or Threshold) dictate the structure of spurious heterogeneity, and to what extent does this compromise **estimation accuracy (Bias, MSE)** and **inference validity (Coverage Probability)** in non-parametric Causal Forests versus semi-parametric Double Machine Learning?"*

> **Project Scope & Metrics:**
> * **Bias Structure:** We analyze the distinct "shapes" of bias introduced by different collider functions.
> * **Inference Failure:** We specifically test for **Coverage Collapse**—instances where standard confidence intervals fail to cover the true effect.
> * **Method Comparison:** We contrast the robustness of **Causal Forests** (which may "learn" the noise) against **DML Partial Linear Regression** (which assumes linearity).
> * **Note:** Ordinary Least Squares (OLS) will serve as a baseline in future iterations.
---

### How to Run

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Clean Old Results**
    **Important:** Before running a new simulation, delete any existing files in the `results/` folder (including the `plots/` subdirectory) to ensure a clean run.

3.  **Run Simulations**
    This script runs the scenarios and saves the raw data to `results/final_results.csv`.
    ```bash
    python main.py
    ```
    > **Warning:** This simulation takes **over an hour** to complete depending on your hardware. Please allow it to finish uninterrupted.

4.  **Generate Plots & Tables**
    This script reads the new results and creates the "Microscope View" and other plots in `results/plots/`.
    ```bash
    python analysis.py
    ```

### Project Structure
* `main.py`: Entry point for running simulations.
* `analysis.py`: Generates tables and figures from the simulation data.
* `src/`: Contains the DGPs (`tree_friendly.py`), Estimators (`econml.py`, `dml.py`), and orchestration logic.