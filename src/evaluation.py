import pandas as pd
import numpy as np

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates performance metrics for simulation results.
    Assumes True Treatment Effect = 1.0 (TreeFriendlyDGP default).
    """
    required_cols = ['tau_hat', 'ci_lower', 'ci_upper', 'scenario']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame missing one of {required_cols}")

    true_effect = 1.0 
    
    # Row-level performance metrics
    df['mse'] = (df['tau_hat'] - true_effect) ** 2
    df['abs_error'] = (df['tau_hat'] - true_effect).abs()
    df['bias'] = df['tau_hat'] - true_effect
    df['ci_length'] = df['ci_upper'] - df['ci_lower']
    
    # Coverage: Does CI cover the true value?
    df['coverage'] = (df['ci_lower'] <= true_effect) & (true_effect <= df['ci_upper'])
    
    # Power: Does CI exclude 0?
    df['reject_null'] = (df['ci_lower'] > 0) | (df['ci_upper'] < 0)

    # Parse Method and Theta from scenario string if not present
    if 'Theta' not in df.columns or 'Method' not in df.columns:
        def parse_scenario(name):
            try:
                parts = name.split('_theta_')
                left = parts[0]
                theta = float(parts[1])

                # left = "{DGP}_{Variant}_{Estimator}"  (e.g., "TreeFriendly_BadControl_OLS")
                dgp = left.split('_', 1)[0]
                method = left.split('_', 1)[1] if '_' in left else left

                return pd.Series([dgp, method, theta], index=['DGP', 'Method', 'Theta'])
            except:
                return pd.Series(['Unknown', 'Unknown', np.nan], index=['DGP', 'Method', 'Theta'])

        df[['DGP', 'Method', 'Theta']] = df['scenario'].apply(parse_scenario)

    # Centered Coverage: Fraction of CIs covering the *mean* estimate of the method.
    # Evaluates the validity of the inference relative to the estimator's own target (ignoring bias).
    scenario_means = df.groupby(['Method', 'Theta'])['tau_hat'].transform('mean')
    df['centered_coverage'] = (df['ci_lower'] <= scenario_means) & (scenario_means <= df['ci_upper'])

    # Aggregate metrics by Method and Theta
    summary = df.groupby(['Method', 'Theta']).agg({
        'bias': ['mean', 'std'],
        'tau_hat': ['mean', 'std', 'var'],
        'ci_length': 'mean',
        'spurious_corr': 'mean', 
        'mse': 'mean', 
        'abs_error': 'mean', 
        'reject_null': 'mean', 
        'coverage': 'mean',
        'centered_coverage': 'mean'
    }).reset_index()
    
    summary.columns = [
        'Method', 'Theta', 
        'Bias_Mean', 'Bias_Std', 
        'Tau_Mean', 'Tau_Std', 'Tau_Var', 
        'CI_Length_Mean', 
        'Spurious_Corr_Mean', 
        'MSE', 
        'MAE', 
        'Power', 
        'Coverage',
        'Centered_Coverage'
    ]
    
    summary['RMSE'] = np.sqrt(summary['MSE'])
    
    return summary, df