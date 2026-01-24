import pandas as pd
import numpy as np

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates performance metrics for simulation results."""
    required_cols = ['tau_hat', 'ci_lower', 'ci_upper', 'scenario']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame missing one of {required_cols}")

    df['true_effect'] = df['tau_hat'] - df['bias']
    
    df['mse'] = (df['tau_hat'] - df['true_effect']) ** 2
    
    df['abs_error'] = (df['tau_hat'] - df['true_effect']).abs()
    
    df['ci_length'] = df['ci_upper'] - df['ci_lower']
    
    df['coverage'] = (df['ci_lower'] <= df['true_effect']) & (df['true_effect'] <= df['ci_upper'])
    
    df['reject_null'] = (df['ci_lower'] > 0) | (df['ci_upper'] < 0)

    if 'Theta' not in df.columns or 'Method' not in df.columns or 'DGP' not in df.columns:
        def parse_scenario(name):
            try:
                parts = name.split('_theta_')
                full_method_name = parts[0]
                theta = float(parts[1])

                if full_method_name.startswith('TreeFriendly_'):
                    dgp = 'TreeFriendly'
                    method = full_method_name.replace('TreeFriendly_', '')
                elif full_method_name.startswith('PLR_'):
                    dgp = 'PLR'
                    method = full_method_name.replace('PLR_', '')
                elif full_method_name.startswith('WGAN_401k_'):
                    dgp = 'WGAN_401k'
                    method = full_method_name.replace('WGAN_401k_', '')
                else:
                    dgp = 'Unknown'
                    method = full_method_name

                return pd.Series(
                    [dgp, method, theta],
                    index=['DGP', 'Method', 'Theta']
                )
            except Exception:
                return pd.Series(
                    ['Unknown', 'Unknown', np.nan],
                    index=['DGP', 'Method', 'Theta']
                )

        df[['DGP', 'Method', 'Theta']] = df['scenario'].apply(parse_scenario)

    scenario_means = df.groupby(['DGP', 'Method', 'Theta'])['tau_hat'].transform('mean')
    df['centered_coverage'] = (
        (df['ci_lower'] <= scenario_means) &
        (scenario_means <= df['ci_upper'])
    )

    summary = df.groupby(['DGP', 'Method', 'Theta']).agg({
        'bias': ['mean', 'std'],
        'tau_hat': ['mean', 'std', 'var'],
        'ci_length': 'mean',
        'spurious_corr_mult': 'mean',
        'spurious_corr_linear': 'mean', 
        'mse': 'mean', 
        'abs_error': 'mean', 
        'reject_null': 'mean', 
        'coverage': 'mean',
        'centered_coverage': 'mean'
    }).reset_index()
    
    summary.columns = [
        'DGP', 'Method', 'Theta', 
        'Bias_Mean', 'Bias_Std', 
        'Tau_Mean', 'Tau_Std', 'Tau_Var', 
        'CI_Length_Mean', 
        'Spurious_Corr_Mult_Mean',
        'Spurious_Corr_Linear_Mean',
        'MSE', 
        'MAE', 
        'Power', 
        'Coverage',
        'Centered_Coverage'
    ]

    summary['RMSE'] = np.sqrt(summary['MSE'])
    
    return summary, df