import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_standard_metrics(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    """
    Generates standard metric plots: Bias, Spurious Correlation, and Coverage.
    """
    output_dir = Path(output_dir)
    print("Generating Standard Plots...")

    # Bias vs Collider Strength
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Theta', y='bias', hue='Method', style='Method', markers=True, err_style='bars', err_kws={'capsize': 5})
    plt.axhline(0, color='black', linestyle=':', label='Zero Bias')
    plt.title('Bias vs Collider Strength')
    plt.ylabel('Bias')
    plt.xlabel('Theta (Collider Strength)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'bias_plot.png')
    plt.close()
    
    # Spurious Correlation vs Collider Strength
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Theta', y='spurious_corr', hue='Method', style='Method', markers=True)
    plt.title('Spurious Heterogeneity vs Collider Strength')
    plt.ylabel('Correlation(CATE, Hidden Collider)')
    plt.xlabel('Theta (Collider Strength)')
    plt.grid(True)
    plt.savefig(output_dir / 'spurious_corr_plot.png')
    plt.close()
    
    # Coverage Rate
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary, x='Theta', y='Coverage', hue='Method', style='Method', markers=True)
    plt.axhline(0.95, color='red', linestyle='--', label='Target (0.95)')
    plt.axhline(0.90, color='gray', linestyle=':', label='Threshold (0.90)')
    plt.title('True Coverage Rate')
    plt.ylabel('Coverage Rate')
    plt.xlabel('Theta (Collider Strength)')
    plt.grid(True)
    plt.savefig(output_dir / 'true_coverage_plot.png')
    plt.close()

def plot_bias_distribution(df: pd.DataFrame, output_dir: Path):
    """
    Generates KDE plots for bias distribution, highlighting the shift 
    from Theta=0 (Clean) to Theta=1 (High Bias).
    """
    output_dir = Path(output_dir)
    
    # DML Bias Distribution
    print("Generating Plot: Distribution of Bias (KDE) for DML...")
    plt.figure(figsize=(10, 6))

    subset = df[df['Theta'].isin([0.0, 1.0]) & df['Method'].str.contains('BadControl_DML')].copy()
    
    if not subset.empty:
        subset['Condition'] = subset['Theta'].apply(lambda x: "Theta=0.0 (No Bias)" if x == 0.0 else "Theta=1.0 (High Bias)")

        ax = sns.kdeplot(data=subset, x='tau_hat', hue='Condition', fill=True, common_norm=False, palette='viridis')
        plt.axvline(1.0, color='black', linestyle='--', label='True Effect (1.0)')
        plt.title('Distribution of DML Estimates (Bad Control)', fontsize=14)
        plt.xlabel('Estimated Treatment Effect (Tau Hat)')
        sns.move_legend(ax, "upper right")
        plt.tight_layout()
        plt.savefig(output_dir / "bias_distribution_dml.png")
        plt.close()

    # EconML Bias Distribution
    print("Generating Plot: Distribution of EconML Estimates (KDE)...")
    plt.figure(figsize=(10, 6))

    subset_econ = df[df['Theta'].isin([0.0, 1.0]) & df['Method'].str.contains('BadControl_EconML')].copy()
    
    if not subset_econ.empty:
        subset_econ['Condition'] = subset_econ['Theta'].apply(lambda x: "Theta=0.0 (No Bias)" if x == 0.0 else "Theta=1.0 (High Bias)")

        ax = sns.kdeplot(data=subset_econ, x='tau_hat', hue='Condition', fill=True, common_norm=False, palette='coolwarm')
        plt.axvline(1.0, color='black', linestyle='--', label='True Effect (1.0)')
        plt.title('Distribution of EconML Estimates (Bad Control)', fontsize=14)
        plt.xlabel('Estimated Treatment Effect (Tau Hat)')
        sns.move_legend(ax, "upper right")
        plt.tight_layout()
        plt.savefig(output_dir / "bias_distribution_econml.png")
        plt.close()

    # OLS Bias Distribution
    print("Generating Plot: Distribution of OLS Estimates (KDE)...")
    plt.figure(figsize=(10, 6))

    subset_ols = df[df['Theta'].isin([0.0, 1.0]) & df['Method'].str.contains('BadControl_OLS')].copy()
    
    if not subset_ols.empty:
        subset_ols['Condition'] = subset_ols['Theta'].apply(lambda x: "Theta=0.0 (No Bias)" if x == 0.0 else "Theta=1.0 (High Bias)")

        ax = sns.kdeplot(data=subset_ols, x='tau_hat', hue='Condition', fill=True, common_norm=False, palette='viridis')
        plt.axvline(1.0, color='black', linestyle='--', label='True Effect (1.0)')
        plt.title('Distribution of OLS Estimates (Bad Control)', fontsize=14)
        plt.xlabel('Estimated Treatment Effect (Tau Hat)')
        sns.move_legend(ax, "upper right")
        plt.tight_layout()
        plt.savefig(output_dir / "bias_distribution_ols.png")
        plt.close()

def plot_bias_variance(df: pd.DataFrame, output_dir: Path):
    """
    Generates a stacked area plot showing Bias^2 vs Variance decomposition across Thetas.
    """
    output_dir = Path(output_dir)
    print("Generating Plot: Bias-Variance Decomposition (Stacked)...")
    
    target_methods = ['BadControl_DML', 'BadControl_EconML', 'BadControl_OLS']
    
    for target_method in target_methods:
        if target_method not in df['Method'].values:
            continue
            
        metrics = []
        thetas = sorted(df['Theta'].unique())
        
        if len(thetas) < 2:
            continue

        for theta in thetas:
            sub = df[(df['Method'] == target_method) & (df['Theta'] == theta)]
            if not sub.empty:
                bias_sq = (sub['tau_hat'].mean() - 1.0)**2
                variance = sub['tau_hat'].var()
                metrics.append({'Theta': theta, 'Bias^2': bias_sq, 'Variance': variance})

        df_bv = pd.DataFrame(metrics)
        if not df_bv.empty:
            plt.figure(figsize=(10, 6))
            plt.stackplot(df_bv['Theta'], df_bv['Bias^2'], df_bv['Variance'], labels=['Bias^2', 'Variance'], colors=['#ff9999', '#66b3ff'], alpha=0.8)
            plt.title(f'Bias-Variance Decomposition ({target_method})', fontsize=14)
            plt.xlabel('Theta (Collider Strength)')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.legend(loc='upper left')
            plt.tight_layout()
            
            method_suffix = target_method.replace("BadControl_", "").lower()
            plt.savefig(output_dir / f"bias_variance_decomposition_{method_suffix}.png")
            plt.close()

def plot_microscope_view(dgp, est, theta, output_dir: Path):
    """
    Generates a scatter plot of estimated CATE vs Hidden Collider (C) to visualize spurious heterogeneity.
    """
    output_dir = Path(output_dir)
    print("Generating Plot: Microscope View (Spurious Heterogeneity)...")
    
    plt.figure(figsize=(10, 6))
    
    if not hasattr(dgp, 'C') or dgp.C is None:
        print("Skipping Microscope View: DGP does not have stored Collider C.")
        plt.close()
        return

    sns.scatterplot(x=dgp.C, y=est.cate_estimates, alpha=0.5, color='red', label='Estimated Effect')
    plt.axhline(1.0, color='green', linestyle='--', linewidth=2, label='True Effect')
    plt.title(f'Microscope View: Causal Forest Spurious Heterogeneity (Theta={theta})')
    plt.xlabel('Hidden Collider (C)')
    plt.ylabel('Estimated Treatment Effect')
    
    plt.savefig(output_dir / f'paradox1_zoom_theta_{theta}.png')
    plt.close()