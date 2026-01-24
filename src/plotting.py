import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_standard_metrics(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    """Generates standard metric plots: Bias, Spurious Correlation, and Coverage."""
    output_dir = Path(output_dir)
    print("Generating Standard Plots...")

    all_thetas = sorted(df['Theta'].unique())
    naive_rows = df[df['Method'].str.contains("Naive")].copy()
    
    if not naive_rows.empty:
        virtual_rows = []
        for method in naive_rows['Method'].unique():
            method_subset = naive_rows[naive_rows['Method'] == method]
            base_data = method_subset[method_subset['Theta'] == 0.0]
            
            if base_data.empty and not method_subset.empty:
                base_data = method_subset.iloc[[0]]
            
            if not base_data.empty:
                for t in all_thetas:
                    if t == 0.0: continue
                    
                    filled = base_data.copy()
                    filled['Theta'] = t 
                    virtual_rows.append(filled)
        
        if virtual_rows:
            df_virtual = pd.concat(virtual_rows, ignore_index=True)
            df = pd.concat([df, df_virtual], ignore_index=True)   

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='Theta',
        y='bias',
        hue='Method',
        style='Method',
        markers=True,
        err_style='bars',
        err_kws={'capsize': 5}
    )
    plt.axhline(0, color='black', linestyle=':', label='Zero Bias')
    plt.title('Bias vs Collider Strength')
    plt.ylabel('Bias')
    plt.xlabel('Theta (Collider Strength)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / 'bias_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='Theta',
        y='spurious_corr_mult',
        hue='Method',
        style='Method',
        markers=True
    )
    plt.title('Spurious Heterogeneity vs Multiplicative Collider Strength')
    plt.ylabel('Correlation(CATE, Multiplicative Collider)')
    plt.xlabel('Theta (Collider Strength)')
    plt.grid(True)
    plt.savefig(output_dir / 'spurious_corr_mult_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='Theta',
        y='spurious_corr_linear',
        hue='Method',
        style='Method',
        markers=True
    )
    plt.title('Spurious Heterogeneity vs Linear Collider Strength')
    plt.ylabel('Correlation(CATE, Linear Collider)')
    plt.xlabel('Theta (Collider Strength)')
    plt.grid(True)
    plt.savefig(output_dir / 'spurious_corr_linear_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=summary,
        x='Theta',
        y='Coverage',
        hue='Method',
        style='Method',
        markers=True,
        errorbar=None
    )
    plt.axhline(0.95, color='red', linestyle='--', label='Target (0.95)')
    plt.axhline(0.90, color='gray', linestyle=':', label='Threshold (0.90)')
    plt.title('True Coverage Rate')
    plt.ylabel('Coverage Rate')
    plt.xlabel('Theta (Collider Strength)')
    plt.grid(True)
    plt.savefig(output_dir / 'true_coverage_plot.png')
    plt.close()

    if 'Centered_Coverage' in summary.columns:
        plt.figure(figsize=(10, 7))
        sns.lineplot(
            data=summary,
            x='Theta',
            y='Centered_Coverage',
            hue='Method',
            style='Method',
            markers=True,
            errorbar=None
        )
        plt.axhline(0.95, color='red', linestyle='--', label='Target (0.95)')
        plt.axhline(0.90, color='gray', linestyle=':', label='Threshold (0.90)')
        plt.title('Centered Coverage Rate (validity check)')
        plt.ylabel('Centered Coverage Rate')
        plt.xlabel('Theta (Collider Strength)')
        plt.ylim(0, 1.05)
        plt.grid(True)
        
        plt.figtext(0.5, 0.02, 
                    "Measures if CI captures the Estimator's Mean (ignoring Bias).\n"
                    "High Centered Coverage + Low True Coverage = 'Precisely Wrong' (Biased but Valid CI).", 
                    ha='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(output_dir / 'centered_coverage_plot.png')
        plt.close()


def plot_bias_distribution(df: pd.DataFrame, output_dir: Path):
    """KDE plots for bias distribution across methods."""
    output_dir = Path(output_dir)

    def kde_plot(method_name, filename, palette='viridis'):
        subset = df[
            (df['Theta'].isin([0.0, 1.0])) &
            (df['Method'] == method_name)
        ].copy()

        if not subset.empty:
            subset['Condition'] = subset['Theta'].apply(
                lambda x: "Theta=0.0 (No Bias)" if x == 0.0 else "Theta=1.0 (High Bias)"
            )
            plt.figure(figsize=(10, 6))
            ax = sns.kdeplot(
                data=subset,
                x='tau_hat',
                hue='Condition',
                fill=True,
                common_norm=False,
                palette=palette
            )
            plt.axvline(1.0, color='black', linestyle='--', label='True Effect (1.0)')
            plt.title(f'Distribution of {method_name} Estimates', fontsize=14)
            plt.xlabel('Estimated Treatment Effect (Tau Hat)')
            sns.move_legend(ax, "upper right")
            plt.tight_layout()
            plt.savefig(output_dir / filename)
            plt.close()

    kde_plot('BadControl_DML', 'bias_distribution_dml_mult.png', palette='viridis')
    kde_plot('BadControl_EconML', 'bias_distribution_econml_mult.png', palette='coolwarm')
    kde_plot('BadControl_OLS', 'bias_distribution_ols_mult.png', palette='magma')

    kde_plot('LinearCollider_DML', 'bias_distribution_dml_linear.png', palette='viridis')
    kde_plot('LinearCollider_EconML', 'bias_distribution_econml_linear.png', palette='coolwarm')
    kde_plot('LinearCollider_OLS', 'bias_distribution_ols_linear.png', palette='magma')


def plot_bias_variance(df: pd.DataFrame, output_dir: Path):
    """Bias^2 vs Variance decomposition for each method."""
    output_dir = Path(output_dir)
    print("Generating Plot: Bias-Variance Decomposition (Stacked)...")

    target_methods = [
        'BadControl_DML', 'BadControl_EconML', 'BadControl_OLS',
        'LinearCollider_DML', 'LinearCollider_EconML', 'LinearCollider_OLS'
    ]

    for target_method in target_methods:
        if target_method not in df['Method'].values:
            continue

        metrics = []
        thetas = sorted(df['Theta'].unique())

        for theta in thetas:
            sub = df[(df['Method'] == target_method) & (df['Theta'] == theta)]
            if not sub.empty:
                bias_sq = (sub['tau_hat'].mean() - 1.0) ** 2
                variance = sub['tau_hat'].var()
                metrics.append({
                    'Theta': theta,
                    'Bias^2': bias_sq,
                    'Variance': variance
                })

        df_bv = pd.DataFrame(metrics)
        if not df_bv.empty:
            plt.figure(figsize=(10, 6))
            plt.stackplot(
                df_bv['Theta'],
                df_bv['Bias^2'],
                df_bv['Variance'],
                labels=['Bias^2', 'Variance'],
                colors=['#ff9999', '#66b3ff'],
                alpha=0.8
            )
            plt.title(f'Bias-Variance Decomposition ({target_method})', fontsize=14)
            plt.xlabel('Theta (Collider Strength)')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.legend(loc='upper left')
            plt.tight_layout()

            if target_method.startswith("BadControl_"):
                method_suffix = target_method.replace("BadControl_", "").lower() + "_mult"
            elif target_method.startswith("LinearCollider_"):
                method_suffix = target_method.replace("LinearCollider_", "").lower() + "_linear"
            else:
                method_suffix = target_method.lower()

            plt.savefig(output_dir / f"bias_variance_decomposition_{method_suffix}.png")
            plt.close()


def plot_microscope_view(dgp, est, theta, output_dir: Path, filename_suffix: str = ""):
    """Scatter plot of estimated CATE vs colliders."""
    output_dir = Path(output_dir)
    print(f"Generating Plot: Microscope View ({filename_suffix})...")

    if not hasattr(est, 'cate_estimates') or est.cate_estimates is None:
        print("Skipping Microscope View: Estimator has no CATE estimates.")
        return

    plt.figure(figsize=(10, 6))
    cols = []
    if hasattr(dgp, 'C') and dgp.C is not None:
        corr_val = np.corrcoef(dgp.C.flatten(), est.cate_estimates.flatten())[0,1]
        cols.append(f"Corr(C, $\\hat{{\\tau}}$) = {corr_val:.2f}")
        sns.scatterplot(
            x=dgp.C,
            y=est.cate_estimates,
            alpha=0.5,
            color='red',
            label='Multiplicative Collider'
        )
    if hasattr(dgp, 'C_linear') and dgp.C_linear is not None:
        corr_val_lin = np.corrcoef(dgp.C_linear.flatten(), est.cate_estimates.flatten())[0,1]
        cols.append(f"Corr(C_lin, $\\hat{{\\tau}}$) = {corr_val_lin:.2f}")
        sns.scatterplot(
            x=dgp.C_linear,
            y=est.cate_estimates,
            alpha=0.5,
            color='blue',
            label='Linear Collider'
        )

    plt.axhline(1.0, color='green', linestyle='--', linewidth=2, label='True Effect')
    
    metrics_str = " | ".join(cols)
    title_text = f'Microscope View: {metrics_str}\n(Theta={theta}) {filename_suffix}'
    plt.title(title_text)
    
    plt.xlabel('Collider Value')
    plt.ylabel('Estimated Treatment Effect')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'paradox1_zoom_theta_{theta}{filename_suffix}.png')
    plt.close()


def plot_cate_distribution(est, theta, output_dir: Path):
    """Plots the histogram of CATE estimates."""
    output_dir = Path(output_dir)
    print("Generating Plot: CATE Distribution...")

    if not hasattr(est, 'cate_estimates') or est.cate_estimates is None:
        return

    cates = est.cate_estimates.flatten()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(cates, kde=True, color='purple', bins=30)
    plt.axvline(1.0, color='black', linestyle='--', linewidth=2, label='True Effect (1.0)')
    plt.title(f'Spurious Heterogeneity Distribution (Theta={theta})', fontsize=14)
    plt.xlabel('Estimated CATE')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'cate_distribution_theta_{theta}.png')
    plt.close()