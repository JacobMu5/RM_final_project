import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from doubleml.datasets import fetch_401K

def generate_wgan_validation_plots(wgan_dgp, output_dir: Path):
    """
    Generates a comprehensive suite of validation plots for the WGAN.

    Args:
        wgan_dgp: The initialized WGANDGP instance.
        output_dir: Path to save the plots.
    """
    print("=" * 60)
    print("WGAN-GP Data Quality Validation (Fixed)")
    print("=" * 60)

    n_obs = 5000
    print(f"Sampling {n_obs} synthetic observations...")
    D_synth, Y_synth, W_synth = wgan_dgp.sample(n_obs, seed=42)

    data_real = fetch_401K(return_type='DataFrame')
    df_real = data_real.drop(columns=['p401'], errors='ignore')

    feature_names = [c for c in data_real.columns if c not in ['net_tfa', 'e401', 'p401']]

    df_synthetic = pd.DataFrame(W_synth[:, :len(feature_names)], columns=feature_names)
    df_synthetic['e401'] = D_synth
    df_synthetic['net_tfa'] = Y_synth

    df_real_labeled = df_real.copy()
    df_real_labeled['source'] = 'Real Data'
    df_synthetic_labeled = df_synthetic.copy()
    df_synthetic_labeled['source'] = 'Synthetic (WGAN)'

    df_combined = pd.concat([df_real_labeled, df_synthetic_labeled], axis=0)

    print("Generating plots...")

    print("   - Distribution comparison...")

    continuous_vars = ['net_tfa', 'age', 'inc', 'fsize', 'educ', 'nifa', 'tw']
    binary_vars = ['db', 'marr', 'twoearn', 'pira', 'hown', 'e401']

    all_vars_ordered = continuous_vars + binary_vars
    n_vars = len(all_vars_ordered)
    n_rows = (n_vars + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    for i, var in enumerate(all_vars_ordered):
        if i < len(axes):
            ax = axes[i]
            
            if var in binary_vars:
                real_counts = df_real[var].value_counts(normalize=True)
                
                synth_rounded = np.round(np.clip(df_synthetic[var].values, 0, 1)).astype(int)
                synth_counts = pd.Series(synth_rounded).value_counts(normalize=True)
                
                categories = [0, 1]
                real_props = [real_counts.get(float(c), 0.0) for c in categories]
                synth_props = [synth_counts.get(c, 0.0) for c in categories]
                
                x = np.arange(2)
                width = 0.35
                
                bars1 = ax.bar(x - width/2, real_props, width, label='Real', 
                              alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
                bars2 = ax.bar(x + width/2, synth_props, width, label='Synthetic',
                              alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
                
                ax.set_xticks(x)
                ax.set_xticklabels(['0', '1'])
                ax.set_ylabel('Proportion', fontweight='bold')
                ax.set_ylim(0, 1.1)
                
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0.005:
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                   f'{height:.2f}', ha='center', va='bottom', 
                                   fontsize=9, fontweight='bold')
            else:
                real_data = df_real[var].dropna().values
                synth_data = df_synthetic[var].values
                
                ax.hist(real_data, bins=30, label='Real', 
                       alpha=0.6, color='steelblue', density=True, edgecolor='black', linewidth=0.3)
                ax.hist(synth_data, bins=30, label='Synthetic', 
                       alpha=0.6, color='coral', density=True, edgecolor='black', linewidth=0.3)
                ax.set_ylabel('Density', fontweight='bold')
            
            ax.set_title(f'{var}', fontweight='bold', fontsize=12)
            ax.set_xlabel(var, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.2)

    for i in range(len(all_vars_ordered), len(axes)):
        axes[i].axis('off')

    plt.suptitle('Distribution Comparison: Real vs Synthetic - ALL Variables', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'wgan_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   - Correlation matrices...")

    key_vars = ['age', 'inc', 'educ', 'fsize', 'e401', 'net_tfa', 'marr', 'twoearn']
    corr_real = df_real[key_vars].corr()
    corr_synthetic = df_synthetic[key_vars].corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(corr_real, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, ax=axes[0], cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Real Data', fontsize=14, fontweight='bold')

    sns.heatmap(corr_synthetic, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, ax=axes[1], cbar_kws={'label': 'Correlation'})
    axes[1].set_title('Synthetic (WGAN-GP)', fontsize=14, fontweight='bold')

    plt.suptitle('Correlation Matrix Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'wgan_correlation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   - Correlation difference...")

    diff_corr = corr_synthetic - corr_real
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(diff_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                vmin=-0.3, vmax=0.3, ax=ax, cbar_kws={'label': 'Difference'})
    ax.set_title('Correlation Error: Synthetic - Real', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'wgan_correlation_error.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   - Q-Q plots...")

    continuous_vars_qq = ['age', 'inc', 'educ', 'fsize']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, var in enumerate(continuous_vars_qq):
        ax = axes[i]
        
        real_data = df_real[var].dropna().values
        synth_data = df_synthetic[var].values
        
        n = min(len(real_data), len(synth_data))
        quantiles = np.linspace(0, 1, n)
        
        real_quantiles = np.quantile(real_data, quantiles)
        synth_quantiles = np.quantile(synth_data, quantiles)
        
        ax.scatter(real_quantiles, synth_quantiles, alpha=0.5, s=20)
        
        min_val = min(real_quantiles.min(), synth_quantiles.min())
        max_val = max(real_quantiles.max(), synth_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Match')
        
        ax.set_xlabel(f'Real {var}', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Synthetic {var}', fontsize=11, fontweight='bold')
        ax.set_title(var, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Q-Q Plots: Real vs Synthetic (Continuous Variables)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'wgan_qq_plots.png', dpi=300, bbox_inches='tight')
    plt.close()



    print("\nGenerated details:")
    print("  - Results saved to:", output_dir)

