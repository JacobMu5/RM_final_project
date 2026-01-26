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
    """
    KDE plots of estimate distributions by DGP and estimator.

    For PLR/TreeFriendly: includes True Effect line at 1.0 (simulation truth).
    For WGAN: NO true effect line (real data, unknown truth). Interpret as sensitivity.
    """
    output_dir = Path(output_dir)

    required_cols = ['DGP', 'Method', 'Theta', 'tau_hat']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame must contain DGP, Method, Theta, and tau_hat columns.")

    dgps = sorted(df['DGP'].dropna().unique())
    estimators = ['DoubleML', 'EconML', 'OLS']

    for dgp in dgps:
        df_dgp = df[df['DGP'] == dgp].copy()

        for est in estimators:
            print(f"Generating Estimate Distribution (KDE): DGP={dgp}, Estimator={est}")

            subsets = []

            # Naive specification
            naive = df_dgp[df_dgp['Method'].str.contains(f'Naive_{est}', regex=False)].copy()
            if not naive.empty:
                naive['Spec'] = 'Naive'
                subsets.append(naive)

            # Multiplicative collider (theta = 1 only)
            bad1 = df_dgp[
                (df_dgp['Method'].str.contains(f'BadControl_{est}', regex=False)) &
                (df_dgp['Theta'] == 1.0)
            ].copy()
            if not bad1.empty:
                bad1['Spec'] = 'Multiplicative Collider (Theta=1.0)'
                subsets.append(bad1)

            # Linear collider (theta = 1 only)
            lin1 = df_dgp[
                (df_dgp['Method'].str.contains(f'LinearCollider_{est}', regex=False)) &
                (df_dgp['Theta'] == 1.0)
            ].copy()
            if not lin1.empty:
                lin1['Spec'] = 'Linear Collider (Theta=1.0)'
                subsets.append(lin1)

            if not subsets:
                print(f"Skipping: No data for DGP={dgp}, Estimator={est}")
                continue

            plot_df = pd.concat(subsets, ignore_index=True)

            plt.figure(figsize=(10, 6))
            ax = sns.kdeplot(
                data=plot_df,
                x='tau_hat',
                hue='Spec',
                fill=True,
                common_norm=False
            )

            if dgp != 'WGAN':
                plt.axvline(1.0, color='black', linestyle='--', label='True Effect (1.0)')
                plt.title(f'{dgp}: Distribution of {est} Estimates', fontsize=14)
            else:
                plt.title(f'{dgp}: Distribution of {est} Estimates (No Ground Truth)', fontsize=14)

            plt.xlabel('Estimated Treatment Effect (Tau Hat)')
            sns.move_legend(ax, "upper right")
            plt.tight_layout()

            filename = (
                f"estimate_distribution_{est.lower()}_{dgp}.png"
                if dgp == 'WGAN'
                else f"bias_distribution_{est.lower()}_{dgp}.png"
            )
            plt.savefig(output_dir / filename)
            plt.close()

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
    """
    Scatter plot of estimated CATE vs colliders.
    Generates ONE plot per DGP.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dgp_name = getattr(dgp, "name", dgp.__class__.__name__)

    print(f"Generating Microscope View: DGP={dgp_name} {filename_suffix}")

    if not hasattr(est, "cate_estimates") or est.cate_estimates is None:
        print("Skipping Microscope View: Estimator has no CATE estimates.")
        return

    plt.figure(figsize=(10, 6))
    title_parts = []

    # --- Multiplicative collider ---
    if hasattr(dgp, "C") and dgp.C is not None:
        corr_val = np.corrcoef(
            dgp.C.flatten(),
            est.cate_estimates.flatten()
        )[0, 1]

        title_parts.append(f"Corr(C, τ̂) = {corr_val:.2f}")

        sns.scatterplot(
            x=dgp.C.flatten(),
            y=est.cate_estimates.flatten(),
            alpha=0.45,
            color="red",
            label="Multiplicative Collider"
        )

    # --- Linear collider ---
    if hasattr(dgp, "C_linear") and dgp.C_linear is not None:
        corr_val_lin = np.corrcoef(
            dgp.C_linear.flatten(),
            est.cate_estimates.flatten()
        )[0, 1]

        title_parts.append(f"Corr(C_lin, τ̂) = {corr_val_lin:.2f}")

        sns.scatterplot(
            x=dgp.C_linear.flatten(),
            y=est.cate_estimates.flatten(),
            alpha=0.45,
            color="blue",
            label="Linear Collider"
        )

    # --- Reference line ---
    if dgp_name.lower() == "wgan":
        ref = 6250.951
        ref_label = "Baseline (WGAN)"
    else:
        ref = 1.0
        ref_label = "True Effect"

    plt.axhline(ref, color="green", linestyle="--", linewidth=2, label=ref_label)

    metrics_str = " | ".join(title_parts)
    plt.title(
        f"Microscope View: {dgp_name}\n{metrics_str} (Theta={theta})",
        fontsize=13
    )

    plt.xlabel("Collider Value")
    plt.ylabel("Estimated Treatment Effect")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fname = f"microscope_{dgp_name.lower()}_theta_{theta}{filename_suffix}.png"
    plt.savefig(output_dir / fname, dpi=200)
    plt.close()

    print(f"Saved: {fname}")

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

def plot_rmse_comparison(df: pd.DataFrame, output_dir: Path):
    """
    RMSE comparison plot, split by DGP and scenario_type.

    - Excludes WGAN (no ground truth).
    - Includes all thetas available in the data.
    - Panels: rows = DGP (PLR, TreeFriendly), cols = scenario_type
    - Lines: estimator (DoubleML, EconML, OLS)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ['DGP', 'Method', 'Theta', 'RMSE']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame must contain DGP, Method, Theta, and RMSE columns.")

    print("Generating Plot: RMSE Comparison (PLR & TreeFriendly)...")

    plot_df = df.copy()
    plot_df = plot_df[plot_df['DGP'].isin(['PLR', 'TreeFriendly'])].copy()

    def parse_scenario_type(method: str) -> str:
        if method.startswith('Naive_'):
            return 'Naive'
        if method.startswith('BadControl_'):
            return 'MultiplicativeCollider'
        if method.startswith('LinearCollider_'):
            return 'LinearCollider'
        return 'Other'

    def parse_estimator(method: str) -> str:
        if 'DoubleML' in method or 'DML' in method:
            return 'DoubleML'
        if 'EconML' in method:
            return 'EconML'
        if 'OLS' in method:
            return 'OLS'
        return 'Unknown'

    plot_df['scenario_type'] = plot_df['Method'].apply(parse_scenario_type)
    plot_df['estimator'] = plot_df['Method'].apply(parse_estimator)

    plot_df = plot_df[
        plot_df['scenario_type'].isin(['Naive', 'MultiplicativeCollider', 'LinearCollider']) &
        plot_df['estimator'].isin(['DoubleML', 'EconML', 'OLS'])
    ].copy()

    plot_df = plot_df.sort_values(['DGP', 'scenario_type', 'Theta', 'estimator'])

    dgps = ['PLR', 'TreeFriendly']
    scenario_order = ['MultiplicativeCollider', 'LinearCollider', 'Naive']
    scenario_title = {
        'MultiplicativeCollider': 'Multiplicative',
        'LinearCollider': 'Linear',
        'Naive': 'Naive'
    }

    fig, axes = plt.subplots(
        nrows=len(dgps),
        ncols=len(scenario_order),
        figsize=(12, 5),
        sharex=True
    )

    fig.suptitle("RMSE Comparison: PLR & TreeFriendly Scenarios", fontsize=14)

    for i, dgp in enumerate(dgps):
        for j, scen in enumerate(scenario_order):
            ax = axes[i, j]
            sub = plot_df[(plot_df['DGP'] == dgp) & (plot_df['scenario_type'] == scen)]

            if sub.empty:
                ax.set_axis_off()
                continue

            sns.lineplot(
                data=sub,
                x='Theta',
                y='RMSE',
                hue='estimator',
                marker='o',
                ax=ax,
                errorbar=None,
                legend=False   # <- CLAVE
            )

            ax.set_title(f"dgp = {dgp} | scenario = {scenario_title[scen]}", fontsize=10)
            ax.set_xlabel("theta" if i == len(dgps) - 1 else "")
            ax.set_ylabel("rmse" if j == 0 else "")
            ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=sns.color_palette()[0], marker='o', label='DoubleML'),
        Line2D([0], [0], color=sns.color_palette()[1], marker='o', label='EconML'),
        Line2D([0], [0], color=sns.color_palette()[2], marker='o', label='OLS'),
    ]

    fig.legend(
        handles=legend_elements,
        title="estimator",
        loc="center right",
        bbox_to_anchor=(0.93, 0.5),
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 0.80, 0.95])
    plt.savefig(output_dir / "rmse_comparison_plr_treefriendly.png", dpi=200)
    plt.close()

    print("Saved: rmse_comparison_plr_treefriendly.png")


def plot_bias_comparison(df: pd.DataFrame, output_dir: Path):
    """
    2x3 layout:
      rows = scenario (Multiplicative, Linear)
      cols = DGP (TreeFriendly, PLR, WGAN)
    - No Naive
    - Para PLR/TreeFriendly: usa bias_mean (o bias)
    - Para WGAN: usa sensibilidad = mean(tau_hat(theta)) - mean(tau_hat(theta=0))
    - Una sola leyenda afuera, sin invadir paneles
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ['DGP', 'Method', 'Theta']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame must contain DGP, Method, and Theta columns.")

    # Elegir columna de bias si existe
    bias_col = None
    if 'Bias_Mean' in df.columns:
        bias_col = 'Bias_Mean'
    elif 'bias_mean' in df.columns:
        bias_col = 'bias_mean'
    elif 'bias' in df.columns:
        bias_col = 'bias'

    # Para WGAN necesitamos tau_hat para armar la sensibilidad
    has_tau_hat = 'tau_hat' in df.columns

    dgps = ['TreeFriendly', 'PLR', 'WGAN']
    scenarios = ['Multiplicative', 'Linear']

    plot_df = df.copy()
    plot_df = plot_df[plot_df['DGP'].isin(dgps)].copy()

    def parse_scenario(method: str) -> str:
        if method.startswith('BadControl_'):
            return 'Multiplicative'
        if method.startswith('LinearCollider_'):
            return 'Linear'
        return 'Other'

    def parse_estimator(method: str) -> str:
        if 'DoubleML' in method or 'DML' in method:
            return 'DoubleML'
        if 'EconML' in method:
            return 'EconML'
        if 'OLS' in method:
            return 'OLS'
        return 'Unknown'

    plot_df['scenario'] = plot_df['Method'].apply(parse_scenario)
    plot_df['estimator'] = plot_df['Method'].apply(parse_estimator)

    plot_df = plot_df[
        plot_df['scenario'].isin(scenarios) &
        plot_df['estimator'].isin(['DoubleML', 'EconML', 'OLS'])
    ].copy()

    # --- construir la métrica a graficar ---
    # Para PLR/TreeFriendly: bias (si existe)
    # Para WGAN: delta vs theta=0 usando tau_hat (si está disponible)
    metric_rows = []

    # (A) PLR/TreeFriendly
    if bias_col is None:
        raise ValueError("Para PLR/TreeFriendly necesito Bias_Mean/bias_mean/bias. No encontré ninguna.")
    sim = plot_df[plot_df['DGP'].isin(['PLR', 'TreeFriendly'])].copy()
    if not sim.empty:
        sim_metric = sim[['DGP', 'scenario', 'estimator', 'Theta', bias_col]].copy()
        sim_metric = sim_metric.rename(columns={bias_col: 'metric'})
        sim_metric['metric_name'] = 'bias_mean'
        metric_rows.append(sim_metric)

    # (B) WGAN (sensibilidad)
    wgan = plot_df[plot_df['DGP'] == 'WGAN'].copy()
    if not wgan.empty:
        if not has_tau_hat:
            # si no hay tau_hat, al menos no rompas: no puedes construir delta
            # (mejor que graficar bias sin sentido)
            print("Warning: WGAN sin tau_hat -> no puedo calcular sensibilidad vs theta=0. Se omite WGAN.")
        else:
            # agregamos por (scenario, estimator, theta)
            wgan_mean = (
                wgan.groupby(['scenario', 'estimator', 'Theta'], as_index=False)['tau_hat']
                .mean()
                .rename(columns={'tau_hat': 'tau_mean'})
            )
            # baseline theta=0 por (scenario, estimator)
            base = wgan_mean[wgan_mean['Theta'] == 0.0][['scenario', 'estimator', 'tau_mean']] \
                .rename(columns={'tau_mean': 'tau_base'})

            wgan_m = wgan_mean.merge(base, on=['scenario', 'estimator'], how='left')
            wgan_m['metric'] = wgan_m['tau_mean'] - wgan_m['tau_base']
            wgan_m['DGP'] = 'WGAN'
            wgan_m['metric_name'] = 'delta_tau_vs_theta0'
            metric_rows.append(wgan_m[['DGP', 'scenario', 'estimator', 'Theta', 'metric', 'metric_name']])

    metric_df = pd.concat(metric_rows, ignore_index=True)
    metric_df = metric_df.sort_values(['scenario', 'DGP', 'Theta', 'estimator'])

    # --- plot 2x3 ---
    fig, axes = plt.subplots(
        nrows=len(scenarios),
        ncols=len(dgps),
        figsize=(15, 6),
        sharex=True,
        sharey=False  # <- CLAVE: evita que WGAN aplaste a los demás
    )
    fig.suptitle("Bias Comparison: PLR, TreeFriendly & WGAN", fontsize=14)

    legend_handles, legend_labels = None, None

    for i, scen in enumerate(scenarios):
        for j, dgp in enumerate(dgps):
            ax = axes[i, j]

            sub = metric_df[(metric_df['scenario'] == scen) & (metric_df['DGP'] == dgp)].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            sns.lineplot(
                data=sub,
                x='Theta',
                y='metric',
                hue='estimator',
                marker='o',
                ax=ax,
                errorbar=None
            )

            # Capturar leyenda una sola vez
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            if ax.get_legend() is not None:
                ax.get_legend().remove()

            ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
            ax.set_title(f"dgp = {dgp} | scenario = {scen}", fontsize=10)
            ax.set_xlabel("theta" if i == len(scenarios) - 1 else "")
            ax.set_ylabel("bias_mean" if (j == 0 and dgp != 'WGAN') else ("Δ tau vs θ=0" if (j == 0 and dgp == 'WGAN') else ""))
            ax.grid(True, alpha=0.3)

    # Leyenda afuera (sin invadir)
    fig.legend(
        legend_handles,
        legend_labels,
        title="estimator",
        loc="center left",
        bbox_to_anchor=(0.90, 0.5),
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(output_dir / "bias_comparison_2x3_plr_treefriendly_wgan.png", dpi=200)
    plt.close()
    print("Saved: bias_comparison_2x3_plr_treefriendly_wgan.png")

def plot_coverage_comparison(summary: pd.DataFrame, output_dir: Path):
    """
    Coverage comparison plot, split by scenario and DGP.

    - Includes PLR, TreeFriendly, WGAN
    - Excludes Naive
    - Panels: rows = scenario (Multiplicative, Linear)
              cols = DGP (TreeFriendly, PLR, WGAN)
    - Lines: estimator (DoubleML, EconML, OLS)
    - Single global legend (no overlap).
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ['DGP', 'Method', 'Theta', 'Coverage']
    if not all(col in summary.columns for col in required_cols):
        raise ValueError("summary must contain DGP, Method, Theta, and Coverage columns.")

    print("Generating Plot: Coverage Comparison (TreeFriendly, PLR, WGAN | No Naive)...")

    plot_df = summary.copy()

    # ---- scenario / estimator parsers --------------------------------------
    def parse_scenario(method: str) -> str:
        if method.startswith('BadControl_'):
            return 'Multiplicative'
        if method.startswith('LinearCollider_'):
            return 'Linear'
        return 'Other'

    def parse_estimator(method: str) -> str:
        if method.endswith('_DoubleML') or method.endswith('_DML'):
            return 'DoubleML'
        if method.endswith('_EconML'):
            return 'EconML'
        if method.endswith('_OLS'):
            return 'OLS'
        if 'DoubleML' in method or 'DML' in method:
            return 'DoubleML'
        if 'EconML' in method:
            return 'EconML'
        if 'OLS' in method:
            return 'OLS'
        return 'Unknown'

    plot_df['scenario'] = plot_df['Method'].apply(parse_scenario)
    plot_df['estimator'] = plot_df['Method'].apply(parse_estimator)

    # ---- filters ------------------------------------------------------------
    plot_df = plot_df[
        plot_df['scenario'].isin(['Multiplicative', 'Linear']) &
        plot_df['estimator'].isin(['DoubleML', 'EconML', 'OLS']) &
        plot_df['DGP'].isin(['TreeFriendly', 'PLR', 'WGAN'])
    ].copy()

    plot_df = plot_df.sort_values(['scenario', 'DGP', 'Theta', 'estimator'])

    # ---- layout -------------------------------------------------------------
    dgps = ['TreeFriendly', 'PLR', 'WGAN']
    scenarios = ['Multiplicative', 'Linear']

    fig, axes = plt.subplots(
        nrows=len(scenarios),
        ncols=len(dgps),
        figsize=(14, 6),
        sharex=True,
        sharey=True
    )

    fig.suptitle(
        "Coverage Rates Comparison: TreeFriendly, PLR & WGAN (No Naive)",
        fontsize=14
    )

    legend_handles, legend_labels = None, None

    for i, scen in enumerate(scenarios):
        for j, dgp in enumerate(dgps):
            ax = axes[i, j]

            sub = plot_df[
                (plot_df['scenario'] == scen) &
                (plot_df['DGP'] == dgp)
            ]

            if sub.empty:
                ax.set_axis_off()
                continue

            sns.lineplot(
                data=sub,
                x='Theta',
                y='Coverage',
                hue='estimator',
                marker='o',
                ax=ax,
                errorbar=None
            )

            # capture legend once
            if legend_handles is None:
                handles, labels = ax.get_legend_handles_labels()
                cleaned = [(h, l) for h, l in zip(handles, labels) if l != 'estimator']
                seen = set()
                uniq = []
                for h, l in cleaned:
                    if l not in seen:
                        uniq.append((h, l))
                        seen.add(l)
                legend_handles = [h for h, _ in uniq]
                legend_labels = [l for _, l in uniq]

            # target lines
            ax.axhline(0.95, color='red', linestyle='--', linewidth=1)
            ax.axhline(0.90, color='gray', linestyle=':', linewidth=1)

            ax.set_title(f"dgp = {dgp}", fontsize=10)
            ax.set_xlabel("theta" if i == len(scenarios) - 1 else "")
            ax.set_ylabel("coverage" if j == 0 else "")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.3)

            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # row label (scenario)
        axes[i, 0].annotate(
            f"scenario = {scen}",
            xy=(-0.25, 0.5),
            xycoords='axes fraction',
            rotation=90,
            va='center',
            ha='center',
            fontsize=11
        )

    # ---- global legend ------------------------------------------------------
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="estimator",
            loc="center left",
            bbox_to_anchor=(0.90, 0.5),
            frameon=True
        )

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    fname = "coverage_comparison_2x3_tree_plr_wgan.png"
    plt.savefig(output_dir / fname, dpi=200)
    plt.close()

    print(f"Saved: {fname}")

def plot_bias_variance_grid(df: pd.DataFrame, output_dir: Path):
    """
    Bias-Variance decomposition (stacked), split by scenario (rows) and DGP (cols).
    2x3 layout:
      rows = scenario (Multiplicative, Linear)
      cols = DGP (TreeFriendly, PLR, WGAN)

    - Excludes Naive (no theta dependence).
    - Separate figure per estimator (DoubleML, EconML, OLS).
    - Uses tau_hat (raw draws) to compute Bias^2 and Variance by theta.
    - Reference (OPTION 2):
        * PLR / TreeFriendly -> tau_ref = 1.0
        * WGAN -> tau_ref = 6250.951
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ["DGP", "Method", "Theta", "tau_hat"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError("DataFrame must contain DGP, Method, Theta, and tau_hat columns.")

    plot_df = df.copy()

    # --- parsers -------------------------------------------------------------
    def parse_scenario(method: str) -> str:
        if method.startswith("BadControl_"):
            return "Multiplicative"
        if method.startswith("LinearCollider_"):
            return "Linear"
        if method.startswith("Naive_"):
            return "Naive"
        return "Other"

    def parse_estimator(method: str) -> str:
        m = str(method)
        if m.endswith("_DML") or m.endswith("_DoubleML") or "DML" in m or "DoubleML" in m:
            return "DoubleML"
        if m.endswith("_EconML") or "EconML" in m:
            return "EconML"
        if m.endswith("_OLS") or "OLS" in m:
            return "OLS"
        return "Unknown"

    plot_df["scenario"] = plot_df["Method"].astype(str).apply(parse_scenario)
    plot_df["estimator"] = plot_df["Method"].astype(str).apply(parse_estimator)

    # keep only what we need
    plot_df = plot_df[plot_df["DGP"].isin(["TreeFriendly", "PLR", "WGAN"])].copy()
    plot_df = plot_df[plot_df["scenario"].isin(["Multiplicative", "Linear"])].copy()
    plot_df = plot_df[plot_df["estimator"].isin(["DoubleML", "EconML", "OLS"])].copy()

    # ordering for the 2x3 grid
    dgps = ["TreeFriendly", "PLR", "WGAN"]
    scenarios = ["Multiplicative", "Linear"]
    estimators = ["DoubleML", "EconML", "OLS"]

    # references (OPTION 2)
    TRUE_EFFECT_DEFAULT = 1.0
    TRUE_EFFECT_WGAN = 6250.951

    for est in estimators:
        df_est = plot_df[plot_df["estimator"] == est].copy()
        if df_est.empty:
            print(f"Skipping estimator={est}: no data.")
            continue

        fig, axes = plt.subplots(
            nrows=len(scenarios),
            ncols=len(dgps),
            figsize=(14, 6),
            sharex=True
        )

        fig.suptitle(f"Bias-Variance Decomposition (Stacked): {est}", fontsize=16)

        legend_handles, legend_labels = None, None

        for i, scen in enumerate(scenarios):
            for j, dgp in enumerate(dgps):
                ax = axes[i, j]

                sub = df_est[(df_est["scenario"] == scen) & (df_est["DGP"] == dgp)].copy()
                if sub.empty:
                    ax.set_axis_off()
                    continue

                # choose reference
                tau_ref = TRUE_EFFECT_WGAN if dgp == "WGAN" else TRUE_EFFECT_DEFAULT

                rows = []
                for theta, g in sub.groupby("Theta"):
                    vals = g["tau_hat"].dropna()
                    if vals.empty:
                        continue

                    mu = float(vals.mean())
                    bias_sq = (mu - tau_ref) ** 2
                    var = float(vals.var(ddof=1)) if len(vals) > 1 else 0.0
                    rows.append({"Theta": float(theta), "Bias^2": bias_sq, "Variance": var})

                bv = pd.DataFrame(rows).sort_values("Theta")
                if bv.empty:
                    ax.set_axis_off()
                    continue

                # stacked plot
                polys = ax.stackplot(
                    bv["Theta"],
                    bv["Bias^2"],
                    bv["Variance"],
                    labels=["Bias^2", "Variance"],
                    alpha=0.8
                )

                # capture legend once (from first non-empty axis)
                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()

                # titles/labels
                ax.set_title(f"dgp = {dgp} | scenario = {scen}", fontsize=11)
                ax.grid(True, alpha=0.3)

                if i == len(scenarios) - 1:
                    ax.set_xlabel("theta")
                else:
                    ax.set_xlabel("")

                if j == 0:
                    ax.set_ylabel("mse")
                else:
                    ax.set_ylabel("")

                # remove per-axis legend
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

        # global legend on the right (no overlap)
        if legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(0.88, 0.5),
                frameon=True,
                title=""
            )

        plt.tight_layout(rect=[0, 0, 0.86, 0.93])
        fname = f"bias_variance_grid_2x3_{est.lower()}.png"
        plt.savefig(output_dir / fname, dpi=200)
        plt.close()

        print(f"Saved: {fname}")


def plot_tau_distribution_1x3_by_dgp(df: pd.DataFrame, output_dir: Path):
    """
    3 figuras (una por DGP: TreeFriendly, PLR, WGAN).
    Cada figura es 1x3 (DoubleML, EconML, OLS).
    En cada panel: distribuciones por escenario (Naive, Multiplicative, Linear).

    - Para colliders: SOLO Theta = 1.0 (se eliminan theta=0).
    - Línea vertical:
        * PLR / TreeFriendly -> tau = 1.0
        * WGAN -> referencia = 6250.951
    - Leyenda única fuera (a la derecha), sin invadir paneles.
    - Robustez: si KDE es inestable (pocos valores únicos / varianza ~0), usa histplot (density).
    - Fix especial: WGAN + OLS usa doble eje Y (Naive en eje izq, Colliders en eje der)
      para evitar que se aplaste por escalas muy distintas.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from typing import Optional, Tuple

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ["DGP", "Method", "Theta", "tau_hat"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError("DataFrame must contain DGP, Method, Theta, and tau_hat columns.")

    df = df.copy()

    estimators = ["DoubleML", "EconML", "OLS"]
    dgps = ["TreeFriendly", "PLR", "WGAN"]
    scenario_order = ["Naive", "Multiplicative", "Linear"]

    palette = {
        "Naive": "#1f77b4",
        "Multiplicative": "#ff7f0e",
        "Linear": "#2ca02c",
    }

    # ---- parsers ------------------------------------------------------------
    def parse_scenario(method: str) -> Optional[str]:
        if method.startswith("Naive_"):
            return "Naive"
        if method.startswith("BadControl_"):
            return "Multiplicative"
        if method.startswith("LinearCollider_"):
            return "Linear"
        return None

    def has_estimator(method: str, est: str) -> bool:
        # compatible con Method tipo: BadControl_DoubleML, Naive_EconML, LinearCollider_OLS, etc.
        return method.endswith(f"_{est}") or f"_{est}" in method

    # ---- helpers ------------------------------------------------------------
    def stable_xlim(values: np.ndarray, ref: float) -> Optional[Tuple[float, float]]:
        """Rango por panel usando cuantiles para evitar que se aplaste la KDE."""
        v = np.asarray(values)
        v = v[np.isfinite(v)]
        if v.size < 5:
            return None

        q01, q99 = np.quantile(v, [0.01, 0.99])
        if q01 == q99:
            q01, q99 = v.min(), v.max()

        span = max(1e-9, q99 - q01)
        pad = 0.10 * span

        lo = min(q01 - pad, ref - pad)
        hi = max(q99 + pad, ref + pad)

        if lo == hi:
            lo -= 1.0
            hi += 1.0

        return float(lo), float(hi)

    def plot_density(ax, x, label, color):
        """KDE si es estable; si no, hist density."""
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        if x.size < 5:
            return None

        # Heurística de estabilidad
        if np.std(x) < 1e-9 or np.unique(x).size < 5:
            artist = sns.histplot(
                x=x,
                bins=min(30, max(5, int(np.sqrt(x.size)))),
                stat="density",
                element="step",
                fill=True,
                alpha=0.20,
                color=color,
                ax=ax,
                label=label,
            )
            return artist

        artist = sns.kdeplot(
            x=x,
            fill=True,
            alpha=0.25,
            linewidth=1.5,
            color=color,
            ax=ax,
            label=label,
            bw_adjust=1.1,
        )
        return artist

    # ---- scenario column + filter thetas -----------------------------------
    df["scenario"] = df["Method"].apply(parse_scenario)
    df = df[df["scenario"].notnull()].copy()

    # eliminar theta=0 para colliders; mantener Naive (normalmente theta=0)
    df = df[(df["scenario"] == "Naive") | (df["Theta"] == 1.0)].copy()

    # ---- main loop ----------------------------------------------------------
    for dgp in dgps:
        sub_dgp = df[df["DGP"] == dgp].copy()
        if sub_dgp.empty:
            print(f"Skipping {dgp}: no data.")
            continue

        ref = 6250.951 if dgp == "WGAN" else 1.0
        ref_label = "WGAN reference" if dgp == "WGAN" else "True τ = 1.0"

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), sharey=False)
        fig.suptitle(f"{dgp}: Distribution of Estimates (Theta=1 for colliders)", fontsize=18)

        # Vamos a construir una leyenda global manualmente
        legend_handles = []
        legend_labels = []

        # Guardamos una vez los handles de escenarios (patches/lines)
        scenario_handle_map = {}
        ref_handle = None

        for j, est in enumerate(estimators):
            ax = axes[j]

            sub_est = sub_dgp[sub_dgp["Method"].apply(lambda m: has_estimator(m, est))].copy()
            if sub_est.empty:
                ax.set_axis_off()
                continue

            # datos por escenario
            data_by_scen = {}
            pooled = []

            for scen in scenario_order:
                vals = sub_est[sub_est["scenario"] == scen]["tau_hat"].dropna().values
                if vals.size > 0:
                    data_by_scen[scen] = vals
                    pooled.append(vals)

            pooled_vals = np.concatenate(pooled) if pooled else np.array([])

            # xlim robusto por panel
            xlim = stable_xlim(pooled_vals, ref=ref)
            if xlim is not None:
                ax.set_xlim(*xlim)

            ax.set_title(est, fontsize=14)
            ax.set_xlabel("tau_hat")
            if j == 0:
                ax.set_ylabel("Density")
            else:
                ax.set_ylabel("")
            ax.grid(True, alpha=0.30)

            # ---- Caso especial: WGAN + OLS -> dual y-axis --------------------
            if dgp == "WGAN" and est == "OLS":
                ax2 = ax.twinx()  # eje derecho para colliders

                # Naive en eje izquierdo
                if "Naive" in data_by_scen:
                    plot_density(ax, data_by_scen["Naive"], "Naive", palette["Naive"])

                # Colliders en eje derecho
                for scen in ["Multiplicative", "Linear"]:
                    if scen in data_by_scen:
                        plot_density(ax2, data_by_scen[scen], scen, palette[scen])

                # Ajustar xlim también para el twin (mismo)
                if xlim is not None:
                    ax2.set_xlim(*xlim)

                # Línea de referencia (la dibujamos en el eje izquierdo para que se vea consistente)
                ref_line = ax.axvline(ref, color="black", linestyle="--", linewidth=1.8)

                # Guardar handles para leyenda global (solo una vez)
                if not scenario_handle_map:
                    # Tomamos handles de ambos ejes
                    h1, l1 = ax.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    for h, l in zip(h1 + h2, l1 + l2):
                        if l not in scenario_handle_map:
                            scenario_handle_map[l] = h

                if ref_handle is None:
                    ref_handle = ref_line

                # Quitar leyendas internas
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                if ax2.get_legend() is not None:
                    ax2.get_legend().remove()

                # Etiquetas de ejes Y para que sea claro (opcional)
                ax.set_ylabel("Density (Naive)")
                ax2.set_ylabel("Density (Colliders)")

            # ---- Caso normal (un solo eje) -----------------------------------
            else:
                for scen in scenario_order:
                    if scen in data_by_scen:
                        plot_density(ax, data_by_scen[scen], scen, palette[scen])

                ref_line = ax.axvline(ref, color="black", linestyle="--", linewidth=1.8)

                # Guardar handles para leyenda global (solo una vez)
                if not scenario_handle_map:
                    h, l = ax.get_legend_handles_labels()
                    for hh, ll in zip(h, l):
                        if ll not in scenario_handle_map:
                            scenario_handle_map[ll] = hh

                if ref_handle is None:
                    ref_handle = ref_line

                if ax.get_legend() is not None:
                    ax.get_legend().remove()

        # ---- construir leyenda global (Scenario + ref) -----------------------
        # Orden fijo
        for scen in scenario_order:
            if scen in scenario_handle_map:
                legend_handles.append(scenario_handle_map[scen])
                legend_labels.append(scen)

        if ref_handle is not None:
            legend_handles.append(ref_handle)
            legend_labels.append(ref_label)

        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                title="Scenario",
                loc="center left",
                bbox_to_anchor=(0.92, 0.5),  # ← más cerca
                frameon=True,
            )

        fig.subplots_adjust(
            left=0.06,
            right=0.88,   # ← más espacio para los paneles
            bottom=0.14,
            top=0.82,
            wspace=0.25,
        )

        fname = f"tau_distribution_1x3_{dgp.lower()}.png"
        plt.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
        plt.close()

