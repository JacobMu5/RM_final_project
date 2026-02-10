import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from typing import Optional, Tuple


ESTIMATOR_PATTERNS = {
    "DoubleML": ["DoubleML", "DML"],
    "EconML": ["EconML"],
    "OLS": ["OLS"]
}

SCENARIO_PREFIXES = {
    "Naive": "Naive_",
    "Multiplicative": "BadControl_",
    "Linear": "LinearCollider_"
}

TRUE_EFFECT_DEFAULT = 1.0
TRUE_EFFECT_WGAN = 6250.951


def _ensure_output_dir(output_dir: Path) -> Path:
    """Ensure output directory exists and return it as Path object.

    Args:
        output_dir: Directory path to create if it doesn't exist.

    Returns:
        Path object of the created/existing directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _parse_scenario(method: str) -> str:
    """Extract scenario from method name based on prefix patterns.

    Args:
        method: Method name string to parse.

    Returns:
        Scenario name or "Other" if no match found.
    """
    method = str(method)
    for scenario, prefix in SCENARIO_PREFIXES.items():
        if method.startswith(prefix):
            return scenario
    return "Other"


def _parse_estimator(method: str) -> str:
    """Extract estimator type from method name.

    Args:
        method: Method name string to parse.

    Returns:
        Estimator name or "Unknown" if no match found.
    """
    method = str(method)
    for estimator, patterns in ESTIMATOR_PATTERNS.items():
        if any(pattern in method for pattern in patterns):
            return estimator
    return "Unknown"


def _has_estimator(method: str, est: str) -> bool:
    """Check if method name contains the specified estimator.

    Args:
        method: Method name string to check.
        est: Estimator name to search for.

    Returns:
        True if estimator is found in method name.
    """
    method = str(method)
    return method.endswith(f"_{est}") or f"_{est}" in method


def _lineplot_no_ci(*, data, x, y, hue, marker, ax):
    """Create seaborn lineplot without confidence intervals for compatibility.

    Args:
        data: DataFrame containing the data to plot.
        x: Column name for x-axis variable.
        y: Column name for y-axis variable.
        hue: Column name for grouping variable.
        marker: Marker style for line points.
        ax: Matplotlib axes object to plot on.

    Returns:
        None. Modifies the axes object in place.
    """
    try:
        sns.lineplot(data=data, x=x, y=y, hue=hue, marker=marker, ax=ax, errorbar=None)
    except TypeError:
        sns.lineplot(data=data, x=x, y=y, hue=hue, marker=marker, ax=ax, ci=None)


def _extract_unique_legend(ax) -> Tuple[list, list]:
    """Extract unique legend handles and labels from axes.

    Args:
        ax: Matplotlib axes object.

    Returns:
        Tuple of (handles, labels) with duplicates removed.
    """
    handles, labels = ax.get_legend_handles_labels()
    cleaned = [(h, l) for h, l in zip(handles, labels) if l != "estimator"]
    seen = set()
    unique = []
    for h, l in cleaned:
        if l not in seen:
            unique.append((h, l))
            seen.add(l)
    return [h for h, _ in unique], [l for _, l in unique]


def _find_bias_column(df: pd.DataFrame) -> Optional[str]:
    """Find the bias column in DataFrame, checking multiple possible names.

    Args:
        df: DataFrame to search for bias column.

    Returns:
        Column name if found, None otherwise.
    """
    for col_name in ["Bias_Mean", "bias_mean", "bias"]:
        if col_name in df.columns:
            return col_name
    return None


def plot_microscope_view(
    dgp,
    est,
    theta,
    output_dir: Path,
    filename_suffix: str = "",
    wgan_baseline: float = TRUE_EFFECT_WGAN,
    wgan_center: bool = True,
):
    """Generate microscope diagnostic plot showing CATE estimates vs multiplicative collider.

    Args:
        dgp: Data generating process object containing the collider variable C and name.
        est: Estimator object containing CATE estimates.
        theta: Strength parameter of the collider.
        output_dir: Directory path where the plot will be saved.
        filename_suffix: Optional suffix to append to the output filename. Defaults to "".
        wgan_baseline: Baseline reference value for WGAN scenarios. Defaults to 6250.951.
        wgan_center: If True, center WGAN plots at the baseline. Defaults to True.

    Returns:
        None. Saves the plot to the specified output directory as a PNG file.
    """
    output_dir = _ensure_output_dir(output_dir)

    dgp_name = getattr(dgp, "name", dgp.__class__.__name__)
    dgp_key = str(dgp_name).lower().strip()
    is_wgan = "wgan" in dgp_key

    print(f"Generating Microscope View: {dgp_name} (Theta={theta}) | is_wgan={is_wgan}")

    if getattr(est, "cate_estimates", None) is None:
        print("Skipping Microscope View: Estimator has no CATE estimates.")
        return

    if getattr(dgp, "C", None) is None:
        print("Skipping Microscope View: DGP has no multiplicative collider.")
        return

    C = np.asarray(dgp.C).flatten()
    tau_hat = np.asarray(est.cate_estimates).flatten()

    mask = np.isfinite(C) & np.isfinite(tau_hat)
    C, tau_hat = C[mask], tau_hat[mask]

    if C.size < 5:
        print("Skipping Microscope View: Not enough finite observations.")
        return

    corr_val = (np.corrcoef(C, tau_hat)[0, 1] 
                if np.std(C) > 0 and np.std(tau_hat) > 0 
                else np.nan)

    if is_wgan and wgan_center:
        y = tau_hat - wgan_baseline
        ref, ref_label = 0.0, "Baseline (WGAN) = 0"
        y_label = "Estimated Treatment Effect (Centered)"
    else:
        y = tau_hat
        if is_wgan:
            ref, ref_label = wgan_baseline, "Baseline (WGAN)"
        else:
            ref, ref_label = TRUE_EFFECT_DEFAULT, "True Effect (1.0)"
        y_label = "Estimated Treatment Effect"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(C, y, alpha=0.45, s=20, color="red", label="Multiplicative Collider")
    ax.axhline(ref, color="green", linestyle="--", linewidth=2, label=ref_label)
    
    corr_txt = f"{corr_val:.2f}" if np.isfinite(corr_val) else "NA"
    ax.set_title(f"Microscope View: Corr(C, τ̂) = {corr_txt}\n(Theta={theta}) {filename_suffix}", 
                 fontsize=13)
    ax.set_xlabel("Collider Value")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    fname = f"microscope_{'wgan' if is_wgan else dgp_key}_theta_{theta}{filename_suffix}.png"
    plt.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {fname}")


def plot_cate_distribution(est, theta, output_dir: Path):
    """Plot histogram of CATE estimates with true effect reference line.

    Args:
        est: Estimator object containing CATE estimates.
        theta: Strength parameter of the collider.
        output_dir: Directory path where the plot will be saved.

    Returns:
        None. Saves the plot to the specified output directory as a PNG file.
    """
    output_dir = _ensure_output_dir(output_dir)
    print("Generating Plot: CATE Distribution...")

    if not hasattr(est, "cate_estimates") or est.cate_estimates is None:
        return

    cates = np.asarray(est.cate_estimates).flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(cates, kde=True, color="purple", bins=30, ax=ax)
    ax.axvline(TRUE_EFFECT_DEFAULT, color="black", linestyle="--", 
               linewidth=2, label="True Effect (1.0)")
    ax.set_title(f"Spurious Heterogeneity Distribution (Theta={theta})", fontsize=14)
    ax.set_xlabel("Estimated CATE")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / f"cate_distribution_theta_{theta}.png")
    plt.close()


def plot_bias_comparison(df: pd.DataFrame, output_dir: Path):
    """Create 2x3 grid comparing bias across scenarios and DGPs.

    Args:
        df: DataFrame containing columns: DGP, Method, Theta, and either bias_mean/Bias_Mean
            (for PLR/TreeFriendly) or tau_hat (for WGAN).
        output_dir: Directory path where the plot will be saved.

    Returns:
        None. Saves the plot to the specified output directory as 'bias_comparison.png'.
    """
    output_dir = _ensure_output_dir(output_dir)

    required_cols = ["DGP", "Method", "Theta"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame must contain DGP, Method, and Theta columns.")

    bias_col = _find_bias_column(df)
    has_tau_hat = "tau_hat" in df.columns

    dgps = ["TreeFriendly", "PLR", "WGAN"]
    scenarios = ["Multiplicative", "Linear"]

    plot_df = df[df["DGP"].isin(dgps)].copy()
    plot_df["scenario"] = plot_df["Method"].apply(_parse_scenario)
    plot_df["estimator"] = plot_df["Method"].apply(_parse_estimator)

    plot_df = plot_df[
        plot_df["scenario"].isin(scenarios) &
        plot_df["estimator"].isin(["DoubleML", "EconML", "OLS"])
    ].copy()

    print("Generating Plot: Bias Comparison (2x3 Grid: TreeFriendly, PLR, WGAN)...")

    metric_rows = []

    if bias_col is None:
        raise ValueError("For PLR/TreeFriendly I need Bias_Mean/bias_mean/bias, but none was found.")

    sim = plot_df[plot_df["DGP"].isin(["PLR", "TreeFriendly"])].copy()
    if not sim.empty:
        sim_metric = sim[["DGP", "scenario", "estimator", "Theta", bias_col]].rename(
            columns={bias_col: "metric"}
        )
        metric_rows.append(sim_metric)

    wgan = plot_df[plot_df["DGP"] == "WGAN"].copy()
    if not wgan.empty:
        if not has_tau_hat:
            print("Warning: WGAN data has no tau_hat -> cannot compute sensitivity. Skipping WGAN.")
        else:
            wgan_mean = wgan.groupby(
                ["scenario", "estimator", "Theta"], as_index=False
            )["tau_hat"].mean().rename(columns={"tau_hat": "tau_mean"})
            
            base = wgan_mean[wgan_mean["Theta"] == 0.0][
                ["scenario", "estimator", "tau_mean"]
            ].rename(columns={"tau_mean": "tau_base"})

            wgan_m = wgan_mean.merge(base, on=["scenario", "estimator"], how="left")
            wgan_m["metric"] = wgan_m["tau_mean"] - wgan_m["tau_base"]
            wgan_m["DGP"] = "WGAN"
            metric_rows.append(wgan_m[["DGP", "scenario", "estimator", "Theta", "metric"]])

    metric_df = pd.concat(metric_rows, ignore_index=True).sort_values(
        ["scenario", "DGP", "Theta", "estimator"]
    )

    fig, axes = plt.subplots(
        nrows=len(scenarios), ncols=len(dgps),
        figsize=(15, 6), sharex=True, sharey=False
    )

    legend_handles, legend_labels = None, None

    for i, scen in enumerate(scenarios):
        for j, dgp in enumerate(dgps):
            ax = axes[i, j]
            sub = metric_df[(metric_df["scenario"] == scen) & (metric_df["DGP"] == dgp)]
            
            if sub.empty:
                ax.set_axis_off()
                continue

            _lineplot_no_ci(data=sub, x="Theta", y="metric", hue="estimator", marker="o", ax=ax)

            if legend_handles is None:
                legend_handles, legend_labels = _extract_unique_legend(ax)

            if ax.get_legend():
                ax.get_legend().remove()

            ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"dgp = {dgp} | scenario = {scen}", fontsize=10)
            ax.set_xlabel("Theta" if i == len(scenarios) - 1 else "")
            ax.set_ylabel("Bias mean" if (j == 0 and dgp in ["TreeFriendly", "PLR"]) 
                         else "Δ τ vs θ=0" if j == 0 else "")
            ax.grid(True, alpha=0.3)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, title="estimator",
                  loc="center left", bbox_to_anchor=(0.90, 0.5), frameon=True)

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(output_dir / "bias_comparison.png", dpi=200)
    plt.close()

    print("Saved: bias_comparison.png")


def plot_coverage_comparison(summary: pd.DataFrame, output_dir: Path):
    """Create 2x3 grid comparing coverage across scenarios and DGPs.

    Args:
        summary: DataFrame containing columns: DGP, Method, Theta, and Coverage.
        output_dir: Directory path where the plot will be saved.

    Returns:
        None. Saves the plot to the specified output directory as 'coverage_comparison.png'.
    """
    output_dir = _ensure_output_dir(output_dir)

    required_cols = ["DGP", "Method", "Theta", "Coverage"]
    if not all(col in summary.columns for col in required_cols):
        raise ValueError("summary must contain DGP, Method, Theta, and Coverage columns.")

    print("Generating Plot: Coverage Comparison (TreeFriendly, PLR, WGAN | No Naive)...")

    plot_df = summary.copy()
    plot_df["scenario"] = plot_df["Method"].astype(str).apply(_parse_scenario)
    plot_df["estimator"] = plot_df["Method"].astype(str).apply(_parse_estimator)

    plot_df = plot_df[
        plot_df["scenario"].isin(["Multiplicative", "Linear"]) &
        plot_df["estimator"].isin(["DoubleML", "EconML", "OLS"]) &
        plot_df["DGP"].isin(["TreeFriendly", "PLR", "WGAN"])
    ].sort_values(["scenario", "DGP", "Theta", "estimator"])

    dgps = ["TreeFriendly", "PLR", "WGAN"]
    scenarios = ["Multiplicative", "Linear"]

    fig, axes = plt.subplots(
        nrows=len(scenarios), ncols=len(dgps),
        figsize=(14, 6), sharex=True, sharey=True
    )

    legend_handles, legend_labels = None, None

    for i, scen in enumerate(scenarios):
        for j, dgp in enumerate(dgps):
            ax = axes[i, j]
            sub = plot_df[(plot_df["scenario"] == scen) & (plot_df["DGP"] == dgp)]
            
            if sub.empty:
                ax.set_axis_off()
                continue

            _lineplot_no_ci(data=sub, x="Theta", y="Coverage", hue="estimator", marker="o", ax=ax)

            if legend_handles is None:
                legend_handles, legend_labels = _extract_unique_legend(ax)

            ax.axhline(0.95, color="red", linestyle="--", linewidth=1)
            ax.axhline(0.90, color="gray", linestyle=":", linewidth=1)
            ax.set_title(f"dgp = {dgp}", fontsize=10)
            ax.set_xlabel("Theta" if i == len(scenarios) - 1 else "")
            ax.set_ylabel("Coverage" if j == 0 else "")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.3)

            if ax.get_legend():
                ax.get_legend().remove()

        axes[i, 0].annotate(
            f"scenario = {scen}", xy=(-0.25, 0.5), xycoords="axes fraction",
            rotation=90, va="center", ha="center", fontsize=11
        )

    if legend_handles:
        fig.legend(legend_handles, legend_labels, title="estimator",
                  loc="center left", bbox_to_anchor=(0.90, 0.5), frameon=True)

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(output_dir / "coverage_comparison.png", dpi=200)
    plt.close()

    print("Saved: coverage_comparison.png")


def plot_bias_variance_grid(df: pd.DataFrame, output_dir: Path):
    """Create 2x3 grid showing bias-variance decomposition across scenarios and DGPs.

    Args:
        df: DataFrame containing columns: DGP, Method, Theta, and tau_hat.
        output_dir: Directory path where the plots will be saved.

    Returns:
        None. Saves plots to the specified output directory as 'bias_variance_{estimator}.png'.
    """
    output_dir = _ensure_output_dir(output_dir)

    required_cols = ["DGP", "Method", "Theta", "tau_hat"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError("DataFrame must contain DGP, Method, Theta, and tau_hat columns.")

    print("Generating Plot: Bias-Variance Grid (2x3)...")

    plot_df = df.copy()
    plot_df["scenario"] = plot_df["Method"].astype(str).apply(_parse_scenario)
    plot_df["estimator"] = plot_df["Method"].astype(str).apply(_parse_estimator)

    plot_df = plot_df[
        plot_df["DGP"].isin(["TreeFriendly", "PLR", "WGAN"]) &
        plot_df["scenario"].isin(["Multiplicative", "Linear"]) &
        plot_df["estimator"].isin(["DoubleML", "EconML", "OLS"])
    ].copy()

    dgps = ["TreeFriendly", "PLR", "WGAN"]
    scenarios = ["Multiplicative", "Linear"]
    estimators = ["DoubleML", "EconML", "OLS"]

    for est in estimators:
        df_est = plot_df[plot_df["estimator"] == est]
        if df_est.empty:
            print(f"Skipping Plot: No data for estimator={est}")
            continue

        fig, axes = plt.subplots(
            nrows=len(scenarios), ncols=len(dgps),
            figsize=(14, 6), sharex=True
        )

        legend_handles, legend_labels = None, None

        for i, scen in enumerate(scenarios):
            for j, dgp in enumerate(dgps):
                ax = axes[i, j]
                sub = df_est[(df_est["scenario"] == scen) & (df_est["DGP"] == dgp)]
                
                if sub.empty:
                    ax.set_axis_off()
                    continue

                tau_ref = TRUE_EFFECT_WGAN if dgp == "WGAN" else TRUE_EFFECT_DEFAULT

                bias_variance_data = []
                for theta, group in sub.groupby("Theta"):
                    vals = group["tau_hat"].dropna()
                    if vals.empty:
                        continue

                    mu = vals.mean()
                    bias_sq = (mu - tau_ref) ** 2
                    var = vals.var(ddof=1) if len(vals) > 1 else 0.0
                    bias_variance_data.append({
                        "Theta": float(theta), 
                        "Bias^2": bias_sq, 
                        "Variance": var
                    })

                bv = pd.DataFrame(bias_variance_data).sort_values("Theta")
                if bv.empty:
                    ax.set_axis_off()
                    continue

                ax.stackplot(bv["Theta"], bv["Bias^2"], bv["Variance"],
                            labels=["Bias^2", "Variance"], alpha=0.8)

                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()

                ax.set_title(f"dgp = {dgp} | scenario = {scen}", fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("Theta" if i == len(scenarios) - 1 else "")
                ax.set_ylabel("MSE" if j == 0 else "")

                if ax.get_legend():
                    ax.get_legend().remove()

        if legend_handles and legend_labels:
            fig.legend(legend_handles, legend_labels, loc="center left",
                      bbox_to_anchor=(0.88, 0.5), frameon=True, title="")

        plt.tight_layout(rect=[0, 0, 0.86, 0.93])
        fname = f"bias_variance_{est.lower()}.png"
        plt.savefig(output_dir / fname, dpi=200)
        plt.close()

        print(f"Saved: {fname}")


def plot_tau_distribution_1x3_by_dgp(df: pd.DataFrame, output_dir: Path):
    """Create estimate distribution plots organized by DGP.

    Args:
        df: DataFrame containing columns: DGP, Method, Theta, and tau_hat.
        output_dir: Directory path where the plots will be saved.

    Returns:
        None. Saves plots to the specified output directory as
        'Distribution_of_Estimates_{dgp}.png'.
    """
    output_dir = _ensure_output_dir(output_dir)

    required_cols = ["DGP", "Method", "Theta", "tau_hat"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError("DataFrame must contain DGP, Method, Theta, tau_hat.")

    print("Generating Plot: Tau Distribution (1x3 by DGP)...")

    estimators = ["OLS", "DoubleML", "EconML"]
    dgps = ["TreeFriendly", "PLR", "WGAN"]
    scenario_order = ["Naive", "Multiplicative", "Linear"]

    palette = {
        "Naive": "#1F4E79",
        "Multiplicative": "#ff7f0e",
        "Linear": "#4C956C",
    }

    x_limits = {
        "WGAN": (-7000, 12000),
        "PLR": (0.4, 1.2),
        "TreeFriendly": (0.4, 1.2),
    }

    def plot_density(ax, x, color):
        """Helper function to plot density."""
        x = x[np.isfinite(x)]
        if x.size < 5:
            return

        if np.std(x) < 1e-9 or np.unique(x).size < 5:
            sns.histplot(
                x=x, bins=min(30, max(5, int(np.sqrt(x.size)))),
                stat="density", element="step", fill=True,
                alpha=0.20, color=color, ax=ax
            )
        else:
            sns.kdeplot(x=x, fill=True, alpha=0.25, linewidth=1.5,
                       color=color, ax=ax, bw_adjust=1.1)

    df = df.copy()
    df["scenario"] = df["Method"].astype(str).apply(_parse_scenario)
    df = df[df["scenario"].notnull() & 
            ((df["scenario"] == "Naive") | (df["Theta"] == 1.0))].copy()

    for dgp in dgps:
        sub_dgp = df[df["DGP"] == dgp]
        if sub_dgp.empty:
            continue

        ref = TRUE_EFFECT_WGAN if dgp == "WGAN" else TRUE_EFFECT_DEFAULT
        ref_label = "WGAN Reference" if dgp == "WGAN" else "True Effect (1.0)"
        xlim_global = x_limits.get(dgp)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        fname = f"Distribution_of_Estimates_{dgp.lower()}.png"
        main_axes = []

        for j, est in enumerate(estimators):
            ax = axes[j]
            sub_est = sub_dgp[sub_dgp["Method"].apply(lambda m: _has_estimator(m, est))]
            
            if sub_est.empty:
                ax.set_axis_off()
                continue

            if dgp == "WGAN" and est == "OLS":
                ax_right = ax.twinx()
                x_naive = sub_est[sub_est["scenario"] == "Naive"]["tau_hat"].values
                plot_density(ax, x_naive, palette["Naive"])

                for scen in ["Multiplicative", "Linear"]:
                    x_s = sub_est[sub_est["scenario"] == scen]["tau_hat"].values
                    plot_density(ax_right, x_s, palette[scen])

                if xlim_global:
                    ax.set_xlim(*xlim_global)
                    ax_right.set_xlim(*xlim_global)

                ax.axvline(ref, color="black", linestyle="--", linewidth=1.8)
                ax.set_ylabel("Density (Naive)")
                ax_right.set_ylabel("Density (Colliders)")
                ax.set_title("OLS")
                ax.grid(True, alpha=0.3)
                main_axes.append(ax)
                continue

            for scen in scenario_order:
                x_s = sub_est[sub_est["scenario"] == scen]["tau_hat"].values
                plot_density(ax, x_s, palette[scen])

            if xlim_global:
                ax.set_xlim(*xlim_global)

            ax.axvline(ref, color="black", linestyle="--", linewidth=1.8)
            ax.set_title(est)
            ax.set_xlabel("Tau Hat")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Density" if j == 0 else "")

            if dgp in ["PLR", "TreeFriendly"]:
                ax.set_ylim(0, 18)

            main_axes.append(ax)

        if dgp == "WGAN" and main_axes:
            y_max = max(ax.get_ylim()[1] for ax in main_axes)
            for ax in main_axes:
                ax.set_ylim(0, y_max)

        legend_elements = [
            Line2D([0], [0], color=palette["Naive"], lw=2, label="Naive"),
            Line2D([0], [0], color=palette["Multiplicative"], lw=2, label="Multiplicative"),
            Line2D([0], [0], color=palette["Linear"], lw=2, label="Linear"),
            Line2D([0], [0], color="black", lw=2, linestyle="--", label=ref_label),
        ]

        fig.legend(handles=legend_elements, title="Scenario", loc="center left",
                  bbox_to_anchor=(0.85, 0.5), frameon=True)

        plt.tight_layout(rect=[0, 0, 0.86, 1])
        plt.savefig(output_dir / fname, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")