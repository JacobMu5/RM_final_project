import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

def plot_microscope_view(dgp, est, theta, output_dir: Path, filename_suffix: str = ""):
    """
    Microscope diagnostic plot.

    Scatter plot of estimated individual treatment effects (CATEs)
    against the multiplicative collider.

    - Linear collider is intentionally omitted for clarity.
    - Shows correlation Corr(C, τ̂).
    - Includes reference line:
        * PLR / TreeFriendly -> True effect (1.0)
        * WGAN -> Baseline reference
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dgp_name = getattr(dgp, "name", dgp.__class__.__name__)
    print(f"Generating Microscope View: {dgp_name} (Theta={theta})")

    if not hasattr(est, "cate_estimates") or est.cate_estimates is None:
        print("Skipping Microscope View: Estimator has no CATE estimates.")
        return

    if not hasattr(dgp, "C") or dgp.C is None:
        print("Skipping Microscope View: DGP has no multiplicative collider.")
        return

    C = dgp.C.flatten()
    tau_hat = est.cate_estimates.flatten()

    corr_val = np.corrcoef(C, tau_hat)[0, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=C,
        y=tau_hat,
        alpha=0.45,
        color="red",
        label="Multiplicative Collider",
    )

    # Reference line
    if dgp_name.lower() == "wgan":
        ref = 6250.951
        ref_label = "Baseline (WGAN)"
    else:
        ref = 1.0
        ref_label = "True Effect"

    plt.axhline(ref, color="green", linestyle="--", linewidth=2, label=ref_label)

    plt.title(
        f"Microscope View: Corr(C, τ̂) = {corr_val:.2f}\n(Theta={theta}) {filename_suffix}",
        fontsize=13,
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

    if not hasattr(est, "cate_estimates") or est.cate_estimates is None:
        return

    cates = est.cate_estimates.flatten()

    plt.figure(figsize=(10, 6))
    sns.histplot(cates, kde=True, color="purple", bins=30)
    plt.axvline(1.0, color="black", linestyle="--", linewidth=2, label="True Effect (1.0)")
    plt.title(f"Spurious Heterogeneity Distribution (Theta={theta})", fontsize=14)
    plt.xlabel("Estimated CATE")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"cate_distribution_theta_{theta}.png")
    plt.close()


def plot_rmse_comparison(df: pd.DataFrame, output_dir: Path):
    """Generates RMSE comparison plots for PLR and TreeFriendly (WGAN excluded: no ground truth)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ["DGP", "Method", "Theta", "RMSE"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame must contain DGP, Method, Theta, and RMSE columns.")

    print("Generating Plot: RMSE Comparison (PLR & TreeFriendly)...")

    plot_df = df.copy()
    plot_df = plot_df[plot_df["DGP"].isin(["PLR", "TreeFriendly"])].copy()

    def parse_scenario_type(method: str) -> str:
        if method.startswith("Naive_"):
            return "Naive"
        if method.startswith("BadControl_"):
            return "MultiplicativeCollider"
        if method.startswith("LinearCollider_"):
            return "LinearCollider"
        return "Other"

    def parse_estimator(method: str) -> str:
        if "DoubleML" in method or "DML" in method:
            return "DoubleML"
        if "EconML" in method:
            return "EconML"
        if "OLS" in method:
            return "OLS"
        return "Unknown"

    plot_df["scenario_type"] = plot_df["Method"].apply(parse_scenario_type)
    plot_df["estimator"] = plot_df["Method"].apply(parse_estimator)

    plot_df = plot_df[
        plot_df["scenario_type"].isin(["Naive", "MultiplicativeCollider", "LinearCollider"])
        & plot_df["estimator"].isin(["DoubleML", "EconML", "OLS"])
    ].copy()

    plot_df = plot_df.sort_values(["DGP", "scenario_type", "Theta", "estimator"])

    dgps = ["PLR", "TreeFriendly"]
    scenario_order = ["MultiplicativeCollider", "LinearCollider", "Naive"]
    scenario_title = {
        "MultiplicativeCollider": "Multiplicative",
        "LinearCollider": "Linear",
        "Naive": "Naive",
    }

    fig, axes = plt.subplots(
        nrows=len(dgps),
        ncols=len(scenario_order),
        figsize=(12, 5),
        sharex=True,
    )

    fig.suptitle("RMSE Comparison: PLR & TreeFriendly Scenarios", fontsize=14)

    for i, dgp in enumerate(dgps):
        for j, scen in enumerate(scenario_order):
            ax = axes[i, j]
            sub = plot_df[(plot_df["DGP"] == dgp) & (plot_df["scenario_type"] == scen)]

            if sub.empty:
                ax.set_axis_off()
                continue

            sns.lineplot(
                data=sub,
                x="Theta",
                y="RMSE",
                hue="estimator",
                marker="o",
                ax=ax,
                errorbar=None,
                legend=False,
            )

            ax.set_title(f"dgp = {dgp} | scenario = {scenario_title[scen]}", fontsize=10)
            ax.set_xlabel("theta" if i == len(dgps) - 1 else "")
            ax.set_ylabel("rmse" if j == 0 else "")
            ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=sns.color_palette()[0], marker="o", label="DoubleML"),
        Line2D([0], [0], color=sns.color_palette()[1], marker="o", label="EconML"),
        Line2D([0], [0], color=sns.color_palette()[2], marker="o", label="OLS"),
    ]

    fig.legend(
        handles=legend_elements,
        title="estimator",
        loc="center right",
        bbox_to_anchor=(0.93, 0.5),
        frameon=True,
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

    - Excludes Naive.
    - For PLR/TreeFriendly: plots bias_mean (or bias).
    - For WGAN: plots sensitivity = mean(tau_hat(theta)) - mean(tau_hat(theta=0)).
    - Single global legend outside (no overlap).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ["DGP", "Method", "Theta"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame must contain DGP, Method, and Theta columns.")

    # Choose bias column if available
    bias_col = None
    if "Bias_Mean" in df.columns:
        bias_col = "Bias_Mean"
    elif "bias_mean" in df.columns:
        bias_col = "bias_mean"
    elif "bias" in df.columns:
        bias_col = "bias"

    has_tau_hat = "tau_hat" in df.columns

    dgps = ["TreeFriendly", "PLR", "WGAN"]
    scenarios = ["Multiplicative", "Linear"]

    plot_df = df.copy()
    plot_df = plot_df[plot_df["DGP"].isin(dgps)].copy()

    def parse_scenario(method: str) -> str:
        if method.startswith("BadControl_"):
            return "Multiplicative"
        if method.startswith("LinearCollider_"):
            return "Linear"
        return "Other"

    def parse_estimator(method: str) -> str:
        if "DoubleML" in method or "DML" in method:
            return "DoubleML"
        if "EconML" in method:
            return "EconML"
        if "OLS" in method:
            return "OLS"
        return "Unknown"

    plot_df["scenario"] = plot_df["Method"].apply(parse_scenario)
    plot_df["estimator"] = plot_df["Method"].apply(parse_estimator)

    plot_df = plot_df[
        plot_df["scenario"].isin(scenarios)
        & plot_df["estimator"].isin(["DoubleML", "EconML", "OLS"])
    ].copy()

    print("Generating Plot: Bias Comparison (2x3 Grid: TreeFriendly, PLR, WGAN)...")

    metric_rows = []

    # (A) PLR / TreeFriendly -> bias metric
    if bias_col is None:
        raise ValueError("For PLR/TreeFriendly I need Bias_Mean/bias_mean/bias, but none was found.")

    sim = plot_df[plot_df["DGP"].isin(["PLR", "TreeFriendly"])].copy()
    if not sim.empty:
        sim_metric = sim[["DGP", "scenario", "estimator", "Theta", bias_col]].copy()
        sim_metric = sim_metric.rename(columns={bias_col: "metric"})
        metric_rows.append(sim_metric)

    # (B) WGAN -> sensitivity vs theta=0
    wgan = plot_df[plot_df["DGP"] == "WGAN"].copy()
    if not wgan.empty:
        if not has_tau_hat:
            print("Warning: WGAN data has no tau_hat -> cannot compute sensitivity vs theta=0. Skipping WGAN.")
        else:
            wgan_mean = (
                wgan.groupby(["scenario", "estimator", "Theta"], as_index=False)["tau_hat"]
                .mean()
                .rename(columns={"tau_hat": "tau_mean"})
            )
            base = (
                wgan_mean[wgan_mean["Theta"] == 0.0][["scenario", "estimator", "tau_mean"]]
                .rename(columns={"tau_mean": "tau_base"})
            )

            wgan_m = wgan_mean.merge(base, on=["scenario", "estimator"], how="left")
            wgan_m["metric"] = wgan_m["tau_mean"] - wgan_m["tau_base"]
            wgan_m["DGP"] = "WGAN"
            metric_rows.append(wgan_m[["DGP", "scenario", "estimator", "Theta", "metric"]])

    metric_df = pd.concat(metric_rows, ignore_index=True)
    metric_df = metric_df.sort_values(["scenario", "DGP", "Theta", "estimator"])

    fig, axes = plt.subplots(
        nrows=len(scenarios),
        ncols=len(dgps),
        figsize=(15, 6),
        sharex=True,
        sharey=False,
    )
    fig.suptitle("Bias Comparison: PLR, TreeFriendly & WGAN", fontsize=14)

    legend_handles, legend_labels = None, None

    for i, scen in enumerate(scenarios):
        for j, dgp in enumerate(dgps):
            ax = axes[i, j]

            sub = metric_df[(metric_df["scenario"] == scen) & (metric_df["DGP"] == dgp)].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            sns.lineplot(
                data=sub,
                x="Theta",
                y="metric",
                hue="estimator",
                marker="o",
                ax=ax,
                errorbar=None,
            )

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            if ax.get_legend() is not None:
                ax.get_legend().remove()

            ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"dgp = {dgp} | scenario = {scen}", fontsize=10)
            ax.set_xlabel("theta" if i == len(scenarios) - 1 else "")
            ax.set_ylabel(
                "bias_mean" if (j == 0 and dgp != "WGAN") else ("Δ tau vs θ=0" if (j == 0 and dgp == "WGAN") else "")
            )
            ax.grid(True, alpha=0.3)

    fig.legend(
        legend_handles,
        legend_labels,
        title="estimator",
        loc="center left",
        bbox_to_anchor=(0.90, 0.5),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(output_dir / "bias_comparison.png", dpi=200)
    plt.close()

    print("Saved: bias_comparison.png")


def plot_coverage_comparison(summary: pd.DataFrame, output_dir: Path):
    """
    Coverage comparison plot, split by scenario and DGP.

    - Includes TreeFriendly, PLR, WGAN.
    - Excludes Naive.
    - Panels: rows = scenario (Multiplicative, Linear); cols = DGP (TreeFriendly, PLR, WGAN).
    - Lines: estimator (DoubleML, EconML, OLS).
    - Single global legend outside (no overlap).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ["DGP", "Method", "Theta", "Coverage"]
    if not all(col in summary.columns for col in required_cols):
        raise ValueError("summary must contain DGP, Method, Theta, and Coverage columns.")

    print("Generating Plot: Coverage Comparison (TreeFriendly, PLR, WGAN | No Naive)...")

    plot_df = summary.copy()

    def parse_scenario(method: str) -> str:
        if method.startswith("BadControl_"):
            return "Multiplicative"
        if method.startswith("LinearCollider_"):
            return "Linear"
        return "Other"

    def parse_estimator(method: str) -> str:
        m = str(method)
        if m.endswith("_DoubleML") or m.endswith("_DML") or "DoubleML" in m or "DML" in m:
            return "DoubleML"
        if m.endswith("_EconML") or "EconML" in m:
            return "EconML"
        if m.endswith("_OLS") or "OLS" in m:
            return "OLS"
        return "Unknown"

    plot_df["scenario"] = plot_df["Method"].astype(str).apply(parse_scenario)
    plot_df["estimator"] = plot_df["Method"].astype(str).apply(parse_estimator)

    plot_df = plot_df[
        plot_df["scenario"].isin(["Multiplicative", "Linear"])
        & plot_df["estimator"].isin(["DoubleML", "EconML", "OLS"])
        & plot_df["DGP"].isin(["TreeFriendly", "PLR", "WGAN"])
    ].copy()

    plot_df = plot_df.sort_values(["scenario", "DGP", "Theta", "estimator"])

    dgps = ["TreeFriendly", "PLR", "WGAN"]
    scenarios = ["Multiplicative", "Linear"]

    fig, axes = plt.subplots(
        nrows=len(scenarios),
        ncols=len(dgps),
        figsize=(14, 6),
        sharex=True,
        sharey=True,
    )

    fig.suptitle("Coverage Rates Comparison: TreeFriendly, PLR & WGAN", fontsize=14)

    legend_handles, legend_labels = None, None

    for i, scen in enumerate(scenarios):
        for j, dgp in enumerate(dgps):
            ax = axes[i, j]

            sub = plot_df[(plot_df["scenario"] == scen) & (plot_df["DGP"] == dgp)]
            if sub.empty:
                ax.set_axis_off()
                continue

            sns.lineplot(
                data=sub,
                x="Theta",
                y="Coverage",
                hue="estimator",
                marker="o",
                ax=ax,
                errorbar=None,
            )

            if legend_handles is None:
                handles, labels = ax.get_legend_handles_labels()
                cleaned = [(h, l) for h, l in zip(handles, labels) if l != "estimator"]
                seen = set()
                uniq = []
                for h, l in cleaned:
                    if l not in seen:
                        uniq.append((h, l))
                        seen.add(l)
                legend_handles = [h for h, _ in uniq]
                legend_labels = [l for _, l in uniq]

            ax.axhline(0.95, color="red", linestyle="--", linewidth=1)
            ax.axhline(0.90, color="gray", linestyle=":", linewidth=1)

            ax.set_title(f"dgp = {dgp}", fontsize=10)
            ax.set_xlabel("theta" if i == len(scenarios) - 1 else "")
            ax.set_ylabel("coverage" if j == 0 else "")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.3)

            if ax.get_legend() is not None:
                ax.get_legend().remove()

        axes[i, 0].annotate(
            f"scenario = {scen}",
            xy=(-0.25, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
        )

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="estimator",
            loc="center left",
            bbox_to_anchor=(0.90, 0.5),
            frameon=True,
        )

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    fname = "coverage_comparison.png"
    plt.savefig(output_dir / fname, dpi=200)
    plt.close()

    print(f"Saved: {fname}")


def plot_bias_variance_grid(df: pd.DataFrame, output_dir: Path):
    """
    Bias-Variance decomposition (stacked), split by scenario (rows) and DGP (cols).
    2x3 layout:
      rows = scenario (Multiplicative, Linear)
      cols = DGP (TreeFriendly, PLR, WGAN)

    - Excludes Naive.
    - Separate figure per estimator (DoubleML, EconML, OLS).
    - Uses tau_hat draws to compute Bias^2 and Variance by Theta.
    - Reference:
        * PLR / TreeFriendly -> tau_ref = 1.0
        * WGAN -> tau_ref = 6250.951
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ["DGP", "Method", "Theta", "tau_hat"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError("DataFrame must contain DGP, Method, Theta, and tau_hat columns.")

    print("Generating Plot: Bias-Variance Grid (2x3)...")

    plot_df = df.copy()

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

    plot_df = plot_df[plot_df["DGP"].isin(["TreeFriendly", "PLR", "WGAN"])].copy()
    plot_df = plot_df[plot_df["scenario"].isin(["Multiplicative", "Linear"])].copy()
    plot_df = plot_df[plot_df["estimator"].isin(["DoubleML", "EconML", "OLS"])].copy()

    dgps = ["TreeFriendly", "PLR", "WGAN"]
    scenarios = ["Multiplicative", "Linear"]
    estimators = ["DoubleML", "EconML", "OLS"]

    TRUE_EFFECT_DEFAULT = 1.0
    TRUE_EFFECT_WGAN = 6250.951

    for est in estimators:
        df_est = plot_df[plot_df["estimator"] == est].copy()
        if df_est.empty:
            print(f"Skipping Plot: No data for estimator={est}")
            continue

        fig, axes = plt.subplots(
            nrows=len(scenarios),
            ncols=len(dgps),
            figsize=(14, 6),
            sharex=True,
        )

        fig.suptitle(f"Bias-Variance Decomposition: {est}", fontsize=16)

        legend_handles, legend_labels = None, None

        for i, scen in enumerate(scenarios):
            for j, dgp in enumerate(dgps):
                ax = axes[i, j]

                sub = df_est[(df_est["scenario"] == scen) & (df_est["DGP"] == dgp)].copy()
                if sub.empty:
                    ax.set_axis_off()
                    continue

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

                ax.stackplot(
                    bv["Theta"],
                    bv["Bias^2"],
                    bv["Variance"],
                    labels=["Bias^2", "Variance"],
                    alpha=0.8,
                )

                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()

                ax.set_title(f"dgp = {dgp} | scenario = {scen}", fontsize=11)
                ax.grid(True, alpha=0.3)

                ax.set_xlabel("theta" if i == len(scenarios) - 1 else "")
                ax.set_ylabel("mse" if j == 0 else "")

                if ax.get_legend() is not None:
                    ax.get_legend().remove()

        if legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(0.88, 0.5),
                frameon=True,
                title="",
            )

        plt.tight_layout(rect=[0, 0, 0.86, 0.93])
        fname = f"bias_variance_{est.lower()}.png"
        plt.savefig(output_dir / fname, dpi=200)
        plt.close()

        print(f"Saved: {fname}")

def plot_tau_distribution_1x3_by_dgp(df: pd.DataFrame, output_dir: Path):
    """
    Creates 3 figures (one per DGP: TreeFriendly, PLR, WGAN).
    Each figure is a 1x3 grid (DoubleML, EconML, OLS).
    Within each panel: estimate distributions by scenario (Naive, Multiplicative, Linear).

    - For colliders: uses ONLY Theta = 1.0 (drops Theta=0).
    - Reference vertical line:
        * PLR / TreeFriendly -> tau = 1.0
        * WGAN -> reference = 6250.951
    - Single global legend outside (right side), close to the figure.
    - For PLR and TreeFriendly: fixed axis scale:
        xlim = (0.4, 1.2), ylim = (0, 18)
    - For WGAN + OLS: uses a dual y-axis (Naive left, Colliders right)
      to avoid flattening densities.
    """
    from matplotlib.lines import Line2D

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ["DGP", "Method", "Theta", "tau_hat"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError(
            "DataFrame must contain DGP, Method, Theta, and tau_hat columns."
        )

    print("Generating Plot: Tau Distribution (1x3 by DGP)...")

    df = df.copy()

    estimators = ["OLS", "DoubleML", "EconML"]

    dgps = ["TreeFriendly", "PLR", "WGAN"]
    scenario_order = ["Naive", "Multiplicative", "Linear"]

    palette = {
        "Naive": "#1F4E79",
        "Multiplicative": "#ff7f0e",
        "Linear": "#4C956C",
    }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def parse_scenario(method: str):
        if method.startswith("Naive_"):
            return "Naive"
        if method.startswith("BadControl_"):
            return "Multiplicative"
        if method.startswith("LinearCollider_"):
            return "Linear"
        return None

    def has_estimator(method: str, est: str) -> bool:
        m = str(method)
        return m.endswith(f"_{est}") or f"_{est}" in m

    def stable_xlim(values: np.ndarray, ref: float | None):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if values.size < 5:
            return None

        q01, q99 = np.quantile(values, [0.01, 0.99])
        if q01 == q99:
            q01, q99 = values.min(), values.max()

        span = max(1e-9, q99 - q01)
        pad = 0.10 * span
        lo, hi = q01 - pad, q99 + pad

        if ref is not None:
            lo = min(lo, ref - pad)
            hi = max(hi, ref + pad)

        if lo == hi:
            lo -= 1.0
            hi += 1.0

        return float(lo), float(hi)

    def plot_density(ax, x, color):
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        if x.size < 5:
            return

        # KDE can fail on degenerate distributions -> fallback to histogram
        if np.std(x) < 1e-9 or np.unique(x).size < 5:
            sns.histplot(
                x=x,
                bins=min(30, max(5, int(np.sqrt(x.size)))),
                stat="density",
                element="step",
                fill=True,
                alpha=0.20,
                color=color,
                ax=ax,
            )
            return

        sns.kdeplot(
            x=x,
            fill=True,
            alpha=0.25,
            linewidth=1.5,
            color=color,
            ax=ax,
            bw_adjust=1.1,
        )

    # ------------------------------------------------------------------ #
    # Prepare data
    # ------------------------------------------------------------------ #
    df["scenario"] = df["Method"].astype(str).apply(parse_scenario)
    df = df[df["scenario"].notnull()].copy()

    # Keep Naive at any theta; colliders only at theta=1.0
    df = df[(df["scenario"] == "Naive") | (df["Theta"] == 1.0)].copy()

    # ------------------------------------------------------------------ #
    # Main loop by DGP
    # ------------------------------------------------------------------ #
    for dgp in dgps:
        sub_dgp = df[df["DGP"] == dgp].copy()
        if sub_dgp.empty:
            print(f"Skipping Plot: No data for DGP={dgp}")
            continue

        ref = 6250.951 if dgp == "WGAN" else 1.0
        ref_label = "WGAN Reference" if dgp == "WGAN" else "True Effect (1.0)"

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), sharey=False)
        fig.suptitle(
            f"{dgp}: Distribution of Estimates",
            fontsize=18,
        )

        # Define filename EARLY (important!)
        fname = f"Distribution_of_Estimates_{dgp.lower()}.png"

        for j, est in enumerate(estimators):
            ax = axes[j]

            sub_est = sub_dgp[
                sub_dgp["Method"].apply(lambda m: has_estimator(m, est))
            ].copy()
            if sub_est.empty:
                ax.set_axis_off()
                continue

            # Special case: WGAN + OLS -> dual y-axis
            if dgp == "WGAN" and est == "OLS":
                ax_left = ax
                ax_right = ax_left.twinx()

                x_naive = sub_est[sub_est["scenario"] == "Naive"]["tau_hat"].values
                plot_density(ax_left, x_naive, palette["Naive"])

                for scen in ["Multiplicative", "Linear"]:
                    x_s = sub_est[sub_est["scenario"] == scen]["tau_hat"].values
                    plot_density(ax_right, x_s, palette[scen])

                pooled = sub_est["tau_hat"].values
                xlim = stable_xlim(pooled, ref=ref)
                if xlim is not None:
                    ax_left.set_xlim(*xlim)

                ax_left.axvline(ref, color="black", linestyle="--", linewidth=1.8)

                ax_left.set_ylabel("Density (Naive)")
                ax_right.set_ylabel("Density (Colliders)")
                ax_left.set_xlabel("tau_hat")
                ax_left.set_title("OLS", fontsize=14)
                ax_left.grid(True, alpha=0.30)
                continue

            # Normal case
            pooled = []
            for scen in scenario_order:
                x_s = sub_est[sub_est["scenario"] == scen]["tau_hat"].values
                if x_s.size > 0:
                    pooled.append(x_s)
                    plot_density(ax, x_s, palette[scen])

            pooled_vals = np.concatenate(pooled) if pooled else np.array([])
            xlim = stable_xlim(pooled_vals, ref=ref)
            if xlim is not None:
                ax.set_xlim(*xlim)

            ax.axvline(ref, color="black", linestyle="--", linewidth=1.8)

            ax.set_title(est, fontsize=14)
            ax.set_xlabel("tau_hat")
            ax.grid(True, alpha=0.30)
            ax.set_ylabel("Density" if j == 0 else "")

            # Fixed scale for simulated DGPs
            if dgp in ["PLR", "TreeFriendly"]:
                ax.set_xlim(0.4, 1.2)
                ax.set_ylim(0.0, 18.0)

        # ------------------------------------------------------------------ #
        # Global legend (manual, robust)
        # ------------------------------------------------------------------ #
        legend_elements = [
            Line2D([0], [0], color=palette["Naive"], lw=2, label="Naive"),
            Line2D([0], [0], color=palette["Multiplicative"], lw=2, label="Multiplicative"),
            Line2D([0], [0], color=palette["Linear"], lw=2, label="Linear"),
            Line2D([0], [0], color="black", lw=2, linestyle="--", label=ref_label),
        ]

        fig.legend(
            handles=legend_elements,
            title="Scenario",
            loc="center left",
            bbox_to_anchor=(0.85, 0.5),
            frameon=True,
        )

        plt.tight_layout(rect=[0, 0, 0.86, 0.92])
        plt.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved: {fname}")
