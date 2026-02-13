import pandas as pd
from pathlib import Path

def generate_table():
    results_path = Path("results/detailed_summary_metrics.csv")
    if not results_path.exists():
        print("Results file not found.")
        return

    df = pd.read_csv(results_path)

    estimators = ["OLS", "DoubleML", "EconML"]

    def get_metrics(dgp, scenario, theta, estimator):
        if scenario == "Naive":
            method_str = f"Naive_{estimator}"
        else:
            method_str = f"{scenario}_{estimator}"
            
        subset = df[
            (df["DGP"] == dgp) & 
            (df["Method"] == method_str) & 
            (abs(df["Theta"] - theta) < 1e-5)
        ]
        if subset.empty:
            return None
        
        row_data = subset.iloc[0]
        return {
            "Est_ATE": row_data["Tau_Mean"],
            "Bias": row_data["Bias_Mean"],
            "RMSE": row_data["RMSE"],
            "Coverage": row_data["Coverage"],
            "CI_Width": row_data["CI_Length_Mean"],
            "Std_Err": row_data["Tau_Std"]
        }

    dgps = ["TreeFriendly", "PLR", "WGAN"]
    
    latex_file = Path("appendix/comparison_table.tex")
    
    with open(latex_file, "w") as f_tex:
        f_tex.write("\\begin{table}[h]\n\\centering\n")
        f_tex.write("\\caption{Comprehensive Estimator Performance Comparison with Relative Bias}\n")
        f_tex.write("\\label{tab:benchmark_comparison_detailed}\n")
        f_tex.write("\\resizebox{\\textwidth}{!}{%\n")
        f_tex.write("\\begin{tabular}{llccccccc}\n\\toprule\n")
        f_tex.write("\\textbf{DGP} & \\textbf{Est.} & \\textbf{Scenario} & \\textbf{True ATE} & \\textbf{Est. ATE} & \\textbf{Bias} & \\textbf{Rel. Bias} & \\textbf{CI Width} & \\textbf{Cov.} \\\\\n\\midrule\n")

        def write_panel(panel_name, scenario, theta, f_tex):
            f_tex.write(f"\\multicolumn{{9}}{{l}}{{\\textit{{{panel_name}}}}} \\\\\n")
            
            for dgp in dgps:
                for est in estimators:
                    metrics = get_metrics(dgp, scenario, theta, est)
                    if metrics:
                        true_ate_val = 1.0 if dgp != "WGAN" else 6250.0
                        rel_bias = metrics['Bias'] / true_ate_val
                        
                        if dgp == "WGAN":
                            true_ate_str = f"{true_ate_val:,.0f}"
                            est_ate_str = f"{metrics['Est_ATE']:,.2f}"
                            bias_str = f"{metrics['Bias']:,.2f}"
                            ci_width_str = f"{metrics['CI_Width']:,.2f}"
                        else:
                            true_ate_str = f"{true_ate_val:.1f}"
                            est_ate_str = f"{metrics['Est_ATE']:.2f}"
                            bias_str = f"{metrics['Bias']:.2f}"
                            ci_width_str = f"{metrics['CI_Width']:.2f}"
                        
                        rel_bias_str = f"{rel_bias:.2f}"
                        cov_str = f"{metrics['Coverage']:.2f}"
                        
                        dgp_tex = f"\\textbf{{{dgp}}}" if dgp == "WGAN" else dgp
                        est_tex = f"\\textbf{{{est}}}" if dgp == "WGAN" else est
                        scen_tex = f"\\textbf{{{scenario if scenario != 'Naive' else 'Naive'}}}" if dgp == "WGAN" else (scenario if scenario != 'Naive' else 'Naive')
                        
                        def b(x, is_wgan): return f"\\textbf{{{x}}}" if is_wgan else x
                        is_wgan = (dgp == "WGAN")
                        
                        row_tex = f"{dgp_tex} & {est_tex} & {scen_tex} & {b(true_ate_str, is_wgan)} & {b(est_ate_str, is_wgan)} & {b(bias_str, is_wgan)} & {b(rel_bias_str, is_wgan)} & {b(ci_width_str, is_wgan)} & {b(cov_str, is_wgan)} \\\\\n"
                        f_tex.write(row_tex)
            
            f_tex.write("\\midrule\n")

        write_panel("Panel A: Baseline (Naive, Theta=0)", "Naive", 0.0, f_tex)
        write_panel("Panel B: Linear Collider", "LinearCollider", 1.0, f_tex)
        f_tex.write(f"\\multicolumn{{9}}{{l}}{{\\textit{{Panel C: Multiplicative Collider}}}} \\\\\n")
        
        for dgp in dgps:
            for est in estimators:
                metrics = get_metrics(dgp, "BadControl", 1.0, est)
                if metrics:
                    true_ate_val = 1.0 if dgp != "WGAN" else 6250.0
                    rel_bias = metrics['Bias'] / true_ate_val
                     
                    if dgp == "WGAN":
                        true_ate_str = f"{true_ate_val:,.0f}"
                        est_ate_str = f"{metrics['Est_ATE']:,.2f}"
                        bias_str = f"{metrics['Bias']:,.2f}"
                        ci_width_str = f"{metrics['CI_Width']:,.2f}"
                    else:
                        true_ate_str = f"{true_ate_val:.1f}"
                        est_ate_str = f"{metrics['Est_ATE']:.2f}"
                        bias_str = f"{metrics['Bias']:.2f}"
                        ci_width_str = f"{metrics['CI_Width']:.2f}"
                    
                    rel_bias_str = f"{rel_bias:.2f}"
                    cov_str = f"{metrics['Coverage']:.2f}"
                    
                    dgp_tex = f"\\textbf{{{dgp}}}" if dgp == "WGAN" else dgp
                    est_tex = f"\\textbf{{{est}}}" if dgp == "WGAN" else est
                    scen_tex = f"\\textbf{{Multiplicative}}" if dgp == "WGAN" else "Multiplicative"
                    
                    def b(x, is_wgan): return f"\\textbf{{{x}}}" if is_wgan else x
                    is_wgan = (dgp == "WGAN")
                    
                    row_tex = f"{dgp_tex} & {est_tex} & {scen_tex} & {b(true_ate_str, is_wgan)} & {b(est_ate_str, is_wgan)} & {b(bias_str, is_wgan)} & {b(rel_bias_str, is_wgan)} & {b(ci_width_str, is_wgan)} & {b(cov_str, is_wgan)} \\\\\n"
                    f_tex.write(row_tex)

        f_tex.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")
    
    print(f"LaTeX table written to {latex_file}")

if __name__ == "__main__":
    generate_table()
