"""
Analyze experiment results: statistics, figures, error taxonomy.
Runs autonomously as part of the pipeline.

Usage:
    python analyze_results.py --results-dir results/20260313_... --output results/.../analysis/
"""

import argparse
import json
import os
import sys
import glob
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Load all results.jsonl files into a single DataFrame."""
    rows = []
    for jsonl_path in glob.glob(os.path.join(results_dir, "**", "results.jsonl"), recursive=True):
        rel_path = os.path.relpath(jsonl_path, results_dir)
        with open(jsonl_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    row["_source_file"] = rel_path
                    # Flatten token info
                    tokens = row.pop("tokens", {})
                    row["total_input_tokens"] = tokens.get("total_input_tokens", 0)
                    row["total_output_tokens"] = tokens.get("total_output_tokens", 0)
                    row["total_latency_ms"] = tokens.get("total_latency_ms", 0)
                    row["num_llm_calls"] = tokens.get("num_calls", 0)
                    rows.append(row)
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} results from {len(set(df.get('_source_file', [])))} files")
    return df


# ═══════════════════════════════════════
# EXP-1: Benchmark Comparison
# ═══════════════════════════════════════

def analyze_exp1(df: pd.DataFrame, output_dir: str):
    """Ablation across conditions C1-C5."""
    exp1 = df[df["_source_file"].str.contains("exp1/refactorbench")]
    if exp1.empty:
        logger.warning("No EXP-1 RefactorBench data found")
        return

    conditions = ["c1_direct", "c2_rag", "c3_multiagent", "c4_full", "c5_memory"]
    present = [c for c in conditions if c in exp1["condition"].values]

    # Violin plot of judge scores by condition
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = exp1[exp1["condition"].isin(present) & exp1["success"]]
    if not plot_data.empty:
        sns.violinplot(data=plot_data, x="condition", y="judge_score",
                       order=present, ax=ax, inner="box", cut=0)
        ax.set_title("EXP-1: Intent Verification Score by Condition (RefactorBench)")
        ax.set_ylabel("Judge Score (0-100)")
        ax.set_xlabel("Condition")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "exp1_violin_conditions.pdf"), dpi=150)
        plt.close()

    # Success rate bar chart
    success_rates = {}
    for cond in present:
        cond_data = exp1[exp1["condition"] == cond]
        if len(cond_data) > 0:
            success_rates[cond] = cond_data["success"].mean() * 100

    if success_rates:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(success_rates.keys(), success_rates.values(), color="steelblue")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("EXP-1: Pipeline Success Rate by Condition")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "exp1_success_rate.pdf"), dpi=150)
        plt.close()

    # Statistical tests
    stats_results = {}
    groups = [exp1[exp1["condition"] == c]["judge_score"].dropna().values for c in present]
    groups = [g for g in groups if len(g) > 1]

    if len(groups) >= 3:
        stat, p = stats.friedmanchisquare(*groups)
        stats_results["friedman"] = {"statistic": float(stat), "p_value": float(p)}

    # Pairwise Wilcoxon
    pairwise = {}
    for i, c1 in enumerate(present):
        for c2 in present[i+1:]:
            g1 = exp1[exp1["condition"] == c1]["judge_score"].dropna()
            g2 = exp1[exp1["condition"] == c2]["judge_score"].dropna()
            min_len = min(len(g1), len(g2))
            if min_len >= 5:
                stat, p = stats.wilcoxon(g1.values[:min_len], g2.values[:min_len])
                # Cliff's delta
                delta = cliffs_delta(g1.values, g2.values)
                pairwise[f"{c1}_vs_{c2}"] = {
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p),
                    "cliffs_delta": float(delta),
                }
    stats_results["pairwise_wilcoxon"] = pairwise

    # Token efficiency
    efficiency = {}
    for cond in present:
        cond_data = exp1[(exp1["condition"] == cond) & exp1["success"]]
        if len(cond_data) > 0:
            efficiency[cond] = {
                "median_judge_score": float(cond_data["judge_score"].median()),
                "mean_judge_score": float(cond_data["judge_score"].mean()),
                "median_total_tokens": float(cond_data["total_input_tokens"].median() +
                                             cond_data["total_output_tokens"].median()),
                "median_latency_ms": float(cond_data["total_latency_ms"].median()),
                "success_rate": float(exp1[exp1["condition"] == cond]["success"].mean()),
            }
    stats_results["efficiency"] = efficiency

    with open(os.path.join(output_dir, "exp1_statistics.json"), "w") as f:
        json.dump(stats_results, f, indent=2)

    logger.info("EXP-1 analysis complete")


# ═══════════════════════════════════════
# EXP-2: Multi-Model Comparison
# ═══════════════════════════════════════

def analyze_exp2(df: pd.DataFrame, output_dir: str):
    """Compare generator models."""
    exp2 = df[df["_source_file"].str.contains("exp2/")]
    if exp2.empty:
        logger.warning("No EXP-2 data found")
        return

    models = exp2["generator_model"].unique()

    # Violin plot by model
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data = exp2[exp2["success"]]
    if not plot_data.empty:
        sns.violinplot(data=plot_data, x="generator_model", y="judge_score",
                       ax=ax, inner="box", cut=0)
        ax.set_title("EXP-2: Intent Verification Score by Generator Model")
        ax.set_ylabel("Judge Score (0-100)")
        ax.set_xlabel("Generator Model")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "exp2_model_comparison.pdf"), dpi=150)
        plt.close()

    # Model summary table
    summary = []
    for model in models:
        m_data = exp2[exp2["generator_model"] == model]
        m_success = m_data[m_data["success"]]
        summary.append({
            "model": model,
            "provider": m_data["generator_provider"].iloc[0] if len(m_data) > 0 else "",
            "n_tasks": len(m_data),
            "success_rate": f"{m_data['success'].mean()*100:.1f}%",
            "median_score": f"{m_success['judge_score'].median():.1f}" if len(m_success) > 0 else "N/A",
            "mean_score": f"{m_success['judge_score'].mean():.1f}" if len(m_success) > 0 else "N/A",
            "median_tokens": f"{(m_success['total_input_tokens'] + m_success['total_output_tokens']).median():.0f}" if len(m_success) > 0 else "N/A",
        })

    with open(os.path.join(output_dir, "exp2_model_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("EXP-2 analysis complete")


# ═══════════════════════════════════════
# EXP-3: Judge Validation
# ═══════════════════════════════════════

def analyze_exp3(df: pd.DataFrame, output_dir: str):
    """Validate LLM judge against execution-based results."""
    # Use all data with both judge_score and success fields
    valid = df[df["judge_score"].notna() & df["success"].notna()].copy()
    if valid.empty:
        logger.warning("No data for EXP-3 analysis")
        return

    results = {}

    # Point-biserial correlation: judge_score vs success (binary)
    if valid["success"].nunique() > 1:
        corr, p = stats.pointbiserialr(valid["success"].astype(int), valid["judge_score"])
        results["judge_vs_success"] = {
            "point_biserial_r": float(corr),
            "p_value": float(p),
        }

    # AUROC: judge score as predictor of success
    try:
        from sklearn.metrics import roc_auc_score
        if valid["success"].nunique() > 1:
            auroc = roc_auc_score(valid["success"].astype(int), valid["judge_score"])
            results["auroc"] = float(auroc)
    except ImportError:
        pass

    # Calibration curve
    bins = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
    calibration = []
    for lo, hi in bins:
        bin_data = valid[(valid["judge_score"] >= lo) & (valid["judge_score"] <= hi)]
        if len(bin_data) > 0:
            calibration.append({
                "bin": f"{lo}-{hi}",
                "n_tasks": len(bin_data),
                "actual_success_rate": float(bin_data["success"].mean()),
                "mean_judge_score": float(bin_data["judge_score"].mean()),
            })
    results["calibration"] = calibration

    # Plot calibration
    if calibration:
        fig, ax = plt.subplots(figsize=(8, 6))
        x = [c["mean_judge_score"] for c in calibration]
        y = [c["actual_success_rate"] * 100 for c in calibration]
        sizes = [c["n_tasks"] * 5 for c in calibration]
        ax.scatter(x, y, s=sizes, alpha=0.7, color="steelblue")
        ax.plot([0, 100], [0, 100], '--', color='gray', alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Mean Judge Score")
        ax.set_ylabel("Actual Success Rate (%)")
        ax.set_title("EXP-3: Judge Calibration Curve")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "exp3_calibration.pdf"), dpi=150)
        plt.close()

    # Per-generator bias check
    if "generator_model" in valid.columns:
        bias_check = {}
        success_only = valid[valid["success"]]
        if len(success_only) > 0 and success_only["generator_model"].nunique() > 1:
            groups = [g["judge_score"].values for _, g in success_only.groupby("generator_model")]
            groups = [g for g in groups if len(g) >= 3]
            if len(groups) >= 2:
                stat, p = stats.kruskal(*groups)
                bias_check["kruskal_wallis"] = {"statistic": float(stat), "p_value": float(p)}
        results["per_generator_bias"] = bias_check

    with open(os.path.join(output_dir, "exp3_judge_validation.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("EXP-3 analysis complete")


# ═══════════════════════════════════════
# EXP-4: Ablation Efficiency
# ═══════════════════════════════════════

def analyze_exp4(df: pd.DataFrame, output_dir: str):
    """Ablation with compute-controlled baseline."""
    # Combine EXP-1 conditions (reused) + EXP-4 a5_ctrl
    ablation_conditions = ["c1_direct", "c2_rag", "c4_full", "c5_memory", "a5_ctrl"]
    abl = df[df["condition"].isin(ablation_conditions) &
             df["_source_file"].str.contains("exp1/refactorbench|exp4")]

    if abl.empty:
        logger.warning("No EXP-4 data found")
        return

    # Efficiency frontier: score vs tokens
    fig, ax = plt.subplots(figsize=(10, 7))
    for cond in ablation_conditions:
        cond_data = abl[(abl["condition"] == cond) & abl["success"]]
        if len(cond_data) > 0:
            total_tokens = cond_data["total_input_tokens"] + cond_data["total_output_tokens"]
            ax.scatter(total_tokens, cond_data["judge_score"],
                       alpha=0.3, label=cond, s=30)
            # Add median marker
            ax.scatter(total_tokens.median(), cond_data["judge_score"].median(),
                       marker="D", s=100, edgecolors="black", zorder=5)

    ax.set_xlabel("Total Tokens (input + output)")
    ax.set_ylabel("Judge Score (0-100)")
    ax.set_title("EXP-4: Quality vs. Compute (Efficiency Frontier)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp4_efficiency_frontier.pdf"), dpi=150)
    plt.close()

    logger.info("EXP-4 analysis complete")


# ═══════════════════════════════════════
# EXP-6: RAG Comparison
# ═══════════════════════════════════════

def analyze_exp6(df: pd.DataFrame, output_dir: str):
    """RAG embedding x retrieval strategy heatmap."""
    exp6 = df[df["_source_file"].str.contains("exp6/")]
    if exp6.empty:
        logger.warning("No EXP-6 data found")
        return

    # Parse config from source file path
    configs = []
    for _, row in exp6.iterrows():
        parts = row["_source_file"].split("/")
        for p in parts:
            if "rag_" in p:
                configs.append(p)
                break
        else:
            configs.append("unknown")
    exp6 = exp6.copy()
    exp6["rag_config"] = configs

    # Aggregate by config
    summary = exp6[exp6["success"]].groupby("rag_config").agg(
        mean_score=("judge_score", "mean"),
        median_score=("judge_score", "median"),
        success_rate=("success", "mean"),
        n=("task_id", "count"),
    ).reset_index()

    summary.to_json(os.path.join(output_dir, "exp6_rag_summary.json"), orient="records", indent=2)

    # Bar chart
    if len(summary) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        summary_sorted = summary.sort_values("mean_score", ascending=False)
        ax.barh(summary_sorted["rag_config"], summary_sorted["mean_score"], color="steelblue")
        ax.set_xlabel("Mean Judge Score")
        ax.set_title("EXP-6: RAG Configuration Comparison")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "exp6_rag_comparison.pdf"), dpi=150)
        plt.close()

    logger.info("EXP-6 analysis complete")


# ═══════════════════════════════════════
# EXP-7: Error Taxonomy
# ═══════════════════════════════════════

ERROR_CATEGORIES = {
    "E1": "Hallucinated import",
    "E2": "Wrong API usage",
    "E3": "Incomplete implementation",
    "E4": "Cross-file inconsistency",
    "E5": "Naming convention violation",
    "E6": "Structural preservation failure",
    "E7": "Logic error",
    "E8": "Framework misuse",
    "E9": "Partial completion",
}

def analyze_exp7(df: pd.DataFrame, output_dir: str):
    """Auto-label failures with error categories."""
    failures = df[~df["success"]].copy()
    if failures.empty:
        logger.info("No failures to analyze for EXP-7")
        return

    # Auto-classify based on error messages and generated code
    categories = []
    for _, row in failures.iterrows():
        error = str(row.get("error", "")).lower()
        code = str(row.get("generated_code", "")).lower()

        if "import" in error or "modulenotfound" in error or "no module" in error:
            categories.append("E1")
        elif "todo" in code or "pass #" in code or "placeholder" in code:
            categories.append("E3")
        elif "typeerror" in error or "argument" in error or "signature" in error:
            categories.append("E2")
        elif "syntax" in error or "parse" in error:
            categories.append("E7")
        elif "timeout" in error or "rate" in error:
            categories.append("API_ERROR")
        else:
            categories.append("E9")  # default: partial completion

    failures["error_category"] = categories

    # Frequency distribution
    freq = Counter(categories)
    freq_named = {f"{k} ({ERROR_CATEGORIES.get(k, 'API Error')})": v
                  for k, v in sorted(freq.items(), key=lambda x: -x[1])}

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(freq_named.keys()), list(freq_named.values()), color="coral")
    ax.set_xlabel("Number of Failures")
    ax.set_title("EXP-7: Error Category Distribution")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp7_error_taxonomy.pdf"), dpi=150)
    plt.close()

    # Save detailed results
    failure_summary = {
        "total_failures": len(failures),
        "category_counts": dict(freq),
        "by_condition": failures.groupby(["condition", "error_category"]).size().reset_index(name="count").to_dict(orient="records"),
        "by_model": failures.groupby(["generator_model", "error_category"]).size().reset_index(name="count").to_dict(orient="records") if "generator_model" in failures.columns else [],
    }

    with open(os.path.join(output_dir, "exp7_error_taxonomy.json"), "w") as f:
        json.dump(failure_summary, f, indent=2)

    logger.info(f"EXP-7 analysis complete: {len(failures)} failures categorized")


# ═══════════════════════════════════════
# LaTeX Table Generation
# ═══════════════════════════════════════

def generate_latex_tables(df: pd.DataFrame, output_dir: str):
    """Generate LaTeX tables ready to paste into the paper."""
    latex = []

    # Table: EXP-1 condition comparison
    exp1 = df[df["_source_file"].str.contains("exp1/refactorbench")]
    if not exp1.empty:
        latex.append("% EXP-1: Condition Comparison (RefactorBench)")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Ablation results across experimental conditions on RefactorBench}")
        latex.append("\\small")
        latex.append("\\begin{tabular}{@{}lccccc@{}}")
        latex.append("\\toprule")
        latex.append("\\textbf{Condition} & \\textbf{Success\\%} & \\textbf{Med. Score} & \\textbf{Mean Score} & \\textbf{Med. Tokens} & \\textbf{Med. Time (s)} \\\\")
        latex.append("\\midrule")

        for cond in ["c1_direct", "c2_rag", "c3_multiagent", "c4_full", "c5_memory"]:
            c = exp1[exp1["condition"] == cond]
            if len(c) == 0:
                continue
            cs = c[c["success"]]
            sr = c["success"].mean() * 100
            med = cs["judge_score"].median() if len(cs) > 0 else 0
            mean = cs["judge_score"].mean() if len(cs) > 0 else 0
            tokens = (cs["total_input_tokens"] + cs["total_output_tokens"]).median() if len(cs) > 0 else 0
            time_s = cs["total_latency_ms"].median() / 1000 if len(cs) > 0 else 0
            latex.append(f"{cond} & {sr:.1f}\\% & {med:.1f} & {mean:.1f} & {tokens:.0f} & {time_s:.1f} \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")

    # Table: EXP-2 model comparison
    exp2 = df[df["_source_file"].str.contains("exp2/")]
    if not exp2.empty:
        latex.append("% EXP-2: Model Comparison")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Multi-model comparison on RefactorBench}")
        latex.append("\\small")
        latex.append("\\begin{tabular}{@{}llccc@{}}")
        latex.append("\\toprule")
        latex.append("\\textbf{Model} & \\textbf{Provider} & \\textbf{Success\\%} & \\textbf{Med. Score} & \\textbf{Mean Score} \\\\")
        latex.append("\\midrule")

        for model in exp2["generator_model"].unique():
            m = exp2[exp2["generator_model"] == model]
            ms = m[m["success"]]
            provider = m["generator_provider"].iloc[0]
            sr = m["success"].mean() * 100
            med = ms["judge_score"].median() if len(ms) > 0 else 0
            mean = ms["judge_score"].mean() if len(ms) > 0 else 0
            model_short = model.split("/")[-1] if "/" in model else model
            latex.append(f"{model_short} & {provider} & {sr:.1f}\\% & {med:.1f} & {mean:.1f} \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

    with open(os.path.join(output_dir, "tables.tex"), "w") as f:
        f.write("\n".join(latex))

    logger.info("LaTeX tables generated")


# ═══════════════════════════════════════
# Utilities
# ═══════════════════════════════════════

def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logger.info(f"Loading results from {args.results_dir}")
    df = load_all_results(args.results_dir)

    if df.empty:
        logger.error("No results found!")
        sys.exit(1)

    logger.info(f"Total results: {len(df)}")

    # Run all analyses
    analyze_exp1(df, args.output)
    analyze_exp2(df, args.output)
    analyze_exp3(df, args.output)
    analyze_exp4(df, args.output)
    analyze_exp6(df, args.output)
    analyze_exp7(df, args.output)
    generate_latex_tables(df, args.output)

    # Save full summary
    summary = {
        "total_results": len(df),
        "total_successful": int(df["success"].sum()) if "success" in df else 0,
        "experiments": list(df["_source_file"].str.split("/").str[0].unique()),
        "conditions": list(df["condition"].unique()) if "condition" in df else [],
        "models": list(df["generator_model"].unique()) if "generator_model" in df else [],
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"All analysis complete. Output: {args.output}")


if __name__ == "__main__":
    main()
