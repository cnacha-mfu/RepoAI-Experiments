"""
RefactorBench AST Evaluation.
Runs structural analysis on generated refactoring patches:
  - Python AST parse validity
  - Function/class signature preservation
  - Cyclomatic complexity delta
  - Diff statistics (lines added/removed/changed)

Usage:
    python eval_refactorbench.py \
        --results-dir results/20260313_120000 \
        --output results/20260313_120000/eval/refactorbench/
"""

import argparse
import ast
import json
import os
import re
import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- AST Analysis ---

def check_python_syntax(code: str) -> dict:
    """Check if Python code is syntactically valid via AST parsing."""
    try:
        tree = ast.parse(code)
        return {"valid": True, "error": None, "node_count": len(list(ast.walk(tree)))}
    except SyntaxError as e:
        return {"valid": False, "error": f"Line {e.lineno}: {e.msg}", "node_count": 0}


def extract_signatures(code: str) -> dict:
    """Extract function/class signatures from Python code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"functions": [], "classes": [], "methods": []}

    functions = []
    classes = []
    methods = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            sig = {
                "name": node.name,
                "args": [a.arg for a in node.args.args],
                "decorators": [_decorator_name(d) for d in node.decorator_list],
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "lineno": node.lineno,
            }
            # Check if it's a method (nested under a class)
            # We'll track this separately based on parent context
            functions.append(sig)

        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.dump(base))
            classes.append({
                "name": node.name,
                "bases": bases,
                "decorators": [_decorator_name(d) for d in node.decorator_list],
                "lineno": node.lineno,
            })

    return {"functions": functions, "classes": classes}


def _decorator_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr
    elif isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return "unknown"


def compute_complexity(code: str) -> int:
    """Estimate cyclomatic complexity (simplified: count branches)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -1

    complexity = 1  # base
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            complexity += 1
    return complexity


def compare_signatures(original_sigs: dict, refactored_sigs: dict) -> dict:
    """Compare function/class signatures between original and refactored code."""
    orig_funcs = {f["name"]: f for f in original_sigs["functions"]}
    ref_funcs = {f["name"]: f for f in refactored_sigs["functions"]}
    orig_classes = {c["name"]: c for c in original_sigs["classes"]}
    ref_classes = {c["name"]: c for c in refactored_sigs["classes"]}

    result = {
        "functions_added": [],
        "functions_removed": [],
        "functions_signature_changed": [],
        "functions_preserved": [],
        "classes_added": [],
        "classes_removed": [],
        "classes_preserved": [],
    }

    # Functions
    for name in set(orig_funcs) | set(ref_funcs):
        if name in orig_funcs and name not in ref_funcs:
            result["functions_removed"].append(name)
        elif name not in orig_funcs and name in ref_funcs:
            result["functions_added"].append(name)
        else:
            if orig_funcs[name]["args"] != ref_funcs[name]["args"]:
                result["functions_signature_changed"].append(name)
            else:
                result["functions_preserved"].append(name)

    # Classes
    for name in set(orig_classes) | set(ref_classes):
        if name in orig_classes and name not in ref_classes:
            result["classes_removed"].append(name)
        elif name not in orig_classes and name in ref_classes:
            result["classes_added"].append(name)
        else:
            result["classes_preserved"].append(name)

    return result


def compute_diff_stats(original: str, refactored: str) -> dict:
    """Compute basic diff statistics between two code strings."""
    orig_lines = original.strip().splitlines()
    ref_lines = refactored.strip().splitlines()

    orig_set = set(orig_lines)
    ref_set = set(ref_lines)

    added = len(ref_set - orig_set)
    removed = len(orig_set - ref_set)
    unchanged = len(orig_set & ref_set)

    return {
        "original_lines": len(orig_lines),
        "refactored_lines": len(ref_lines),
        "lines_added": added,
        "lines_removed": removed,
        "lines_unchanged": unchanged,
        "churn_ratio": (added + removed) / max(len(orig_lines), 1),
    }


# --- Evaluation Pipeline ---

def evaluate_task(result: dict) -> dict:
    """Evaluate a single RefactorBench task result."""
    eval_result = {
        "task_id": result.get("task_id", "unknown"),
        "condition": result.get("condition", "unknown"),
        "success": result.get("success", False),
    }

    if not result.get("success") or not result.get("generated_code"):
        eval_result["eval_status"] = "skipped_no_output"
        return eval_result

    generated = result["generated_code"]
    original = result.get("original_code", "")

    # 1. Syntax check
    syntax = check_python_syntax(generated)
    eval_result["syntax_valid"] = syntax["valid"]
    eval_result["syntax_error"] = syntax["error"]
    eval_result["ast_node_count"] = syntax["node_count"]

    # 2. Signature analysis (only if both original and refactored are valid Python)
    if original:
        orig_syntax = check_python_syntax(original)
        if orig_syntax["valid"] and syntax["valid"]:
            orig_sigs = extract_signatures(original)
            ref_sigs = extract_signatures(generated)
            sig_comparison = compare_signatures(orig_sigs, ref_sigs)
            eval_result["signature_analysis"] = sig_comparison
            eval_result["api_preserved"] = (
                len(sig_comparison["functions_removed"]) == 0
                and len(sig_comparison["functions_signature_changed"]) == 0
                and len(sig_comparison["classes_removed"]) == 0
            )

    # 3. Complexity
    if syntax["valid"]:
        eval_result["complexity"] = compute_complexity(generated)
        if original:
            orig_complexity = compute_complexity(original)
            if orig_complexity >= 0:
                eval_result["original_complexity"] = orig_complexity
                eval_result["complexity_delta"] = eval_result["complexity"] - orig_complexity

    # 4. Diff stats
    if original:
        eval_result["diff_stats"] = compute_diff_stats(original, generated)

    # 5. Judge score passthrough
    if "judge_score" in result:
        eval_result["judge_score"] = result["judge_score"]
    if "judge_explanation" in result:
        eval_result["judge_explanation"] = result["judge_explanation"]

    eval_result["eval_status"] = "complete"
    return eval_result


def evaluate_results_file(results_path: str) -> list:
    """Evaluate all results in a JSONL file."""
    evaluations = []
    with open(results_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                evaluation = evaluate_task(result)
                evaluations.append(evaluation)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON at line {line_num} in {results_path}")
            except Exception as e:
                logger.warning(f"Error evaluating line {line_num}: {e}")
    return evaluations


def compute_aggregate_stats(evaluations: list) -> dict:
    """Compute aggregate statistics across all evaluated tasks."""
    if not evaluations:
        return {}

    total = len(evaluations)
    completed = [e for e in evaluations if e.get("eval_status") == "complete"]
    successful = [e for e in evaluations if e.get("success")]

    stats = {
        "total_tasks": total,
        "completed_evals": len(completed),
        "successful_generations": len(successful),
        "success_rate": len(successful) / total if total else 0,
    }

    # Syntax
    syntax_checked = [e for e in completed if "syntax_valid" in e]
    if syntax_checked:
        valid = sum(1 for e in syntax_checked if e["syntax_valid"])
        stats["syntax_valid_rate"] = valid / len(syntax_checked)
        stats["syntax_valid_count"] = valid
        stats["syntax_total_checked"] = len(syntax_checked)

    # API preservation
    api_checked = [e for e in completed if "api_preserved" in e]
    if api_checked:
        preserved = sum(1 for e in api_checked if e["api_preserved"])
        stats["api_preserved_rate"] = preserved / len(api_checked)

    # Complexity
    complexity_tasks = [e for e in completed if "complexity_delta" in e]
    if complexity_tasks:
        deltas = [e["complexity_delta"] for e in complexity_tasks]
        stats["avg_complexity_delta"] = sum(deltas) / len(deltas)
        stats["complexity_reduced"] = sum(1 for d in deltas if d < 0)
        stats["complexity_increased"] = sum(1 for d in deltas if d > 0)
        stats["complexity_unchanged"] = sum(1 for d in deltas if d == 0)

    # Diff stats
    diff_tasks = [e for e in completed if "diff_stats" in e]
    if diff_tasks:
        churns = [e["diff_stats"]["churn_ratio"] for e in diff_tasks]
        stats["avg_churn_ratio"] = sum(churns) / len(churns)
        stats["avg_lines_added"] = sum(e["diff_stats"]["lines_added"] for e in diff_tasks) / len(diff_tasks)
        stats["avg_lines_removed"] = sum(e["diff_stats"]["lines_removed"] for e in diff_tasks) / len(diff_tasks)

    # Judge scores
    judge_tasks = [e for e in completed if "judge_score" in e and e["judge_score"] is not None]
    if judge_tasks:
        scores = [e["judge_score"] for e in judge_tasks]
        stats["avg_judge_score"] = sum(scores) / len(scores)
        stats["median_judge_score"] = sorted(scores)[len(scores) // 2]
        stats["judge_pass_rate_70"] = sum(1 for s in scores if s >= 70) / len(scores)
        stats["judge_pass_rate_80"] = sum(1 for s in scores if s >= 80) / len(scores)

    return stats


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="RefactorBench AST Evaluation")
    parser.add_argument("--results-dir", required=True,
                        help="Root results directory (contains exp1/, exp2/, etc.)")
    parser.add_argument("--output", required=True,
                        help="Output directory for evaluation results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning for RefactorBench results in {results_dir}")

    # Find all results.jsonl files under refactorbench directories
    result_files = list(results_dir.glob("**/refactorbench/**/results.jsonl"))
    # Also check exp2/ and exp4/ (they use refactorbench too)
    result_files += list(results_dir.glob("exp2/**/results.jsonl"))
    result_files += list(results_dir.glob("exp4/**/results.jsonl"))
    # Also check exp6/
    result_files += list(results_dir.glob("exp6/**/results.jsonl"))

    # Deduplicate
    result_files = list(set(result_files))

    if not result_files:
        logger.warning("No results.jsonl files found!")
        return

    logger.info(f"Found {len(result_files)} result files to evaluate")

    all_stats = {}
    all_evals = {}

    for rf in sorted(result_files):
        # Derive a label from the path
        rel = rf.relative_to(results_dir)
        label = str(rel.parent).replace(os.sep, "/")
        logger.info(f"Evaluating: {label}")

        evaluations = evaluate_results_file(str(rf))
        stats = compute_aggregate_stats(evaluations)

        all_stats[label] = stats
        all_evals[label] = evaluations

        # Write per-condition evaluation
        cond_dir = output_dir / label
        cond_dir.mkdir(parents=True, exist_ok=True)

        with open(cond_dir / "evaluations.jsonl", "w") as f:
            for ev in evaluations:
                f.write(json.dumps(ev) + "\n")

        with open(cond_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"  {label}: {stats.get('success_rate', 0):.1%} success, "
                     f"{stats.get('syntax_valid_rate', 0):.1%} syntax valid, "
                     f"avg judge={stats.get('avg_judge_score', 0):.1f}")

    # Write summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_conditions": len(all_stats),
            "conditions": all_stats,
        }, f, indent=2)

    # Write LaTeX table
    latex_path = output_dir / "table_refactorbench.tex"
    _write_latex_table(all_stats, latex_path)

    logger.info(f"Evaluation complete. Summary: {summary_path}")


def _write_latex_table(stats: dict, path: Path):
    """Write a LaTeX table summarizing RefactorBench results."""
    with open(path, "w") as f:
        f.write("% Auto-generated by eval_refactorbench.py\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{RefactorBench AST Evaluation Results}\n")
        f.write("\\label{tab:refactorbench-eval}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Condition & Tasks & Success\\% & Syntax\\% & API Pres.\\% & Avg $\\Delta$CC & Judge \\\\\n")
        f.write("\\midrule\n")

        for label, s in sorted(stats.items()):
            if not s:
                continue
            cond = label.split("/")[-1] if "/" in label else label
            f.write(f"{cond} & "
                    f"{s.get('total_tasks', 0)} & "
                    f"{s.get('success_rate', 0)*100:.1f} & "
                    f"{s.get('syntax_valid_rate', 0)*100:.1f} & "
                    f"{s.get('api_preserved_rate', 0)*100:.1f} & "
                    f"{s.get('avg_complexity_delta', 0):+.1f} & "
                    f"{s.get('avg_judge_score', 0):.1f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


if __name__ == "__main__":
    main()
