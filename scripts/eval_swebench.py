"""
SWE-bench Docker Evaluation Wrapper.
Converts RepoAI experiment outputs into SWE-bench prediction format,
then runs the official SWE-bench Docker evaluation harness.

Usage:
    python eval_swebench.py \
        --results-dir results/20260313_120000 \
        --output results/20260313_120000/eval/swebench/ \
        --max-workers 4
"""

import argparse
import json
import os
import subprocess
import sys
import logging
import tempfile
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def results_to_predictions(results_path: str) -> list:
    """
    Convert JSONL results to SWE-bench prediction format.
    Each prediction = {"instance_id": ..., "model_patch": ..., "model_name_or_path": ...}
    """
    predictions = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not result.get("success") or not result.get("generated_code"):
                continue

            instance_id = result.get("task_id", "")
            if not instance_id:
                continue

            # Build a unified diff/patch from the generated code
            patch = result.get("patch", "")
            if not patch and result.get("generated_code"):
                # If no explicit patch, create one from the generated code
                # SWE-bench expects a git-style patch
                patch = _make_patch(result)

            if patch:
                predictions.append({
                    "instance_id": instance_id,
                    "model_patch": patch,
                    "model_name_or_path": result.get("condition", "repoai"),
                })

    return predictions


def _make_patch(result: dict) -> str:
    """
    Build a minimal git-format patch from generated code.
    Falls back to a simple replacement patch if structured data is available.
    """
    generated = result.get("generated_code", "")
    original = result.get("original_code", "")
    file_path = result.get("file_path", "unknown.py")

    if not generated:
        return ""

    if original:
        # Build a simple unified diff
        import difflib
        orig_lines = original.splitlines(keepends=True)
        gen_lines = generated.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines, gen_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
        return "".join(diff)

    # If no original, wrap as a full-file replacement patch
    lines = generated.splitlines()
    patch_lines = [
        f"--- a/{file_path}",
        f"+++ b/{file_path}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    for line in lines:
        patch_lines.append(f"+{line}")
    return "\n".join(patch_lines) + "\n"


def write_predictions_file(predictions: list, output_path: str):
    """Write predictions in SWE-bench JSONL format."""
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    logger.info(f"Wrote {len(predictions)} predictions to {output_path}")


def run_swebench_eval(predictions_path: str, output_dir: str,
                      dataset: str = "princeton-nlp/SWE-bench_Verified",
                      max_workers: int = 4,
                      timeout: int = 1800) -> dict:
    """
    Run the official SWE-bench evaluation harness via Docker.
    Returns the evaluation results dict.
    """
    # Check if swebench is installed
    try:
        import swebench
        swebench_available = True
    except ImportError:
        swebench_available = False

    if not swebench_available:
        logger.warning("SWE-bench not installed. Attempting CLI evaluation...")

    # Method 1: Use swebench CLI (preferred)
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--predictions_path", predictions_path,
        "--swe_bench_tasks", dataset,
        "--log_dir", os.path.join(output_dir, "logs"),
        "--testbed", os.path.join(output_dir, "testbed"),
        "--max_workers", str(max_workers),
        "--timeout", str(timeout),
    ]

    logger.info(f"Running SWE-bench evaluation: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout * max_workers + 600,  # generous timeout
        )

        if proc.returncode != 0:
            logger.error(f"SWE-bench eval failed (rc={proc.returncode})")
            logger.error(f"stderr: {proc.stderr[:2000]}")
            return {"status": "error", "error": proc.stderr[:2000]}

        logger.info("SWE-bench evaluation completed successfully")

        # Parse results
        return _parse_eval_results(output_dir, predictions_path)

    except subprocess.TimeoutExpired:
        logger.error("SWE-bench evaluation timed out")
        return {"status": "timeout"}
    except FileNotFoundError:
        logger.warning("swebench CLI not found. Trying Docker-based evaluation...")
        return _run_docker_eval(predictions_path, output_dir, dataset, max_workers, timeout)


def _run_docker_eval(predictions_path: str, output_dir: str,
                     dataset: str, max_workers: int, timeout: int) -> dict:
    """Fallback: Run SWE-bench eval using Docker directly."""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.path.abspath(predictions_path)}:/predictions.jsonl:ro",
        "-v", f"{os.path.abspath(output_dir)}:/output",
        "aorwall/swe-bench-docker:latest",
        "--predictions_path", "/predictions.jsonl",
        "--swe_bench_tasks", dataset,
        "--log_dir", "/output/logs",
        "--max_workers", str(max_workers),
    ]

    logger.info(f"Running Docker-based SWE-bench evaluation")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout * max_workers + 600,
        )

        if proc.returncode != 0:
            logger.error(f"Docker eval failed: {proc.stderr[:2000]}")
            return {"status": "error", "error": proc.stderr[:2000]}

        return _parse_eval_results(output_dir, predictions_path)

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Docker evaluation failed: {e}")
        return {"status": "error", "error": str(e)}


def _parse_eval_results(output_dir: str, predictions_path: str) -> dict:
    """Parse SWE-bench evaluation output files into a summary dict."""
    results = {
        "status": "complete",
        "total": 0,
        "resolved": 0,
        "failed": 0,
        "error_instances": 0,
        "resolve_rate": 0.0,
        "resolved_ids": [],
        "failed_ids": [],
    }

    # Look for the evaluation report
    log_dir = Path(output_dir) / "logs"
    report_files = list(log_dir.glob("**/report.json")) if log_dir.exists() else []

    if report_files:
        with open(report_files[0]) as f:
            report = json.load(f)

        resolved = report.get("resolved", [])
        failed = report.get("failed", []) + report.get("unresolved", [])
        errors = report.get("error", [])

        results["resolved"] = len(resolved)
        results["failed"] = len(failed)
        results["error_instances"] = len(errors)
        results["total"] = len(resolved) + len(failed) + len(errors)
        results["resolve_rate"] = len(resolved) / results["total"] if results["total"] else 0
        results["resolved_ids"] = resolved
        results["failed_ids"] = failed
    else:
        # Try to count from log directories
        if log_dir.exists():
            instance_dirs = [d for d in log_dir.iterdir() if d.is_dir()]
            for inst_dir in instance_dirs:
                results["total"] += 1
                # Check for test results
                test_output = inst_dir / "test_output.txt"
                if test_output.exists():
                    content = test_output.read_text()
                    if "PASSED" in content or "passed" in content.lower():
                        results["resolved"] += 1
                        results["resolved_ids"].append(inst_dir.name)
                    else:
                        results["failed"] += 1
                        results["failed_ids"].append(inst_dir.name)
                else:
                    results["error_instances"] += 1

            if results["total"]:
                results["resolve_rate"] = results["resolved"] / results["total"]

    return results


def main():
    parser = argparse.ArgumentParser(description="SWE-bench Docker Evaluation")
    parser.add_argument("--results-dir", required=True,
                        help="Root results directory")
    parser.add_argument("--output", required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Max parallel Docker workers")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Timeout per instance in seconds")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning for SWE-bench results in {results_dir}")

    # Find all SWE-bench result files
    swebench_results = list(results_dir.glob("**/swebench/**/results.jsonl"))
    # Also SWE-Refactor
    swerefactor_results = list(results_dir.glob("**/swerefactor/**/results.jsonl"))

    all_eval_results = {}

    # Process SWE-bench Verified
    for rf in sorted(swebench_results):
        rel = rf.relative_to(results_dir)
        label = str(rel.parent).replace(os.sep, "/")
        logger.info(f"Processing SWE-bench results: {label}")

        predictions = results_to_predictions(str(rf))
        if not predictions:
            logger.warning(f"No valid predictions from {label}")
            continue

        cond_dir = output_dir / label
        cond_dir.mkdir(parents=True, exist_ok=True)

        pred_path = str(cond_dir / "predictions.jsonl")
        write_predictions_file(predictions, pred_path)

        eval_results = run_swebench_eval(
            pred_path,
            str(cond_dir),
            dataset="princeton-nlp/SWE-bench_Verified",
            max_workers=args.max_workers,
            timeout=args.timeout,
        )
        all_eval_results[label] = eval_results

        # Save per-condition results
        with open(cond_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"  {label}: {eval_results.get('resolve_rate', 0):.1%} resolved "
                     f"({eval_results.get('resolved', 0)}/{eval_results.get('total', 0)})")

    # Process SWE-Refactor (if present)
    for rf in sorted(swerefactor_results):
        rel = rf.relative_to(results_dir)
        label = str(rel.parent).replace(os.sep, "/")
        logger.info(f"Processing SWE-Refactor results: {label}")

        predictions = results_to_predictions(str(rf))
        if not predictions:
            logger.warning(f"No valid predictions from {label}")
            continue

        cond_dir = output_dir / label
        cond_dir.mkdir(parents=True, exist_ok=True)

        pred_path = str(cond_dir / "predictions.jsonl")
        write_predictions_file(predictions, pred_path)

        # SWE-Refactor uses a different dataset
        eval_results = run_swebench_eval(
            pred_path,
            str(cond_dir),
            dataset="swerefactor/SWE-Refactor",
            max_workers=args.max_workers,
            timeout=args.timeout,
        )
        all_eval_results[label] = eval_results

        with open(cond_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

    # Write summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "conditions": all_eval_results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Write LaTeX table
    _write_latex_table(all_eval_results, output_dir / "table_swebench.tex")

    logger.info(f"SWE-bench evaluation complete. Summary: {summary_path}")


def _write_latex_table(results: dict, path: Path):
    """Write LaTeX table for SWE-bench results."""
    with open(path, "w") as f:
        f.write("% Auto-generated by eval_swebench.py\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{SWE-bench Execution-Based Evaluation}\n")
        f.write("\\label{tab:swebench-eval}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Condition & Total & Resolved & Resolve\\% \\\\\n")
        f.write("\\midrule\n")

        for label, r in sorted(results.items()):
            cond = label.split("/")[-1] if "/" in label else label
            f.write(f"{cond} & "
                    f"{r.get('total', 0)} & "
                    f"{r.get('resolved', 0)} & "
                    f"{r.get('resolve_rate', 0)*100:.1f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


if __name__ == "__main__":
    main()
