"""
Benchmark runner with checkpoint/resume and retry logic.
Runs RepoAI pipeline on benchmark tasks and logs results as JSONL.

Usage:
    python run_benchmark.py \
        --experiment exp1 \
        --condition c4_full \
        --benchmark refactorbench \
        --tasks 100 \
        --output results/exp1/c4_full/ \
        --checkpoint results/exp1/c4_full/checkpoint.json
"""

import argparse
import json
import os
import sys
import time
import logging
import signal
from pathlib import Path
from datetime import datetime

# Add script directory and parent directory to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/Repo-AI/.env"))

from config import ModelConfig, CONFIGS, get_llm, get_embeddings, TokenTracker, timed_llm_call
# Import memory module directly to avoid conflict with Repo-AI's agents/ package
import importlib.util
_mem_spec = importlib.util.spec_from_file_location(
    "exp_memory", os.path.join(_SCRIPT_DIR, "agents", "memory.py"))
_mem_mod = importlib.util.module_from_spec(_mem_spec)
_mem_spec.loader.exec_module(_mem_mod)
ArchitecturalMemory = _mem_mod.ArchitecturalMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Graceful shutdown
_shutdown = False
def _handle_signal(signum, frame):
    global _shutdown
    logger.warning("Shutdown signal received. Will finish current task and exit.")
    _shutdown = True
signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# --- Checkpoint Management ---

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(path: str, checkpoint: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)

def append_result(output_dir: str, result: dict):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


# --- Benchmark Task Loading ---

def load_benchmark_tasks(benchmark: str, max_tasks: int = None) -> list:
    """Load tasks from a benchmark. Returns list of dicts with task_id, instruction, repo_path, etc."""
    if benchmark == "refactorbench":
        return _load_refactorbench_tasks(max_tasks)
    elif benchmark == "swebench_verified":
        return _load_swebench_tasks(max_tasks)
    elif benchmark == "swerefactor":
        return _load_swerefactor_tasks(max_tasks)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def _load_refactorbench_tasks(max_tasks=None):
    """Load RefactorBench tasks from ~/RefactorBench.

    Structure: problems/{base,descriptive,lazy}_problems/{repo}_refactor/*-task.txt
    We use base_problems (medium specificity) as the default.
    Each task has a matching test in tests/{repo}_refactor/ and repo in repositories/{repo}_refactor/.
    """
    rb_dir = os.path.expanduser("~/RefactorBench")
    tasks = []

    # Use base_problems (medium specificity) — 100 tasks total
    problems_dir = Path(rb_dir) / "problems" / "base_problems"
    if not problems_dir.exists():
        logger.error(f"RefactorBench problems dir not found: {problems_dir}")
        return []

    for repo_dir in sorted(problems_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        repo_name = repo_dir.name  # e.g., "django_refactor"

        for task_file in sorted(repo_dir.glob("*-task.txt")):
            if max_tasks and len(tasks) >= max_tasks:
                break

            instruction = task_file.read_text(encoding="utf-8", errors="replace").strip()
            task_name = task_file.stem.replace("-task", "")
            task_id = f"rb_{repo_name}_{task_name}"

            # Find matching test file
            test_file = Path(rb_dir) / "tests" / repo_name / f"{task_name}-test.py"
            # Some test files may have slightly different naming
            if not test_file.exists():
                test_candidates = list((Path(rb_dir) / "tests" / repo_name).glob(f"{task_name}*"))
                test_file = test_candidates[0] if test_candidates else None

            # Repository path
            repo_path = str(Path(rb_dir) / "repositories" / repo_name)

            tasks.append({
                "task_id": task_id,
                "benchmark": "refactorbench",
                "instruction": instruction,
                "repo_name": repo_name,
                "repo_path": repo_path,
                "task_file": str(task_file),
                "test_file": str(test_file) if test_file else "",
            })

        if max_tasks and len(tasks) >= max_tasks:
            break

    logger.info(f"Loaded {len(tasks)} RefactorBench tasks")
    return tasks


def _load_swebench_tasks(max_tasks=None):
    """Load SWE-bench Verified tasks from HuggingFace dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    except Exception:
        # Fallback: try loading from local file
        ds_path = os.path.expanduser("~/SWE-bench/data/swe-bench-verified.json")
        if os.path.exists(ds_path):
            with open(ds_path) as f:
                ds = json.load(f)
        else:
            logger.error("Cannot load SWE-bench Verified. Install 'datasets' or download manually.")
            return []

    tasks = []
    for item in ds:
        if max_tasks and len(tasks) >= max_tasks:
            break
        tasks.append({
            "task_id": item["instance_id"],
            "benchmark": "swebench_verified",
            "instruction": item["problem_statement"],
            "repo": item.get("repo", ""),
            "base_commit": item.get("base_commit", ""),
            "hints_text": item.get("hints_text", ""),
        })

    logger.info(f"Loaded {len(tasks)} SWE-bench Verified tasks")
    return tasks


def _load_swerefactor_tasks(max_tasks=None):
    """Load SWE-Refactor tasks. Placeholder -- adjust path based on actual dataset structure."""
    sr_dir = os.path.expanduser("~/SWE-Refactor")
    tasks = []

    data_file = os.path.join(sr_dir, "data", "tasks.json")
    if not os.path.exists(data_file):
        # Try alternative locations
        for candidate in [
            os.path.join(sr_dir, "tasks.json"),
            os.path.join(sr_dir, "dataset.json"),
            os.path.join(sr_dir, "data.json"),
        ]:
            if os.path.exists(candidate):
                data_file = candidate
                break

    if not os.path.exists(data_file):
        logger.warning(f"SWE-Refactor data not found at {sr_dir}. Skipping.")
        return []

    with open(data_file) as f:
        data = json.load(f)

    for item in (data if isinstance(data, list) else data.values()):
        if max_tasks and len(tasks) >= max_tasks:
            break
        tasks.append({
            "task_id": item.get("id", item.get("instance_id", f"sr_{len(tasks)}")),
            "benchmark": "swerefactor",
            "instruction": item.get("instruction", item.get("description", "")),
            "repo": item.get("repo", ""),
            "refactoring_type": item.get("type", ""),
        })

    logger.info(f"Loaded {len(tasks)} SWE-Refactor tasks")
    return tasks


# --- Pipeline Execution ---

def run_pipeline(task: dict, condition: str, config: ModelConfig,
                 rag_config: str = None) -> dict:
    """
    Run the RepoAI pipeline on a single task under a given condition.
    Returns a result dict with all metrics.
    """
    tracker = TokenTracker(task["task_id"], "exp", condition)
    start_time = time.time()

    result = {
        "task_id": task["task_id"],
        "benchmark": task.get("benchmark", ""),
        "condition": condition,
        "generator_model": config.generator_model,
        "generator_provider": config.generator_provider,
        "judge_model": config.judge_model,
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        # Step 1: Get the instruction
        instruction = task["instruction"]

        if condition == "c1_direct":
            # Direct LLM call, no RAG, no multi-agent, no validation
            generated_code = _run_direct(instruction, config, tracker)

        elif condition == "c2_rag":
            # Single LLM + RAG context
            generated_code = _run_with_rag(instruction, task, config, tracker, rag_config)

        elif condition == "c3_multiagent":
            # Multi-agent + RAG, no validation
            generated_code = _run_multiagent(instruction, task, config, tracker,
                                             rag_config, validate=False)

        elif condition in ("c4_full", "nim_qwen", "nim_deepseek", "nim_devstral",
                           "nim_llama33", "nim_llama4_maverick", "nim_gemma2_9b",
                           "gemini_flash", "gemini_pro"):
            # Full pipeline: RAG + multi-agent + validation
            generated_code = _run_multiagent(instruction, task, config, tracker,
                                             rag_config, validate=True)

        elif condition == "c5_memory":
            # Full pipeline + persistent memory
            generated_code = _run_multiagent(instruction, task, config, tracker,
                                             rag_config, validate=True, use_memory=True)

        elif condition == "a5_ctrl":
            # Single agent, same token budget as multi-agent (self-refine)
            generated_code = _run_self_refine(instruction, task, config, tracker, rag_config)

        else:
            raise ValueError(f"Unknown condition: {condition}")

        # Step 2: Judge the output (cross-model)
        judge_score = _run_judge(instruction, generated_code, config, tracker)

        result.update({
            "success": True,
            "generated_code": generated_code[:5000],  # truncate for storage
            "judge_score": judge_score,
            "tokens": tracker.to_dict(),
            "wall_time_ms": (time.time() - start_time) * 1000,
        })

        # Learn from result (C5 memory condition)
        if condition == "c5_memory":
            mem = _get_memory()
            mem.learn_from_result(
                task_id=task["task_id"],
                task_description=instruction,
                original_code=task.get("original_code", ""),
                generated_code=generated_code[:2000],
                judge_score=judge_score,
                language="python",
                repo_name=task.get("repo", ""),
            )

    except Exception as e:
        logger.error(f"Task {task['task_id']} failed: {e}")
        result.update({
            "success": False,
            "error": str(e),
            "tokens": tracker.to_dict(),
            "wall_time_ms": (time.time() - start_time) * 1000,
        })

    return result


def _run_direct(instruction: str, config: ModelConfig, tracker: TokenTracker) -> str:
    """C1: Direct single LLM call."""
    llm = get_llm(config, role="generator")
    prompt = f"Refactor the code as described below. Output only the code.\n\n{instruction}"
    return timed_llm_call(llm, prompt, tracker, "direct_generation",
                          config.generator_model, config.generator_provider)


def _run_with_rag(instruction: str, task: dict, config: ModelConfig,
                  tracker: TokenTracker, rag_config: str = None) -> str:
    """C2: Single LLM + RAG context."""
    # Retrieve context
    context = _retrieve_context(instruction, task, config, tracker)

    llm = get_llm(config, role="generator")
    prompt = (
        f"You are a code refactoring assistant. Use the following repository context "
        f"to inform your changes.\n\n"
        f"## Repository Context\n{context}\n\n"
        f"## Task\n{instruction}\n\n"
        f"Output only the modified code."
    )
    return timed_llm_call(llm, prompt, tracker, "rag_generation",
                          config.generator_model, config.generator_provider)


def _run_multiagent(instruction: str, task: dict, config: ModelConfig,
                    tracker: TokenTracker, rag_config: str = None,
                    validate: bool = True, use_memory: bool = False) -> str:
    """C3/C4/C5: Multi-agent pipeline."""
    # Step 1: Route (classify intent)
    router_llm = get_llm(config, role="router")
    route_prompt = (
        f"Classify the following request as either REPO_OVERVIEW or REFACTOR_FILE. "
        f"Reply with just the classification.\n\n{instruction}"
    )
    intent = timed_llm_call(router_llm, route_prompt, tracker, "routing",
                            config.router_model, config.router_provider)

    # Step 2: Retrieve context
    context = _retrieve_context(instruction, task, config, tracker)

    # Step 3: Plan
    planner_llm = get_llm(config, role="generator")
    plan_prompt = (
        f"Given this refactoring request and repository context, plan what files need "
        f"to be created, modified, or deleted. Output a JSON list of operations.\n\n"
        f"## Context\n{context[:4000]}\n\n"
        f"## Request\n{instruction}\n\n"
        f"Output format: [{{'action': 'create|modify|delete', 'path': '...', 'description': '...'}}]"
    )
    plan = timed_llm_call(planner_llm, plan_prompt, tracker, "planning",
                          config.generator_model, config.generator_provider)

    # Step 4: Generate code
    gen_llm = get_llm(config, role="generator")
    memory_context = ""
    if use_memory:
        mem = _get_memory()
        memory_context = mem.build_context(
            instruction, language="python",
            repo_name=task.get("repo", ""),
        )

    gen_prompt = (
        f"Generate the refactored code based on this plan and context.\n\n"
        f"## Plan\n{plan[:3000]}\n\n"
        f"## Repository Context\n{context[:4000]}\n\n"
        f"## Request\n{instruction}\n"
        f"{memory_context}\n"
        f"Output the complete modified files with clear file path headers."
    )
    generated = timed_llm_call(gen_llm, gen_prompt, tracker, "generation",
                               config.generator_model, config.generator_provider)

    # Step 5: Validate (if enabled)
    if validate:
        val_llm = get_llm(config, role="judge")
        val_prompt = (
            f"Check the following generated code for syntax errors, missing imports, "
            f"and inconsistencies. List any issues found.\n\n"
            f"## Generated Code\n{generated[:6000]}\n\n"
            f"Reply with 'PASS' if no critical issues, or list the issues."
        )
        validation = timed_llm_call(val_llm, val_prompt, tracker, "validation",
                                    config.judge_model, config.judge_provider)

        # If validation found issues and memory is enabled, do reconciliation
        if use_memory and "PASS" not in validation.upper():
            reconcile_prompt = (
                f"The following code was generated but has issues:\n{validation[:2000]}\n\n"
                f"## Original Code\n{generated[:6000]}\n\n"
                f"Fix the issues and output the corrected code."
            )
            generated = timed_llm_call(gen_llm, reconcile_prompt, tracker,
                                       "reconciliation",
                                       config.generator_model, config.generator_provider)

    return generated


def _run_self_refine(instruction: str, task: dict, config: ModelConfig,
                     tracker: TokenTracker, rag_config: str = None) -> str:
    """A5-ctrl: Single agent with self-refinement (same token budget as multi-agent)."""
    context = _retrieve_context(instruction, task, config, tracker)

    llm = get_llm(config, role="generator")

    # Round 1: Generate
    gen_prompt = (
        f"Refactor the code as described. Use the context below.\n\n"
        f"## Context\n{context[:4000]}\n\n"
        f"## Task\n{instruction}\n\n"
        f"Output the complete modified code."
    )
    generated = timed_llm_call(llm, gen_prompt, tracker, "selfrefine_gen1",
                               config.generator_model, config.generator_provider)

    # Round 2: Critique
    critique_prompt = (
        f"Review the following code for correctness, completeness, and issues:\n\n"
        f"{generated[:6000]}\n\n"
        f"Original request: {instruction}\n\n"
        f"List all issues found."
    )
    critique = timed_llm_call(llm, critique_prompt, tracker, "selfrefine_critique",
                              config.generator_model, config.generator_provider)

    # Round 3: Refine based on critique
    refine_prompt = (
        f"Fix the following issues in the code:\n{critique[:2000]}\n\n"
        f"## Code to fix\n{generated[:6000]}\n\n"
        f"Output the corrected complete code."
    )
    refined = timed_llm_call(llm, refine_prompt, tracker, "selfrefine_gen2",
                             config.generator_model, config.generator_provider)

    return refined


def _retrieve_context(instruction: str, task: dict, config: ModelConfig,
                      tracker: TokenTracker) -> str:
    """Retrieve relevant code context using embeddings + FAISS."""
    # For now, return task-provided context or empty string
    # Full RAG integration will be done when modifying the actual RepoAI code
    context_parts = []
    for key in ["repo_content", "context", "hints_text"]:
        if key in task and task[key]:
            context_parts.append(str(task[key])[:4000])
    return "\n".join(context_parts) if context_parts else "(no context available)"


def _run_judge(instruction: str, generated_code: str,
               config: ModelConfig, tracker: TokenTracker) -> float:
    """Cross-model intent verification. Returns score 0-100."""
    judge_llm = get_llm(config, role="judge")
    prompt = (
        f"Rate how well the following code fulfills the user's request on a scale of 0-100.\n\n"
        f"## User Request\n{instruction[:2000]}\n\n"
        f"## Generated Code\n{generated_code[:5000]}\n\n"
        f"Reply with ONLY a JSON object: {{\"score\": <0-100>, \"reason\": \"...\"}}"
    )
    response = timed_llm_call(judge_llm, prompt, tracker, "judging",
                              config.judge_model, config.judge_provider)

    # Parse score
    try:
        # Try JSON parse
        data = json.loads(response)
        return float(data.get("score", 0))
    except (json.JSONDecodeError, ValueError):
        # Fallback: find a number in the response
        import re
        numbers = re.findall(r'\b(\d{1,3})\b', response)
        for n in numbers:
            val = int(n)
            if 0 <= val <= 100:
                return float(val)
        return 0.0


# --- Memory Singleton ---

_memory_instance = None

def _get_memory() -> ArchitecturalMemory:
    """Get or create the shared ArchitecturalMemory instance."""
    global _memory_instance
    if _memory_instance is None:
        mem_dir = os.path.expanduser("~/Repo-AI/experiments/memory_store")
        _memory_instance = ArchitecturalMemory(mem_dir)
    return _memory_instance


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="RepoAI Benchmark Runner")
    parser.add_argument("--experiment", required=True, help="Experiment name (exp1, exp2, etc.)")
    parser.add_argument("--condition", default="c4_full", help="Condition ID (c1_direct, c4_full, etc.)")
    parser.add_argument("--generator", default=None, help="Override generator as 'provider:model'")
    parser.add_argument("--benchmark", required=True, help="Benchmark name")
    parser.add_argument("--tasks", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file path")
    parser.add_argument("--rag-config", default=None, help="RAG configuration name")
    args = parser.parse_args()

    # Load config
    config = CONFIGS.get(args.condition, ModelConfig())

    # Override generator if specified
    if args.generator:
        provider, model = args.generator.split(":", 1)
        config.generator_model = model
        config.generator_provider = provider

    # Ensure API keys are loaded
    config.__post_init__()

    # Load tasks
    tasks = load_benchmark_tasks(args.benchmark, args.tasks)
    if not tasks:
        logger.error(f"No tasks loaded for benchmark {args.benchmark}")
        sys.exit(1)

    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint)

    # Run tasks
    completed = sum(1 for v in checkpoint.values() if v in ("done", "failed"))
    total = len(tasks)
    logger.info(f"Starting {args.experiment}/{args.condition}: "
                f"{completed}/{total} already done, {total - completed} remaining")

    for i, task in enumerate(tasks):
        if _shutdown:
            logger.info("Shutting down gracefully...")
            break

        if task["task_id"] in checkpoint:
            continue

        logger.info(f"[{completed+1}/{total}] Running task {task['task_id']}...")

        result = run_pipeline(task, args.condition, config, args.rag_config)
        append_result(args.output, result)

        status = "done" if result.get("success") else "failed"
        checkpoint[task["task_id"]] = status
        save_checkpoint(args.checkpoint, checkpoint)

        completed += 1

        # Brief pause between tasks to respect rate limits
        time.sleep(1)

    # Summary
    done = sum(1 for v in checkpoint.values() if v == "done")
    failed = sum(1 for v in checkpoint.values() if v == "failed")
    logger.info(f"Finished: {done} done, {failed} failed, {total - done - failed} remaining")


if __name__ == "__main__":
    main()
