# RepoAI Experiments

Experiment scripts, raw results, and analysis data for the paper:

> **RepoAI: Automated Code Refactoring through Multi-Agent LLM Orchestration and Retrieval-Augmented Generation**
>
> Submitted to *Science of Computer Programming*

## Overview

This repository contains the complete experimental pipeline and reproducible results for evaluating the RepoAI multi-agent code refactoring system. The evaluation comprises **1,404 task attempts** across **7 experiments**, testing **7 LLM models** from **4 families** under **13 experimental conditions**.

## Repository Structure

```
.
├── scripts/                    # Experiment execution scripts
│   ├── config.py              # Model configuration (NIM API abstraction)
│   ├── run_benchmark.py       # Main benchmark runner with checkpoint/resume
│   ├── analyze_results.py     # Statistical analysis pipeline
│   ├── eval_refactorbench.py  # RefactorBench evaluation (cross-family judging)
│   ├── eval_swebench.py       # SWE-bench evaluation
│   ├── run_all_experiments.sh # Master orchestrator script
│   ├── setup_vm.sh            # GCE VM setup script
│   └── agents/
│       └── memory.py          # Architectural Memory Module (C5 condition)
├── results/
│   ├── analysis/              # Generated analysis artifacts
│   │   ├── summary.json       # Top-level experiment summary
│   │   ├── exp1_statistics.json   # Friedman & Wilcoxon tests
│   │   ├── exp2_model_summary.json # Multi-model comparison
│   │   ├── exp3_judge_validation.json # Cross-family judge calibration
│   │   ├── exp6_rag_summary.json  # RAG configuration comparison
│   │   ├── exp7_error_taxonomy.json # Error classification
│   │   └── tables.tex         # LaTeX tables for paper
│   ├── exp1/                  # EXP-1: Pipeline ablation (C1-C5)
│   │   ├── refactorbench/     # 100 tasks × 5 conditions
│   │   └── swebench/          # 50 tasks × 2 conditions
│   ├── exp2/                  # EXP-2: Multi-model comparison (7 models)
│   ├── exp4/                  # EXP-4: Single-agent control (A5-ctrl)
│   ├── exp6/                  # EXP-6: RAG configuration (6 configs)
│   └── eval/                  # Processed evaluation results
│       ├── refactorbench/
│       └── swebench/
├── .env.template              # Environment variable template
└── README.md
```

## Experiments

| ID | Research Question | Description | Tasks |
|----|------------------|-------------|-------|
| EXP-1 | RQ1: Pipeline Ablation | Cumulative component comparison (C1-C5) | 500 (RefactorBench) + 100 (SWE-bench) |
| EXP-2 | RQ2: Model Generalisability | 7 models from 4 families | 684 |
| EXP-3 | RQ3: Judge Validity | Cross-family judge calibration analysis | Derived from EXP-1/2 |
| EXP-4 | RQ4: Multi-Agent Necessity | Single-agent self-refinement control | 100 |
| EXP-6 | RQ5: RAG Configuration | 2 embeddings × 3 retrieval strategies | 120 |
| EXP-7 | RQ6: Failure Modes | Error taxonomy across all experiments | Derived |

## Pipeline Conditions

| Condition | Components |
|-----------|-----------|
| C1 (Direct) | Single LLM call, no retrieval or coordination |
| C2 (RAG) | + BGE-M3 embeddings with MMR retrieval |
| C3 (Multi-Agent) | + Planner/Coder/Reviewer agent coordination |
| C4 (Full) | + Validation pipeline and automated PR generation |
| C5 (Memory) | + Architectural memory module for cross-file context |
| A5-ctrl | Single agent, 3-round self-refinement (same token budget as C3) |

## Models Evaluated

| Model | Family | Parameters | Provider |
|-------|--------|-----------|----------|
| Qwen 2.5 Coder 32B Instruct | Qwen | 32B | NVIDIA NIM |
| Gemma 2 9B IT | Google | 9B | NVIDIA NIM |
| Devstral 2 123B Instruct | Mistral | 123B | NVIDIA NIM |
| Llama 4 Maverick 17B 128E | Meta | 17B (MoE) | NVIDIA NIM |
| Llama 3.3 70B Instruct | Meta | 70B | NVIDIA NIM |
| DeepSeek R1 Distill Qwen 32B | DeepSeek | 32B | NVIDIA NIM |
| Llama 4 Scout 17B 16E | Meta | 17B (MoE) | NVIDIA NIM |

## Key Results

- **Friedman test**: χ² = 11.08, p = 0.026 (significant difference across conditions)
- **C4 Full Pipeline**: Mean score 96.8 vs C1 Direct 93.2 (Wilcoxon p = 0.007)
- **Best model**: Gemma 2 9B (mean 96.1) — smallest model, highest score
- **All 170 failures**: Category E9 (format errors), model-specific not pipeline-specific
- **Best RAG config**: BGE-M3 + MMR (k=8), mean 97.0

## Cross-Family Judging Protocol

To eliminate self-evaluation bias:
- Qwen/DeepSeek/Gemma/Devstral outputs → judged by **Llama 3.3 70B**
- Llama family outputs → judged by **Qwen 2.5 Coder 32B**

## Reproduction

1. Copy `.env.template` to `.env` and fill in API keys
2. Set up a GCE VM: `bash scripts/setup_vm.sh`
3. Run all experiments: `bash scripts/run_all_experiments.sh`
4. Analyse results: `python scripts/analyze_results.py`

## License

This repository is provided for academic research purposes. Please cite the paper if you use this data or code.
