#!/bin/bash
###############################################################################
# MASTER ORCHESTRATOR - Fire and Forget
#
# Runs ALL experiments autonomously. Designed to run inside tmux.
# Survives SSH disconnect. Checkpoint/resume on restart.
#
# Usage:
#   tmux new -s experiment
#   cd ~/Repo-AI/experiments
#   ./run_all_experiments.sh 2>&1 | tee ~/experiment_master.log
#   # Ctrl+B, D to detach. Close laptop.
###############################################################################
set -uo pipefail  # no -e: we handle errors per-experiment

# ── Setup ──
cd ~/Repo-AI
source ~/repoai-env/bin/activate
export $(grep -v '^#' .env | xargs)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS=~/Repo-AI/experiments/results/$TIMESTAMP
mkdir -p $RESULTS
SCRIPT_DIR=~/Repo-AI/experiments

echo "=============================================="
echo "  REPOAI EXPERIMENT PIPELINE"
echo "  Started: $(date)"
echo "  Results: $RESULTS"
echo "=============================================="

# Helper: run a benchmark with retry on total failure
run_bench() {
    local name="$1"
    shift
    echo ""
    echo "[$(date +%H:%M:%S)] ── $name ──"
    python3 $SCRIPT_DIR/run_benchmark.py "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] WARNING: $name exited with code $rc"
    fi
    return $rc
}

# ═══════════════════════════════════════════════
# PHASE 1: EXP-6 - RAG Comparison (run first)
# ═══════════════════════════════════════════════
echo ""
echo "====== EXP-6: RAG Pipeline Comparison ======"

for EMBED in rag_nemotron_embed rag_bge_embed; do
    for STRAT in top5 top10 mmr8; do
        run_bench "EXP-6 embed=$EMBED strat=$STRAT" \
            --experiment exp6_rag \
            --condition c4_full \
            --benchmark refactorbench \
            --tasks 20 \
            --output "$RESULTS/exp6/${EMBED}_${STRAT}/" \
            --checkpoint "$RESULTS/exp6/${EMBED}_${STRAT}/checkpoint.json" \
            --rag-config "${EMBED}_${STRAT}" &
    done
done
wait
echo "[$(date +%H:%M:%S)] EXP-6 complete."

# Pick best RAG config (highest average judge score)
BEST_RAG=$(python3 -c "
import json, os, glob
best_score, best_config = 0, 'rag_nemotron_embed_top5'
for d in glob.glob('$RESULTS/exp6/*/results.jsonl'):
    scores = []
    for line in open(d):
        r = json.loads(line)
        if r.get('success') and r.get('judge_score'):
            scores.append(r['judge_score'])
    avg = sum(scores)/len(scores) if scores else 0
    cfg = os.path.basename(os.path.dirname(d))
    if avg > best_score:
        best_score, best_config = avg, cfg
print(best_config)
" 2>/dev/null || echo "rag_nemotron_embed_top5")
echo "[$(date +%H:%M:%S)] Best RAG config: $BEST_RAG"
echo "$BEST_RAG" > "$RESULTS/best_rag_config.txt"


# ═══════════════════════════════════════════════
# PHASE 2: EXP-1 - Benchmark Evaluation (5 conds)
# ═══════════════════════════════════════════════
echo ""
echo "====== EXP-1: Benchmark-Scale Evaluation ======"

# RefactorBench: all 5 conditions in parallel
for COND in c1_direct c2_rag c3_multiagent c4_full c5_memory; do
    run_bench "EXP-1 RefactorBench $COND" \
        --experiment exp1 \
        --condition $COND \
        --benchmark refactorbench \
        --tasks 100 \
        --output "$RESULTS/exp1/refactorbench/$COND/" \
        --checkpoint "$RESULTS/exp1/refactorbench/$COND/checkpoint.json" \
        --rag-config "$BEST_RAG" &
done
wait
echo "[$(date +%H:%M:%S)] EXP-1 RefactorBench complete."

# SWE-bench Verified: conditions c1 and c4 only
for COND in c1_direct c4_full; do
    run_bench "EXP-1 SWE-bench $COND" \
        --experiment exp1 \
        --condition $COND \
        --benchmark swebench_verified \
        --tasks 50 \
        --output "$RESULTS/exp1/swebench/$COND/" \
        --checkpoint "$RESULTS/exp1/swebench/$COND/checkpoint.json" \
        --rag-config "$BEST_RAG" &
done
wait
echo "[$(date +%H:%M:%S)] EXP-1 SWE-bench complete."

# SWE-Refactor: conditions c1 and c4 only
for COND in c1_direct c4_full; do
    run_bench "EXP-1 SWE-Refactor $COND" \
        --experiment exp1 \
        --condition $COND \
        --benchmark swerefactor \
        --tasks 100 \
        --output "$RESULTS/exp1/swerefactor/$COND/" \
        --checkpoint "$RESULTS/exp1/swerefactor/$COND/checkpoint.json" \
        --rag-config "$BEST_RAG" &
done
wait
echo "[$(date +%H:%M:%S)] EXP-1 all benchmarks complete."


# ═══════════════════════════════════════════════
# PHASE 3: EXP-2 - Multi-Model Comparison
# ═══════════════════════════════════════════════
echo ""
echo "====== EXP-2: Multi-Model Comparison ======"

# NIM models: run in parallel (separate rate limits per model)
for MODEL_CFG in nim_qwen nim_deepseek nim_devstral; do
    run_bench "EXP-2 $MODEL_CFG" \
        --experiment exp2 \
        --condition $MODEL_CFG \
        --benchmark refactorbench \
        --tasks 100 \
        --output "$RESULTS/exp2/$MODEL_CFG/" \
        --checkpoint "$RESULTS/exp2/$MODEL_CFG/checkpoint.json" \
        --rag-config "$BEST_RAG" &
done
wait

# Gemini models: SKIPPED (free tier has 20 RPD limit)
# To enable: set up billing at https://aistudio.google.com with $50 student credit
# Then uncomment and re-run:
# run_bench "EXP-2 gemini_flash" \
#     --experiment exp2 \
#     --condition gemini_flash \
#     --benchmark refactorbench \
#     --tasks 100 \
#     --output "$RESULTS/exp2/gemini_flash/" \
#     --checkpoint "$RESULTS/exp2/gemini_flash/checkpoint.json" \
#     --rag-config "$BEST_RAG"
#
# run_bench "EXP-2 gemini_pro" \
#     --experiment exp2 \
#     --condition gemini_pro \
#     --benchmark refactorbench \
#     --tasks 50 \
#     --output "$RESULTS/exp2/gemini_pro/" \
#     --checkpoint "$RESULTS/exp2/gemini_pro/checkpoint.json" \
#     --rag-config "$BEST_RAG"
echo "[$(date +%H:%M:%S)] Gemini models skipped (free tier rate limit). Enable billing to run."

echo "[$(date +%H:%M:%S)] EXP-2 complete."


# ═══════════════════════════════════════════════
# PHASE 4: EXP-4 - Ablation Control
# ═══════════════════════════════════════════════
echo ""
echo "====== EXP-4: Ablation A5-ctrl ======"

run_bench "EXP-4 a5_ctrl" \
    --experiment exp4 \
    --condition a5_ctrl \
    --benchmark refactorbench \
    --tasks 100 \
    --output "$RESULTS/exp4/a5_ctrl/" \
    --checkpoint "$RESULTS/exp4/a5_ctrl/checkpoint.json" \
    --rag-config "$BEST_RAG"

echo "[$(date +%H:%M:%S)] EXP-4 complete."


# ═══════════════════════════════════════════════
# PHASE 5: Benchmark Evaluation (Docker)
# ═══════════════════════════════════════════════
echo ""
echo "====== Benchmark Test Evaluation ======"

# RefactorBench AST evaluation
echo "[$(date +%H:%M:%S)] Running RefactorBench AST evaluations..."
python3 $SCRIPT_DIR/eval_refactorbench.py \
    --results-dir "$RESULTS" \
    --output "$RESULTS/eval/refactorbench/" 2>&1 || echo "RefactorBench eval failed"

# SWE-bench Docker evaluation (the big one)
if [ -d "$RESULTS/exp1/swebench" ]; then
    echo "[$(date +%H:%M:%S)] Running SWE-bench Docker evaluations..."
    python3 $SCRIPT_DIR/eval_swebench.py \
        --results-dir "$RESULTS" \
        --output "$RESULTS/eval/swebench/" \
        --max-workers 4 2>&1 || echo "SWE-bench eval failed"
fi

echo "[$(date +%H:%M:%S)] Benchmark evaluations complete."


# ═══════════════════════════════════════════════
# PHASE 6: Analysis (EXP-3, EXP-7, Stats, Figures)
# ═══════════════════════════════════════════════
echo ""
echo "====== Analysis ======"

python3 $SCRIPT_DIR/analyze_results.py \
    --results-dir "$RESULTS" \
    --output "$RESULTS/analysis/" 2>&1 || echo "Analysis failed"

echo "[$(date +%H:%M:%S)] Analysis complete."


# ═══════════════════════════════════════════════
# PHASE 7: Push Results to GitHub
# ═══════════════════════════════════════════════
echo ""
echo "====== Pushing Results to GitHub ======"

cd ~/Repo-AI
git add experiments/results/
git commit -m "Add experiment results $TIMESTAMP

Experiments run: EXP-1 (benchmarks), EXP-2 (multi-model), EXP-4 (ablation), EXP-6 (RAG)
Best RAG config: $BEST_RAG" 2>/dev/null || echo "Nothing new to commit"

git push origin experiment 2>/dev/null || echo "Push failed -- check GITHUB_TOKEN"


# ═══════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════
echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Finished: $(date)"
echo "  Results:  $RESULTS"
echo "  Analysis: $RESULTS/analysis/"
echo "=============================================="
echo ""
echo "Summary:"
echo "  RefactorBench results: $(find $RESULTS/exp1/refactorbench -name results.jsonl 2>/dev/null | wc -l) conditions"
echo "  SWE-bench results:     $(find $RESULTS/exp1/swebench -name results.jsonl 2>/dev/null | wc -l) conditions"
echo "  Multi-model results:   $(find $RESULTS/exp2 -name results.jsonl 2>/dev/null | wc -l) models"
echo "  Total result files:    $(find $RESULTS -name results.jsonl | wc -l)"
echo ""
