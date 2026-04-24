#!/usr/bin/env bash
# Reproduce every BrowserART, OS-Harm, RedCode-Gen, and BigCodeBench result
# from the ICML paper "Architecture Matters for Multi-Agent Security",
# end-to-end, locally. Each runner is self-contained — no orbit, no inspectmas.
#
# Usage:
#   ./run_all.sh                            # defaults: gpt-4o, seed 42
#   ./run_all.sh openai/gpt-4o              # pick model
#   ./run_all.sh openai/gpt-4o 42           # pick model + seed
#   MAX_SAMPLES=8 ./run_all.sh              # per-condition Inspect parallelism
#   PARALLEL=3 ./run_all.sh                 # condition-level concurrency (be careful with RAM)
#   ONLY=osharm_harmful ./run_all.sh        # one bucket only
#
# Six buckets, one per (scenario, kind):
#   browserart_harmful  — 13 conditions x 100 BrowserART tasks
#   browserart_benign   — 13 conditions x 42  benign WebArena tasks
#   osharm_harmful      — 13 conditions x 44  OS-Harm misuse tasks
#   osharm_benign       — 13 conditions x 50  OSWorld benign tasks
#   redcode_harmful     — 13 conditions x 160 RedCode-Gen malware tasks
#   redcode_benign      — 13 conditions x 50  BigCodeBench tasks (paper default)
#
# Environment variables:
#   MODEL        default: openai/gpt-4o
#   SEED         default: 42
#   PARALLEL     default: 1   (concurrent conditions per bucket; warning: RAM)
#   MAX_SAMPLES  default: 4   (parallel samples within a single condition run)
#   BCB_LIMIT    default: 50  (BigCodeBench sample count)
#   ONLY         default: ""  (run all); one of the bucket names above
#   EXTRA_ARGS   extra args forwarded to every runner

set -euo pipefail

MODEL="${1:-${MODEL:-openai/gpt-4o}}"
SEED="${2:-${SEED:-42}}"
PARALLEL="${PARALLEL:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
BCB_LIMIT="${BCB_LIMIT:-50}"
ONLY="${ONLY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# All 13 conditions in paper-table order (Tables 8-10).
CONDITIONS=(
    single_agent
    star_batch_relaxed
    star_batch
    star_step
    star_2_specialist
    star_3_specialist
    star_specialist
    mesh_round_robin
    mesh_delegation
    memory_own_reasoning
    memory_full
    mesh_delegation_cot
    mesh_delegation_full
)

# shellcheck disable=SC2206
EXTRA_ARR=($EXTRA_ARGS)

run_condition() {
    local script="$1"
    local extra_args=("${@:2}")
    for cond in "${CONDITIONS[@]}"; do
        echo "  -> $script  --condition $cond"
        python -u "$script" \
            --model "$MODEL" \
            --seed "$SEED" \
            --condition "$cond" \
            --max-samples "$MAX_SAMPLES" \
            "${extra_args[@]}" \
            "${EXTRA_ARR[@]}"
    done
}

run_bucket() {
    local bucket="$1"
    shift
    if [[ -n "$ONLY" && "$ONLY" != "$bucket" ]]; then
        return 0
    fi
    echo ""
    echo "=================================================================="
    echo "  [$bucket]  model=$MODEL  seed=$SEED  max_samples=$MAX_SAMPLES"
    echo "=================================================================="
    "$@"
}

run_bucket browserart_harmful run_condition browserart.py   --kind harmful
run_bucket browserart_benign  run_condition browserart.py   --kind benign
run_bucket osharm_harmful     run_condition osharm.py       --kind harmful
run_bucket osharm_benign      run_condition osharm.py       --kind benign
run_bucket redcode_harmful    run_condition redcode_gen.py
run_bucket redcode_benign     run_condition bigcodebench.py --limit "$BCB_LIMIT"

echo ""
echo "All requested buckets complete."
