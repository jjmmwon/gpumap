#!/bin/bash
# Run the full benchmark suite (micro + e2e) for one or more datasets.
# Results are saved under evaluation/bench_results/ with the given tag.
#
# Usage (from core/):
#   ./evaluation/bench/run_suite.sh <dataset> <tag>
#   ./evaluation/bench/run_suite.sh mnist baseline
#   ./evaluation/bench/run_suite.sh fashion_mnist post_opt
#   DATASETS="mnist fashion_mnist" ./evaluation/bench/run_suite.sh _ baseline
#
# Set PREPARE_BASELINES=1 to build fidelity / NND recall baselines first:
#   PREPARE_BASELINES=1 ./evaluation/bench/run_suite.sh mnist baseline
#
# After optimization, run again with a new tag and compare:
#   python -m evaluation.bench.compare \
#       bench_results/micro_mnist_baseline.json \
#       bench_results/micro_mnist_post_opt.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$CORE_DIR/evaluation/bench_results"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Arguments ──────────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <dataset> <tag>"
    echo "  dataset : e.g. mnist, fashion_mnist, wine_reviews"
    echo "  tag     : e.g. baseline, post_opt_1"
    echo ""
    echo "Override dataset list with: DATASETS='d1 d2' $0 _ <tag>"
    exit 1
fi

DATASET_ARG="$1"
TAG="$2"
DATASETS="${DATASETS:-$DATASET_ARG}"

# ── Microbenchmark settings ────────────────────────────────────────────────
MICRO_N_INSTANCES="${MICRO_N_INSTANCES:-10000}"
MICRO_N_REPEATS="${MICRO_N_REPEATS:-5}"

# ── E2E benchmark settings ─────────────────────────────────────────────────
E2E_STRATEGY="${E2E_STRATEGY:-weighted}"
E2E_LATENCY="${E2E_LATENCY:-10}"

# ── Baseline preparation settings ──────────────────────────────────────────
PREPARE_BASELINES="${PREPARE_BASELINES:-0}"
BASELINE_N_TRIALS="${BASELINE_N_TRIALS:-5}"
BASELINE_GT_MAX_RUNS="${BASELINE_GT_MAX_RUNS:-300}"
BASELINE_GT_MA_WINDOW="${BASELINE_GT_MA_WINDOW:-5}"
BASELINE_GT_MA_THRESHOLD="${BASELINE_GT_MA_THRESHOLD:-0.2}"
BASELINE_N_FIDELITY_RUNS="${BASELINE_N_FIDELITY_RUNS:-10}"

# ── Run ────────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

for DATASET in $DATASETS; do
    if [[ "$DATASET" == "_" ]]; then continue; fi

    echo "============================================================"
    echo "  Suite: dataset=$DATASET  tag=$TAG"
    echo "============================================================"

    MICRO_OUT="$RESULTS_DIR/micro_${DATASET}_${TAG}.json"

    # ── 0. Baseline preparation (optional) ────────────────────────────────
    if [[ "$PREPARE_BASELINES" == "1" ]]; then
        echo ""
        echo ">>> [0/2] Baseline preparation  (n_trials=$BASELINE_N_TRIALS, gt_max_runs=$BASELINE_GT_MAX_RUNS)"
        "$PYTHON_BIN" -m evaluation.bench.prepare_baselines \
            --dataset "$DATASET" \
            --n-trials "$BASELINE_N_TRIALS" \
            --gt-max-runs "$BASELINE_GT_MAX_RUNS" \
            --gt-ma-window "$BASELINE_GT_MA_WINDOW" \
            --gt-ma-threshold "$BASELINE_GT_MA_THRESHOLD" \
            --n-fidelity-runs "$BASELINE_N_FIDELITY_RUNS"
        echo "    Baselines saved: data/${DATASET}/baselines/"
    fi

    # ── 1. Microbenchmark ──────────────────────────────────────────────────
    echo ""
    echo ">>> [1/2] Microbenchmark  (n=$MICRO_N_INSTANCES, repeats=$MICRO_N_REPEATS)"
    "$PYTHON_BIN" -m evaluation.bench.micro \
        --dataset "$DATASET" \
        --n-instances "$MICRO_N_INSTANCES" \
        --n-repeats "$MICRO_N_REPEATS" \
        --output "$MICRO_OUT"
    echo "    Saved: $MICRO_OUT"

    # ── 2. E2E benchmark ───────────────────────────────────────────────────
    echo ""
    echo ">>> [2/2] E2E benchmark  (strategy=$E2E_STRATEGY, latency=${E2E_LATENCY}s, tag=$TAG)"
    "$PYTHON_BIN" -m evaluation.bench.e2e \
        --dataset "$DATASET" \
        --strategy "$E2E_STRATEGY" \
        --target-latency "$E2E_LATENCY" \
        --tag "$TAG"
    echo "    Saved: progressive_results/${DATASET}_${E2E_STRATEGY}_${E2E_LATENCY}_${TAG}_*/"

    echo ""
    echo ">>> Done: $DATASET / $TAG"
done

echo ""
echo "============================================================"
echo "  All benchmarks complete.  Results in: $RESULTS_DIR"
echo ""
echo "  To compare two tags:"
echo "    python -m evaluation.bench.compare \\"
echo "        bench_results/micro_<dataset>_<tag_a>.json \\"
echo "        bench_results/micro_<dataset>_<tag_b>.json"
echo "============================================================"
