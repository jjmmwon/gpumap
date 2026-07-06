#!/bin/bash
# Prepare baseline data (exact KNN, NNDescent recall, ground truth, UMAP fidelity)
# for one or more datasets. Each dataset runs in a separate process so GPU memory
# is fully released before the next one starts.
#
# Usage (from core/):
#   ./evaluation/bench/run_baselines.sh <dataset> [dataset2 ...]
#   ./evaluation/bench/run_baselines.sh mnist
#   ./evaluation/bench/run_baselines.sh mnist fashion_mnist wine_reviews
#
# Options (env vars):
#   N_TRIALS=5              NNDescent recall: number of trials
#   GT_MAX_RUNS=300         Ground truth: max UMAP runs
#   GT_MA_WINDOW=5          Ground truth: SMA window for h/k convergence
#   GT_MA_THRESHOLD=0.2     Ground truth: SMA threshold for convergence
#   N_FIDELITY_RUNS=10      UMAP fidelity baseline: number of runs
#   PHASES="knn nng_recall ground_truth fidelity"  Phases to run
#   FORCE=0                 Set to 1 to recompute even if output exists
#   PYTHON_BIN=python

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dataset> [dataset2 ...]"
    exit 1
fi

DATASETS=("$@")

N_TRIALS="${N_TRIALS:-5}"
GT_MAX_RUNS="${GT_MAX_RUNS:-300}"
GT_MA_WINDOW="${GT_MA_WINDOW:-5}"
GT_MA_THRESHOLD="${GT_MA_THRESHOLD:-0.2}"
N_FIDELITY_RUNS="${N_FIDELITY_RUNS:-10}"
PHASES="${PHASES:-}"
FORCE="${FORCE:-0}"

for DATASET in "${DATASETS[@]}"; do
    echo "============================================================"
    echo "  Baselines: dataset=$DATASET"
    echo "============================================================"

    ARGS=(
        --dataset "$DATASET"
        --n-trials "$N_TRIALS"
        --gt-max-runs "$GT_MAX_RUNS"
        --gt-ma-window "$GT_MA_WINDOW"
        --gt-ma-threshold "$GT_MA_THRESHOLD"
        --n-fidelity-runs "$N_FIDELITY_RUNS"
    )
    [[ -n "$PHASES" ]] && ARGS+=(--phases $PHASES)
    [[ "$FORCE" == "1" ]] && ARGS+=(--force)

    "$PYTHON_BIN" -m evaluation.bench.prepare_baselines "${ARGS[@]}"

    echo "  → data/${DATASET}/baselines/"
    echo ""
done

echo "All baselines done."
