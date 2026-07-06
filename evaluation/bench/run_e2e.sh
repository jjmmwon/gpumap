#!/bin/bash
# Run the end-to-end GPUMAP benchmark for one or more datasets.
# Each dataset runs in a separate process so GPU memory is fully released
# before the next one starts.
#
# Usage (from core/):
#   ./evaluation/bench/run_e2e.sh <tag> <dataset> [dataset2 ...]
#   ./evaluation/bench/run_e2e.sh baseline mnist
#   ./evaluation/bench/run_e2e.sh v1-opt mnist fashion_mnist wine_reviews
#
# Options (env vars):
#   STRATEGY=weighted       Strategy: weighted | sequential
#   LATENCY=10              Target latency in seconds
#   WEIGHTS=""              Time budget weights "ins upd emb" (e.g. "1 12 1")
#   SKIP_QUALITY=0          Set to 1 to skip final NNG recall + fidelity
#   PYTHON_BIN=python

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <tag> <dataset> [dataset2 ...]"
    exit 1
fi

TAG="$1"
shift
DATASETS=("$@")

STRATEGY="${STRATEGY:-weighted}"
LATENCY="${LATENCY:-10}"
WEIGHTS="${WEIGHTS:-}"
SKIP_QUALITY="${SKIP_QUALITY:-0}"

for DATASET in "${DATASETS[@]}"; do
    echo "============================================================"
    echo "  E2E: dataset=$DATASET  tag=$TAG  strategy=$STRATEGY  latency=${LATENCY}s"
    echo "============================================================"

    ARGS=(
        --dataset "$DATASET"
        --strategy "$STRATEGY"
        --target-latency "$LATENCY"
        --tag "$TAG"
    )
    [[ -n "$WEIGHTS" ]] && ARGS+=(--weights $WEIGHTS)
    [[ "$SKIP_QUALITY" == "1" ]] && ARGS+=(--skip-final-quality)

    "$PYTHON_BIN" -m evaluation.bench.e2e "${ARGS[@]}"

    echo "  → progressive_results/${DATASET}_${STRATEGY}_${LATENCY}_${TAG}_*/"
    echo ""
done

echo "All E2E benchmarks done."
