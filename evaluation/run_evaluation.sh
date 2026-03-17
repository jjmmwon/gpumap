#!/bin/bash
set -euo pipefail

# Usage examples:
#   ./run_evaluation.sh 10 weighted 1 mnist fashion_mnist
#   ./run_evaluation.sh 10 sequential 5 mnist

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_evaluation.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <target_latency> <strategy> <eval_step_size> <dataset1> [dataset2 ...]"
    echo "Example: $0 10 weighted 1 mnist fashion_mnist"
    exit 1
fi

TARGET_LATENCY="$1"
STRATEGY="$2"
EVAL_STEP_SIZE="$3"
shift 3

case "$STRATEGY" in
    weighted|sequential|phase_based)
        ;;
    *)
        echo "Unsupported strategy: $STRATEGY"
        echo "Allowed strategies: weighted, sequential, phase_based"
        exit 1
        ;;
esac

if ! [[ "$EVAL_STEP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_STEP_SIZE must be a positive integer."
    exit 1
fi

for DATASET in "$@"; do
    DATASET_NAME="${DATASET%.npy}"
    echo "============================================================"
    echo "Running sampled progressive evaluation"
    echo "  dataset: $DATASET_NAME"
    echo "  target_latency: $TARGET_LATENCY"
    echo "  strategy: $STRATEGY"
    echo "  eval_step_size: $EVAL_STEP_SIZE"
    echo "============================================================"

    "$PYTHON_BIN" "$PYTHON_SCRIPT" \
        --dataset "$DATASET_NAME" \
        --strategy "$STRATEGY" \
        --target-latency "$TARGET_LATENCY" \
        --eval-step-size "$EVAL_STEP_SIZE"
done
