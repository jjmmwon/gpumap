#!/bin/bash
set -euo pipefail

# Usage examples:
#   ./run_progressive_gpumap.sh 10 weighted mnist fashion_mnist
#   ./run_progressive_gpumap.sh 5 phase_based mnist

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_progressive_gpumap.py"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <target_latency> <strategy> <dataset1> [dataset2 ...]"
    echo "Example: $0 10 weighted mnist fashion_mnist svhn"
    exit 1
fi

TARGET_LATENCY="$1"
STRATEGY="$2"
shift 2

case "$STRATEGY" in
    weighted|sequential|phase_based)
        ;;
    *)
        echo "Unsupported strategy: $STRATEGY"
        echo "Allowed strategies: weighted, sequential, phase_based"
        exit 1
        ;;
esac

for DATASET in "$@"; do
    DATASET_NAME="${DATASET%.npy}"
    echo "============================================================"
    echo "Running progressive GPUMAP"
    echo "  dataset: $DATASET_NAME"
    echo "  target_latency: $TARGET_LATENCY"
    echo "  strategy: $STRATEGY"
    echo "============================================================"

    python "$PYTHON_SCRIPT" \
        "$DATASET_NAME" \
        "$TARGET_LATENCY" \
        "$STRATEGY"
done
