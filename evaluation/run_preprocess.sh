#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dataset_name> [additional preprocess args]"
    echo "Example: $0 mnist"
    exit 1
fi

DATASET_NAME="${1%.npy}"
shift

DATA_PATH="$SCRIPT_DIR/data/${DATASET_NAME}.npy"
OUTPUT_DIR="$SCRIPT_DIR/preprocess_results/${DATASET_NAME}"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "Data file not found: $DATA_PATH"
    echo "Available datasets in $SCRIPT_DIR/data:"
    find "$SCRIPT_DIR/data" -maxdepth 1 -type f -name "*.npy" -printf "  - %f\n" | sort
    exit 1
fi

cd "$SCRIPT_DIR/.."
python -m evaluation.preprocess_baseline_data \
    --data-path  "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
