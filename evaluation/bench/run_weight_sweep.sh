#!/bin/bash
# Run weight_sweep.py sequentially for one or more datasets.
#
# Each dataset runs to completion (weight_sweep.py launches e2e.py per weight
# config as its own subprocess, with its own retries) before the next dataset
# starts. Deliberately does NOT use `set -e` for the sweep itself: if one
# dataset fails outright (e.g. missing baselines), that's logged and the
# script moves on to the next dataset rather than losing the whole run — a
# pass/fail summary is printed at the end, and the script exits non-zero iff
# at least one dataset failed.
#
# Usage (from core/):
#   ./evaluation/bench/run_weight_sweep.sh <dataset> [dataset2 ...]
#   ./evaluation/bench/run_weight_sweep.sh emnist_digits svhn mnist
#   DATASETS="emnist_digits svhn mnist" ./evaluation/bench/run_weight_sweep.sh
#
# Reproduces, per dataset:
#   python -m evaluation.bench.weight_sweep --dataset <dataset> \
#       --include-sequential-baseline --eval-quality-trend --max-measurements 20
#
# Options (env vars):
#   INCLUDE_SEQUENTIAL_BASELINE=1   Pass --include-sequential-baseline (default: 1; set 0 to skip)
#   EVAL_QUALITY_TREND=1            Pass --eval-quality-trend (default: 1; set 0 to skip)
#   MAX_MEASUREMENTS=20             Pass --max-measurements N (default: 20; set "" to skip)
#   TARGET_LATENCY=10               Pass --target-latency SEC (default: 10)
#   TAG_PREFIX=sweep                Pass --tag-prefix (default: sweep)
#   RETRIES=2                       Pass --retries N (default: 2)
#   FORCE=0                         Set to 1 to pass --force (rerun completed configs)
#   WEIGHTS=""                      Pass --weights <space-separated INS,UPD,EMB triples>
#                                    e.g. WEIGHTS="1,1,1 1,12,4" (default: built-in grid)
#   EXTRA_ARGS=""                   Any additional args appended verbatim
#   PYTHON_BIN=python

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -ge 1 ]]; then
    DATASETS=("$@")
elif [[ -n "${DATASETS:-}" ]]; then
    read -r -a DATASETS <<< "$DATASETS"
else
    echo "Usage: $0 <dataset> [dataset2 ...]"
    echo "  or:  DATASETS='d1 d2 d3' $0"
    exit 1
fi

INCLUDE_SEQUENTIAL_BASELINE="${INCLUDE_SEQUENTIAL_BASELINE:-1}"
EVAL_QUALITY_TREND="${EVAL_QUALITY_TREND:-1}"
MAX_MEASUREMENTS="${MAX_MEASUREMENTS-20}"
TARGET_LATENCY="${TARGET_LATENCY:-10}"
TAG_PREFIX="${TAG_PREFIX:-sweep}"
RETRIES="${RETRIES:-2}"
FORCE="${FORCE:-0}"
WEIGHTS="${WEIGHTS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$CORE_DIR"

OK=()
FAILED=()

for DATASET in "${DATASETS[@]}"; do
    echo "============================================================"
    echo "  Weight sweep: dataset=$DATASET"
    echo "============================================================"

    ARGS=(
        --dataset "$DATASET"
        --target-latency "$TARGET_LATENCY"
        --tag-prefix "$TAG_PREFIX"
        --retries "$RETRIES"
    )
    [[ "$INCLUDE_SEQUENTIAL_BASELINE" == "1" ]] && ARGS+=(--include-sequential-baseline)
    [[ "$EVAL_QUALITY_TREND" == "1" ]] && ARGS+=(--eval-quality-trend)
    [[ -n "$MAX_MEASUREMENTS" ]] && ARGS+=(--max-measurements "$MAX_MEASUREMENTS")
    [[ "$FORCE" == "1" ]] && ARGS+=(--force)
    [[ -n "$WEIGHTS" ]] && ARGS+=(--weights $WEIGHTS)
    [[ -n "$EXTRA_ARGS" ]] && ARGS+=($EXTRA_ARGS)

    echo ">>> $PYTHON_BIN -m evaluation.bench.weight_sweep ${ARGS[*]}"
    if "$PYTHON_BIN" -m evaluation.bench.weight_sweep "${ARGS[@]}"; then
        OK+=("$DATASET")
    else
        echo "  !!! weight_sweep FAILED for dataset=$DATASET (continuing to next dataset)"
        FAILED+=("$DATASET")
    fi
    echo ""
done

echo "============================================================"
echo "  All datasets done.  ok=${#OK[@]}  failed=${#FAILED[@]}"
[[ ${#OK[@]} -gt 0 ]] && echo "  OK     : ${OK[*]}"
[[ ${#FAILED[@]} -gt 0 ]] && echo "  FAILED : ${FAILED[*]}"
echo "============================================================"

[[ ${#FAILED[@]} -eq 0 ]]
