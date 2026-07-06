"""
Microbenchmark: isolated per-operation GPU timing for insert, update, and embed.

Runs each operation type in isolation with a fixed problem size and reports
GPU time (from C++ internal timers), throughput, and summary statistics.
Intended to be run before and after optimizations and compared with compare.py.

Usage (from core/):
    python -m evaluation.bench.micro --dataset mnist
    python -m evaluation.bench.micro --dataset mnist --n-instances 10000
    python -m evaluation.bench.micro --dataset mnist --output bench_results/baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for _p in (str(REPO_ROOT), str(PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpumap import GPUMAP, GPUMAPParams  # noqa: E402
from gpumap.gpumap import find_ab_params  # noqa: E402
from gpumap.data_loader import NumpyDataLoader  # noqa: E402
from gpumap.schedule import WeightedStrategy, DefaultLinearEstimator  # noqa: E402
from evaluation.runner import (  # noqa: E402
    resolve_dataset_path,
    DEFAULT_DATA_ROOT,
)

LOG = logging.getLogger("bench.micro")

BENCH_RESULTS_DIR = Path(__file__).resolve().parent.parent / "bench_results"
N_WARMUP_PASSES = 2
N_REPEATS_DEFAULT = 5


def _git_hash() -> str:
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return h.decode().strip()
    except Exception:
        return "unknown"


def _stats(times: list[float]) -> dict[str, float]:
    a = np.array(times)
    return {
        "min": float(a.min()),
        "p5": float(np.percentile(a, 5)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
    }


def _throughput(n_ops: int, times: list[float]) -> list[float]:
    return [n_ops / t if t > 0 else 0.0 for t in times]


def _build_model(data: np.ndarray, n_instances: int) -> tuple[GPUMAP, NumpyDataLoader]:
    """Construct a fresh model backed by a numpy array."""
    import tempfile, os

    # Write a temporary .npy so NumpyDataLoader can load it
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    np.save(tmp.name, data[:n_instances])
    tmp.close()

    loader = NumpyDataLoader("bench", tmp.name)
    os.unlink(tmp.name)

    params = GPUMAPParams()
    params.n_instances = loader.n_instances
    params.n_features = loader.n_features

    params.a, params.b = find_ab_params(1.0, params.min_dist)

    strategy = WeightedStrategy(
        n_instances=loader.n_instances,
        n_neighbors=params.n_neighbors,
        insertion_estimator=DefaultLinearEstimator(),
        update_estimator=DefaultLinearEstimator(),
        embedding_estimator=DefaultLinearEstimator(),
    )
    model = GPUMAP(data_loader=loader, params=params, strategy=strategy)
    return model, loader


def bench_insert(
    data: np.ndarray,
    n_instances: int,
    n_repeats: int,
) -> dict[str, Any]:
    """
    Insert n_instances data points in (n_warmup + n_repeats) equal batches.
    Returns timing info for the n_repeats measurement batches.
    """
    total_batches = N_WARMUP_PASSES + n_repeats
    batch_size = max(1, n_instances // total_batches)
    # Ensure we have enough data
    needed = batch_size * total_batches
    actual_n = min(n_instances, len(data))
    if needed > actual_n:
        batch_size = max(1, actual_n // total_batches)
        needed = batch_size * total_batches

    LOG.info(
        "INSERT: n_instances=%d, batch_size=%d, warmup=%d, measure=%d",
        actual_n,
        batch_size,
        N_WARMUP_PASSES,
        n_repeats,
    )

    model, _ = _build_model(data, needed)
    try:
        # Warmup passes
        for _ in range(N_WARMUP_PASSES):
            model.run_with_n_ops(batch_size, 0, 0)

        # Measurement passes
        gpu_times: list[float] = []
        wall_times: list[float] = []
        for i in range(n_repeats):
            t0 = time.perf_counter()
            res = model.run_with_n_ops(batch_size, 0, 0)
            wall_times.append(time.perf_counter() - t0)
            gpu_times.append(float(res.insertion_time))
            LOG.info(
                "  insert[%d/%d]: gpu=%.3fms wall=%.3fms n_inserted=%d",
                i + 1,
                n_repeats,
                gpu_times[-1] * 1e3,
                wall_times[-1] * 1e3,
                res.n_inserted_instances,
            )
    finally:
        model.close()

    tp_gpu = _throughput(batch_size, gpu_times)
    tp_wall = _throughput(batch_size, wall_times)
    return {
        "batch_size": batch_size,
        "n_warmup": N_WARMUP_PASSES,
        "n_repeats": n_repeats,
        "gpu_times_sec": gpu_times,
        "wall_times_sec": wall_times,
        "gpu_throughput_per_sec": tp_gpu,
        "wall_throughput_per_sec": tp_wall,
        "gpu_stats": _stats(gpu_times),
        "wall_stats": _stats(wall_times),
    }


def bench_update(
    data: np.ndarray,
    n_instances: int,
    n_repeats: int,
) -> dict[str, Any]:
    """
    Insert all n_instances, then benchmark update-only passes.
    Each pass processes all n_instances nodes (n_ops >= n_inserted triggers full sweep).
    """
    actual_n = min(n_instances, len(data))
    LOG.info(
        "UPDATE: n_instances=%d, n_ops=%d, warmup=%d, measure=%d",
        actual_n,
        actual_n,
        N_WARMUP_PASSES,
        n_repeats,
    )
    model, _ = _build_model(data, actual_n)
    try:
        # Insert all data first
        model.run_with_n_ops(actual_n, 0, 0)

        n_ops = actual_n  # >= n_inserted → always does a full sweep

        # Warmup
        for _ in range(N_WARMUP_PASSES):
            model.run_with_n_ops(0, n_ops, 0)

        # Measurement
        gpu_times: list[float] = []
        wall_times: list[float] = []
        for i in range(n_repeats):
            t0 = time.perf_counter()
            res = model.run_with_n_ops(0, n_ops, 0)
            wall_times.append(time.perf_counter() - t0)
            gpu_times.append(float(res.update_time))
            LOG.info(
                "  update[%d/%d]: gpu=%.3fms wall=%.3fms changed=%d",
                i + 1,
                n_repeats,
                gpu_times[-1] * 1e3,
                wall_times[-1] * 1e3,
                res.n_update_changed_edges,
            )
    finally:
        model.close()

    tp_gpu = _throughput(n_ops, gpu_times)
    tp_wall = _throughput(n_ops, wall_times)
    return {
        "n_ops": n_ops,
        "n_warmup": N_WARMUP_PASSES,
        "n_repeats": n_repeats,
        "gpu_times_sec": gpu_times,
        "wall_times_sec": wall_times,
        "gpu_throughput_per_sec": tp_gpu,
        "wall_throughput_per_sec": tp_wall,
        "gpu_stats": _stats(gpu_times),
        "wall_stats": _stats(wall_times),
    }


def bench_embed(
    data: np.ndarray,
    n_instances: int,
    n_repeats: int,
) -> dict[str, Any]:
    """
    Insert all n_instances and run a single update warmup, then benchmark
    embed-only passes. n_ops >= n_inserted ensures a full sweep each time.
    """
    actual_n = min(n_instances, len(data))
    LOG.info(
        "EMBED: n_instances=%d, n_ops=%d, warmup=%d, measure=%d",
        actual_n,
        actual_n,
        N_WARMUP_PASSES,
        n_repeats,
    )
    model, _ = _build_model(data, actual_n)
    try:
        # Insert all data + one update pass so the graph is populated
        model.run_with_n_ops(actual_n, 0, 0)
        model.run_with_n_ops(0, actual_n, 0)

        n_ops = actual_n  # >= n_inserted → always does a full sweep

        # Warmup embed passes
        for _ in range(N_WARMUP_PASSES):
            model.run_with_n_ops(0, 0, n_ops)

        # Measurement
        gpu_times: list[float] = []
        wall_times: list[float] = []
        for i in range(n_repeats):
            t0 = time.perf_counter()
            res = model.run_with_n_ops(0, 0, n_ops)
            wall_times.append(time.perf_counter() - t0)
            gpu_times.append(float(res.embedding_time))
            LOG.info(
                "  embed[%d/%d]: gpu=%.3fms wall=%.3fms",
                i + 1,
                n_repeats,
                gpu_times[-1] * 1e3,
                wall_times[-1] * 1e3,
            )
    finally:
        model.close()

    tp_gpu = _throughput(n_ops, gpu_times)
    tp_wall = _throughput(n_ops, wall_times)
    return {
        "n_ops": n_ops,
        "n_warmup": N_WARMUP_PASSES,
        "n_repeats": n_repeats,
        "gpu_times_sec": gpu_times,
        "wall_times_sec": wall_times,
        "gpu_throughput_per_sec": tp_gpu,
        "wall_throughput_per_sec": tp_wall,
        "gpu_stats": _stats(gpu_times),
        "wall_stats": _stats(wall_times),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. mnist)")
    p.add_argument("--npy-path", default=None, metavar="FILE", help="Path to .npy data file (overrides dataset lookup)")
    p.add_argument("--n-instances", type=int, default=10000, metavar="N", help="Number of instances to benchmark (default: 10000)")
    p.add_argument("--n-repeats", type=int, default=N_REPEATS_DEFAULT, metavar="N", help="Measurement repetitions per operation (default: 5)")
    p.add_argument("--output", default=None, metavar="FILE", help="Output JSON path (default: bench_results/micro_<dataset>_<ts>.json)")
    p.add_argument("--ops", nargs="+", choices=["insert", "update", "embed"], default=["insert", "update", "embed"], help="Which operations to benchmark (default: all)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # Resolve data path
    if args.npy_path:
        npy_path = Path(args.npy_path)
    else:
        npy_path = resolve_dataset_path(args.dataset, DEFAULT_DATA_ROOT)

    data = np.load(npy_path).astype(np.float32)
    n_instances = min(args.n_instances, len(data))
    LOG.info("Dataset: %s  shape=%s  using first %d instances", npy_path.name, data.shape, n_instances)

    results: dict[str, Any] = {
        "benchmark_type": "micro",
        "dataset": args.dataset,
        "npy_path": str(npy_path),
        "n_instances": n_instances,
        "n_features": int(data.shape[1]),
        "n_repeats": args.n_repeats,
        "timestamp": datetime.now().isoformat(),
        "git_hash": _git_hash(),
    }

    if "insert" in args.ops:
        LOG.info("=== INSERT benchmark ===")
        results["insert"] = bench_insert(data, n_instances, args.n_repeats)

    if "update" in args.ops:
        LOG.info("=== UPDATE benchmark ===")
        results["update"] = bench_update(data, n_instances, args.n_repeats)

    if "embed" in args.ops:
        LOG.info("=== EMBED benchmark ===")
        results["embed"] = bench_embed(data, n_instances, args.n_repeats)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        BENCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = BENCH_RESULTS_DIR / f"micro_{args.dataset}_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    LOG.info("Results saved → %s", out_path)

    # Print summary
    _print_micro_summary(results)


def _print_micro_summary(results: dict[str, Any]) -> None:
    print()
    print(f"{'─'*60}")
    print(f"  Microbenchmark: {results['dataset']}  "
          f"n={results['n_instances']}  git={results['git_hash']}")
    print(f"{'─'*60}")
    fmt = "  {:<8}  gpu_median={:>8.1f}ms  throughput={:>10,.0f} pts/s"
    for op in ("insert", "update", "embed"):
        if op not in results:
            continue
        r = results[op]
        s = r["gpu_stats"]
        tp = r["gpu_throughput_per_sec"]
        med_tp = float(np.median(tp)) if tp else 0.0
        print(fmt.format(op.upper(), s["median"] * 1e3, med_tp))
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
