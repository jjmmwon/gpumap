"""
End-to-end benchmark: run GPUMAP to completion and report timing + quality.

Runs until embedding_completed and records total wall time, per-phase time
breakdown, and NNG recall on a representative subsample of the final graph.
Results are saved as JSON for comparison with bench/compare.py.

Usage (from core/):
    python -m evaluation.bench.e2e --dataset mnist
    python -m evaluation.bench.e2e --dataset mnist --strategy weighted --target-latency 10
    python -m evaluation.bench.e2e --dataset fashion_mnist --output bench_results/baseline_e2e.json
    python -m evaluation.bench.e2e --dataset mnist --skip-recall
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
from evaluation.run_progressive_gpumap import (  # noqa: E402
    build_data_loader,
    build_params,
    build_persistent_estimators,
    build_strategy,
    capture_memory_snapshot,
    close_estimator,
    is_finished,
    plan_run,
    resolve_dataset_path,
    resolve_label_path,
    to_numpy,
    DEFAULT_DATA_ROOT,
    DEFAULT_MAX_RUNS,
    DEFAULT_MAX_STALLED_RUNS,
)
from evaluation.utils import compute_knn_indices_and_distances  # noqa: E402
from evaluation.evaluate_emb_certainty import eval_emb_certainty  # noqa: E402

LOG = logging.getLogger("bench.e2e")

BENCH_RESULTS_DIR = Path(__file__).resolve().parent.parent / "bench_results"
BASELINES_DIR_NAME = "baselines"   # subdir under data/{dataset}/
RECALL_SUBSAMPLE = 5000            # max rows used for NNG recall computation
MAX_TIMEOUT_SEC = 3600             # hard wall-clock timeout


def _load_baselines(dataset: str) -> dict:
    """
    Return baseline dict from data/{dataset}/baselines/ if files exist.
    Keys: 'nng_recall_baseline', 'umap_fidelity_baseline', 'ground_truth_path'.
    """
    baselines_dir = DEFAULT_DATA_ROOT / dataset / BASELINES_DIR_NAME
    result: dict = {}

    recall_path = baselines_dir / "nng_recall_baseline.json"
    if recall_path.exists():
        with open(recall_path) as f:
            result["nng_recall_baseline"] = json.load(f)

    fidelity_path = baselines_dir / "umap_fidelity_baseline.json"
    if fidelity_path.exists():
        with open(fidelity_path) as f:
            result["umap_fidelity_baseline"] = json.load(f)

    gt_path = baselines_dir / "ground_truth_neighbors.npz"
    if gt_path.exists():
        result["ground_truth_path"] = str(gt_path)

    return result


def compute_fidelity(
    embedding: np.ndarray,
    ground_truth_path: str,
    n_neighbors: int,
) -> float:
    """
    Compute fidelity of an embedding against ground truth KNN union.
    Fidelity = mean fraction of embedding KNN that falls in the ground truth set.
    """
    with np.load(ground_truth_path) as f:
        key = f"k_{n_neighbors}"
        if key not in f.files:
            available = list(f.files)
            LOG.warning("Key %s not in ground_truth_neighbors.npz; available: %s", key, available)
            key = available[0] if available else None
        if key is None:
            return float("nan")
        ground_truth = f[key].astype(np.int32)

    emb_idx, _ = compute_knn_indices_and_distances(embedding, n_neighbors)
    score, _ = eval_emb_certainty(emb_idx[:, :n_neighbors], ground_truth)
    return float(score)


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


def compute_nng_recall(
    data: np.ndarray,
    nng: np.ndarray,
    n_check: int = RECALL_SUBSAMPLE,
) -> tuple[float, str]:
    """
    Compute NNG recall, using cuML brute-force if available, else sklearn.
    Only the first n_check rows are evaluated to keep runtime bounded.
    Returns (recall, method_name).
    """
    n = min(n_check, nng.shape[0])
    data_sub = np.ascontiguousarray(data[:n], dtype=np.float32)
    nng_sub = np.ascontiguousarray(nng[:n], dtype=np.int32)

    # Filter out sentinel indices (MAX_UINT or >= n total rows)
    n_total = data.shape[0]
    k = nng_sub.shape[1]

    try:
        import cupy as cp
        from cuml.neighbors import NearestNeighbors

        X_gpu = cp.asarray(data_sub)
        exact = (
            NearestNeighbors(n_neighbors=k + 1, algorithm="brute")
            .fit(X_gpu)
            .kneighbors(X_gpu, return_distance=False)
        )[:, 1:].get().astype(np.int32)
        method = "cuml_brute"
    except Exception:
        from sklearn.neighbors import NearestNeighbors as skNN

        nn = skNN(n_neighbors=k + 1, algorithm="auto")
        nn.fit(data_sub)
        exact = nn.kneighbors(return_distance=False)[:, 1:].astype(np.int32)
        method = "sklearn"

    # Mask sentinels in nng_sub (indices >= n_total are invalid)
    valid_mask = (nng_sub >= 0) & (nng_sub < n_total)
    nng_masked = np.where(valid_mask, nng_sub, -1)

    hit = (nng_masked[:, :, None] == exact[:, None, :]).any(axis=2)
    recall = float(hit[valid_mask[:, :hit.shape[1]]].mean()) if valid_mask.any() else 0.0
    return recall, method


def run_e2e(
    dataset: str,
    npy_path: str | None,
    label_path_override: str | None,
    strategy_name: str,
    target_latency: float,
    compute_recall: bool,
    compute_fidelity_flag: bool = True,
) -> dict[str, Any]:
    dataset_path = (
        Path(npy_path)
        if npy_path
        else resolve_dataset_path(dataset, DEFAULT_DATA_ROOT)
    )
    label_path = (
        Path(label_path_override)
        if label_path_override
        else resolve_label_path(dataset, dataset_path)
    )

    data_loader = build_data_loader(
        dataset,
        npy_path=str(dataset_path),
        label_npy_path=str(label_path) if label_path else None,
    )
    params = build_params(data_loader)
    ins_est, upd_est, emb_est = build_persistent_estimators(dataset)
    strategy = build_strategy(
        strategy_name,
        data_loader.n_instances,
        params.n_neighbors,
        ins_est,
        upd_est,
        emb_est,
    )

    model = GPUMAP(data_loader=data_loader, params=params, strategy=strategy)

    # Accumulators
    cumul_insertion_time = 0.0
    cumul_update_time = 0.0
    cumul_embedding_time = 0.0
    cumul_wall_time = 0.0
    total_runs = 0
    stop_reason = "max_runs_reached"
    stalled_runs = 0
    prev_state: tuple[int, int, int] | None = None
    peak_gpu_mb = -1.0

    bench_start = time.perf_counter()
    try:
        for run_id in range(1, DEFAULT_MAX_RUNS + 1):
            if time.perf_counter() - bench_start > MAX_TIMEOUT_SEC:
                stop_reason = "timeout"
                break

            planned_ops, _ = plan_run(model, target_latency)
            t0 = time.perf_counter()
            result = model.run_with_n_ops(
                planned_ops.insertion,
                planned_ops.update,
                planned_ops.embedding,
            )
            wall_time = time.perf_counter() - t0
            cumul_wall_time += wall_time

            model._strategy.update(result)
            if result.insertion_completed:
                model._strategy.on_operation_complete("insertion")
            if result.update_completed:
                model._strategy.on_operation_complete("update")
            if result.embedding_completed:
                model._strategy.on_operation_complete("embedding")

            cumul_insertion_time += float(result.insertion_time)
            cumul_update_time += float(result.update_time)
            cumul_embedding_time += float(result.embedding_time)
            total_runs += 1

            # Track peak GPU usage
            snap = capture_memory_snapshot()
            if snap.gpu_used_mb > peak_gpu_mb:
                peak_gpu_mb = snap.gpu_used_mb

            current_state = (
                int(result.n_inserted_instances),
                int(result.update_queue_size),
                int(result.embedding_queue_size),
            )
            actual_ops = (
                int(result.n_insertion_ops)
                + int(result.n_update_ops)
                + int(result.n_embedding_ops)
            )
            stalled_runs = stalled_runs + 1 if (current_state == prev_state and actual_ops == 0) else 0
            prev_state = current_state

            if run_id % 10 == 0 or is_finished(result):
                LOG.info(
                    "Run %d: inserted=%d ops=(%d,%d,%d) wall=%.2fs cumul=%.1fs",
                    run_id,
                    result.n_inserted_instances,
                    result.n_insertion_ops,
                    result.n_update_ops,
                    result.n_embedding_ops,
                    wall_time,
                    cumul_wall_time,
                )

            if is_finished(result):
                stop_reason = "completed"
                break

            if stalled_runs >= DEFAULT_MAX_STALLED_RUNS:
                stop_reason = "stalled"
                LOG.warning("Stopping: %d stalled runs", stalled_runs)
                break

        total_wall = time.perf_counter() - bench_start

        # Final quality measurement
        quality: dict[str, Any] = {}
        baselines = _load_baselines(dataset)
        if compute_recall:
            LOG.info("Computing NNG recall on first %d instances...", RECALL_SUBSAMPLE)
            nng = to_numpy(model.get_neighbor_indices())
            data_X = data_loader.X
            recall, method = compute_nng_recall(
                data_X, nng, n_check=min(RECALL_SUBSAMPLE, len(data_X))
            )
            quality["nng_recall"] = recall
            quality["recall_n_check"] = min(RECALL_SUBSAMPLE, nng.shape[0])
            quality["method"] = method
            LOG.info("NNG recall=%.4f (method=%s)", recall, method)

            # Compare against NNDescent baseline if available
            nb = baselines.get("nng_recall_baseline")
            if nb:
                quality["nng_recall_baseline_mean"] = nb["mean_recall"]
                quality["nng_recall_vs_baseline"] = recall - nb["mean_recall"]
                LOG.info(
                    "NNG recall vs baseline: %.4f (baseline mean=%.4f, Δ=%+.4f)",
                    recall, nb["mean_recall"], recall - nb["mean_recall"],
                )

        if compute_fidelity_flag and "ground_truth_path" in baselines:
            LOG.info("Computing embedding fidelity against ground truth...")
            embedding = to_numpy(model.get_embedding())
            n_neighbors = params.n_neighbors
            try:
                fidelity = compute_fidelity(embedding, baselines["ground_truth_path"], n_neighbors)
                quality["fidelity"] = fidelity
                quality["fidelity_k"] = n_neighbors
                LOG.info("Fidelity=%.4f (k=%d)", fidelity, n_neighbors)

                # Compare against UMAP fidelity baseline if available
                fb = baselines.get("umap_fidelity_baseline")
                if fb:
                    quality["fidelity_baseline_mean"] = fb["mean"]
                    quality["fidelity_vs_baseline"] = fidelity - fb["mean"]
                    LOG.info(
                        "Fidelity vs UMAP baseline: %.4f (baseline mean=%.4f, Δ=%+.4f)",
                        fidelity, fb["mean"], fidelity - fb["mean"],
                    )
            except Exception as exc:
                LOG.warning("Fidelity computation failed: %s", exc)
        elif compute_fidelity_flag:
            LOG.info(
                "No ground truth found at %s — skipping fidelity. "
                "Run: python -m evaluation.bench.prepare_baselines --dataset %s",
                DEFAULT_DATA_ROOT / dataset / BASELINES_DIR_NAME,
                dataset,
            )

    finally:
        model.close()
        close_estimator(ins_est)
        close_estimator(upd_est)
        close_estimator(emb_est)

    phase_total = cumul_insertion_time + cumul_update_time + cumul_embedding_time

    def _frac(t: float) -> float:
        return t / phase_total if phase_total > 0 else 0.0

    def _tp(n: int, t: float) -> float:
        return n / t if t > 0 else 0.0

    n = data_loader.n_instances
    return {
        "benchmark_type": "e2e",
        "dataset": dataset,
        "n_instances": n,
        "n_features": data_loader.n_features,
        "strategy": strategy_name,
        "target_latency_sec": target_latency,
        "total_wall_time_sec": total_wall,
        "cumulative_model_wall_time_sec": cumul_wall_time,
        "total_runs": total_runs,
        "stop_reason": stop_reason,
        "phase_times_sec": {
            "insertion": cumul_insertion_time,
            "update": cumul_update_time,
            "embedding": cumul_embedding_time,
        },
        "phase_fractions": {
            "insertion": _frac(cumul_insertion_time),
            "update": _frac(cumul_update_time),
            "embedding": _frac(cumul_embedding_time),
        },
        "throughput_per_sec": {
            "insertion": _tp(n, cumul_insertion_time),
            "update": _tp(n, cumul_update_time),
            "embedding": _tp(n, cumul_embedding_time),
        },
        "quality": quality,
        "memory": {
            "peak_gpu_mb": peak_gpu_mb,
        },
        "timestamp": datetime.now().isoformat(),
        "git_hash": _git_hash(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. mnist)")
    p.add_argument("--npy-path", default=None, metavar="FILE")
    p.add_argument("--label-path", default=None, metavar="FILE")
    p.add_argument("--strategy", default="weighted", choices=["weighted", "sequential"])
    p.add_argument("--target-latency", type=float, default=10.0, metavar="SEC")
    p.add_argument("--skip-recall", action="store_true", help="Skip NNG recall computation")
    p.add_argument(
        "--skip-fidelity", action="store_true",
        help="Skip fidelity computation (even if ground truth baselines exist)",
    )
    p.add_argument("--output", default=None, metavar="FILE")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    LOG.info(
        "E2E benchmark: dataset=%s strategy=%s latency=%.1fs",
        args.dataset,
        args.strategy,
        args.target_latency,
    )

    result = run_e2e(
        dataset=args.dataset,
        npy_path=args.npy_path,
        label_path_override=args.label_path,
        strategy_name=args.strategy,
        target_latency=args.target_latency,
        compute_recall=not args.skip_recall,
        compute_fidelity_flag=not args.skip_fidelity,
    )

    if args.output:
        out_path = Path(args.output)
    else:
        BENCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = BENCH_RESULTS_DIR / f"e2e_{args.dataset}_{args.strategy}_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    LOG.info("Results saved → %s", out_path)

    _print_e2e_summary(result)


def _print_e2e_summary(r: dict[str, Any]) -> None:
    print()
    print(f"{'─'*60}")
    print(f"  E2E: {r['dataset']}  strategy={r['strategy']}  "
          f"n={r['n_instances']}  git={r['git_hash']}")
    print(f"{'─'*60}")
    print(f"  Total wall time : {r['total_wall_time_sec']:.1f}s")
    print(f"  Stop reason     : {r['stop_reason']}")
    print(f"  Total runs      : {r['total_runs']}")
    pt = r["phase_times_sec"]
    pf = r["phase_fractions"]
    tp = r["throughput_per_sec"]
    print(f"  Phase breakdown:")
    for ph in ("insertion", "update", "embedding"):
        print(
            f"    {ph:<10}: {pt[ph]:>7.1f}s  ({pf[ph]*100:>4.1f}%)  "
            f"~{tp[ph]:>8,.0f} pts/s"
        )
    q = r.get("quality", {})
    if q.get("nng_recall") is not None:
        recall_line = f"  NNG recall      : {q['nng_recall']:.4f}  (n_check={q['recall_n_check']}, method={q['method']})"
        if "nng_recall_vs_baseline" in q:
            recall_line += f"  baseline_mean={q['nng_recall_baseline_mean']:.4f}  Δ={q['nng_recall_vs_baseline']:+.4f}"
        print(recall_line)
    if q.get("fidelity") is not None:
        fidelity_line = f"  Fidelity        : {q['fidelity']:.4f}  (k={q['fidelity_k']})"
        if "fidelity_vs_baseline" in q:
            fidelity_line += f"  baseline_mean={q['fidelity_baseline_mean']:.4f}  Δ={q['fidelity_vs_baseline']:+.4f}"
        print(fidelity_line)
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
