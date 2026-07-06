"""
e2e.py — End-to-end GPUMAP benchmark.

Runs GPUMAP with a given target_latency and strategy until all operations complete.
Per run:
  - saves NNG and embedding to runs/run_{id:06d}.npz
  - records planned ops, actual ops, estimated time, run time per operation
  - records cumulative wall time
After all operations complete:
  - computes NNG recall (vs baselines/original_nn_infos.npz)
  - computes embedding fidelity (vs baselines/ground_truth_neighbors.npz)
  - saves benchmark_results.json

Output directory:
  progressive_results/{dataset}_{strategy}_{latency}_{tag}_{ts}/
      benchmark_results.json
      runs/
          run_000001.npz    (embedding, nn_indices)
          run_000002.npz
          ...

Usage (from core/):
    python -m evaluation.bench.e2e --dataset mnist
    python -m evaluation.bench.e2e --dataset mnist --target-latency 10 --strategy weighted --tag v1
    python -m evaluation.bench.e2e --dataset mnist --skip-final-quality
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for _p in (str(REPO_ROOT), str(PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpumap import GPUMAP  # noqa: E402
from evaluation.runner import (  # noqa: E402
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
    sanitize_path_component,
    to_numpy,
    DEFAULT_DATA_ROOT,
    DEFAULT_MAX_RUNS,
    DEFAULT_MAX_STALLED_RUNS,
    DEFAULT_OUTPUT_ROOT,
)
from evaluation.utils import compute_knn_indices_and_distances  # noqa: E402
from evaluation.evaluate_emb_certainty import eval_emb_certainty  # noqa: E402

LOG = logging.getLogger("bench.e2e")

BASELINES_DIR_NAME = "baselines"
RECALL_SUBSAMPLE = 10_000


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


def _load_baselines(dataset: str) -> dict[str, Any]:
    baselines_dir = DEFAULT_DATA_ROOT / dataset / BASELINES_DIR_NAME
    result: dict[str, Any] = {}

    orig_knn_path = baselines_dir / "original_nn_infos.npz"
    if orig_knn_path.exists():
        result["original_nn_infos_path"] = str(orig_knn_path)

    gt_path = baselines_dir / "ground_truth_neighbors.npz"
    if gt_path.exists():
        result["ground_truth_path"] = str(gt_path)

    fidelity_path = baselines_dir / "umap_fidelity_baseline.json"
    if fidelity_path.exists():
        with open(fidelity_path) as f:
            result["umap_fidelity_baseline"] = json.load(f)

    return result


def _compute_final_quality(
    model: GPUMAP,
    baselines: dict[str, Any],
    n_neighbors: int,
    n_total: int,
) -> dict[str, Any]:
    quality: dict[str, Any] = {}

    # NNG recall vs brute-force KNN
    if "original_nn_infos_path" in baselines:
        LOG.info("Computing NNG recall ...")
        nng = to_numpy(model.get_neighbor_indices())
        n_check = min(RECALL_SUBSAMPLE, n_total)
        nng_sub = nng[:n_check].astype(np.int32)

        with np.load(baselines["original_nn_infos_path"]) as f:
            exact_sub = f["indices"].astype(np.int32)[:n_check]

        k = min(nng_sub.shape[1], exact_sub.shape[1])
        nng_sub = nng_sub[:, :k]
        exact_sub = exact_sub[:, :k]

        valid = (nng_sub >= 0) & (nng_sub < n_total)
        nng_masked = np.where(valid, nng_sub, -1)
        hit = (nng_masked[:, :, None] == exact_sub[:, None, :]).any(axis=2)
        recall = float(hit[valid[:, : hit.shape[1]]].mean()) if valid.any() else 0.0

        quality["nng_recall"] = recall
        quality["nng_recall_n_check"] = n_check
        quality["nng_recall_k"] = k
        LOG.info("NNG recall=%.4f  (n_check=%d, k=%d)", recall, n_check, k)
    else:
        LOG.info(
            "No original_nn_infos.npz found — skipping NNG recall. "
            "Run: python -m evaluation.bench.prepare_baselines --dataset <dataset>"
        )

    # Embedding fidelity vs ground truth
    if "ground_truth_path" in baselines:
        LOG.info("Computing embedding fidelity ...")
        embedding = to_numpy(model.get_embedding())

        ground_truth = None
        with np.load(baselines["ground_truth_path"]) as f:
            key = f"k_{n_neighbors}"
            if key not in f.files:
                available = list(f.files)
                LOG.warning(
                    "Key %s not in ground_truth_neighbors.npz — available: %s",
                    key, available,
                )
                key = available[0] if available else None
            if key is not None:
                ground_truth = f[key].astype(np.int32)

        if ground_truth is not None:
            emb_idx, _ = compute_knn_indices_and_distances(embedding, n_neighbors)
            score, _ = eval_emb_certainty(emb_idx[:, :n_neighbors], ground_truth)
            quality["fidelity"] = float(score)
            quality["fidelity_k"] = n_neighbors

            fb = baselines.get("umap_fidelity_baseline")
            if fb:
                quality["fidelity_baseline_mean"] = fb["mean"]
                quality["fidelity_baseline_std"] = fb["std"]
                quality["fidelity_vs_baseline"] = float(score) - fb["mean"]
                LOG.info(
                    "Fidelity=%.4f  baseline_mean=%.4f  Δ=%+.4f",
                    score, fb["mean"], float(score) - fb["mean"],
                )
            else:
                LOG.info("Fidelity=%.4f  (no fidelity baseline)", score)
    else:
        LOG.info(
            "No ground_truth_neighbors.npz found — skipping fidelity. "
            "Run: python -m evaluation.bench.prepare_baselines --dataset <dataset>"
        )

    return quality


def _build_run_record(
    run_id: int,
    planned_ops: Any,
    result: Any,
    estimated_times: Any,
    wall_time: float,
    cumulative_wall_time: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "operations": {
            "insertion": {
                "planned_ops": int(planned_ops.insertion),
                "actual_ops": int(result.n_insertion_ops),
                "estimated_time": float(estimated_times.insertion),
                "run_time": float(result.insertion_time),
            },
            "update": {
                "planned_ops": int(planned_ops.update),
                "actual_ops": int(result.n_update_ops),
                "estimated_time": float(estimated_times.update),
                "run_time": float(result.update_time),
            },
            "embedding": {
                "planned_ops": int(planned_ops.embedding),
                "actual_ops": int(result.n_embedding_ops),
                "estimated_time": float(estimated_times.embedding),
                "run_time": float(result.embedding_time),
            },
        },
        "wall_time": float(wall_time),
        "cumulative_wall_time": float(cumulative_wall_time),
        "n_inserted_instances": int(result.n_inserted_instances),
        "insertion_completed": bool(result.insertion_completed),
        "update_completed": bool(result.update_completed),
        "embedding_completed": bool(result.embedding_completed),
        "update_queue_size": int(result.update_queue_size),
        "embedding_queue_size": int(result.embedding_queue_size),
        "memory": asdict(capture_memory_snapshot()),
    }


def _compute_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def _sum_time(key: str) -> float:
        return sum(r["operations"][key]["run_time"] for r in records)

    def _sum_actual(key: str) -> int:
        return sum(r["operations"][key]["actual_ops"] for r in records)

    total_insert = _sum_time("insertion")
    total_update = _sum_time("update")
    total_embed = _sum_time("embedding")
    total_phase = total_insert + total_update + total_embed

    n_insert = _sum_actual("insertion")
    n_update = _sum_actual("update")
    n_embed = _sum_actual("embedding")

    def _frac(v: float) -> float | None:
        return v / total_phase if total_phase > 0 else None

    def _tp(ops: int, t: float) -> float | None:
        return ops / t if t > 0 else None

    return {
        "phase_times_sec": {
            "insertion": total_insert,
            "update": total_update,
            "embedding": total_embed,
        },
        "phase_fractions": {
            "insertion": _frac(total_insert),
            "update": _frac(total_update),
            "embedding": _frac(total_embed),
        },
        "throughput_per_sec": {
            "insertion": _tp(n_insert, total_insert),
            "update": _tp(n_update, total_update),
            "embedding": _tp(n_embed, total_embed),
        },
    }


def run_e2e(
    dataset: str,
    npy_path: str | None,
    strategy_name: str,
    target_latency: float,
    tag: str,
    compute_quality: bool,
    weights: tuple[float, float, float] | None = None,
) -> Path:
    dataset_path = (
        Path(npy_path)
        if npy_path
        else resolve_dataset_path(dataset, DEFAULT_DATA_ROOT)
    )
    label_path = resolve_label_path(dataset, dataset_path)

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
        weights=weights,
    )
    model = GPUMAP(data_loader=data_loader, params=params, strategy=strategy)

    # Output directory
    latency_str = sanitize_path_component(f"{target_latency:g}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        sanitize_path_component(dataset),
        sanitize_path_component(strategy_name),
        latency_str,
    ]
    if tag:
        parts.append(sanitize_path_component(tag))
    parts.append(timestamp)
    output_dir = DEFAULT_OUTPUT_ROOT / "_".join(parts)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(output_dir / "benchmark.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    LOG.addHandler(fh)

    LOG.info(
        "E2E: dataset=%s  strategy=%s  latency=%.1fs  tag=%s",
        dataset, strategy_name, target_latency, tag or "(none)",
    )
    LOG.info("Output dir: %s", output_dir)

    records: list[dict[str, Any]] = []
    cumulative_wall_time = 0.0
    stop_reason = "max_runs_reached"
    stalled_runs = 0
    prev_state: tuple[int, int, int] | None = None

    bench_start = time.perf_counter()
    try:
        for run_id in range(1, DEFAULT_MAX_RUNS + 1):
            planned_ops, estimated_times = plan_run(model, target_latency)

            t0 = time.perf_counter()
            result = model.run_with_n_ops(
                planned_ops.insertion,
                planned_ops.update,
                planned_ops.embedding,
            )
            wall_time = time.perf_counter() - t0
            cumulative_wall_time += wall_time

            # Save per-run NNG and embedding
            embedding = to_numpy(model.get_embedding())
            nn_indices = to_numpy(model.get_neighbor_indices())
            np.savez_compressed(
                runs_dir / f"run_{run_id:06d}.npz",
                embedding=embedding.astype(np.float32),
                nn_indices=nn_indices.astype(np.int32),
            )

            record = _build_run_record(
                run_id=run_id,
                planned_ops=planned_ops,
                result=result,
                estimated_times=estimated_times,
                wall_time=wall_time,
                cumulative_wall_time=cumulative_wall_time,
            )
            records.append(record)

            current_state = (
                int(result.n_inserted_instances),
                int(result.update_queue_size),
                int(result.embedding_queue_size),
            )
            actual_total = (
                int(result.n_insertion_ops)
                + int(result.n_update_ops)
                + int(result.n_embedding_ops)
            )
            stalled_runs = (
                stalled_runs + 1
                if current_state == prev_state and actual_total == 0
                else 0
            )
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
                    cumulative_wall_time,
                )

            if is_finished(result):
                stop_reason = "completed"
                break

            if stalled_runs >= DEFAULT_MAX_STALLED_RUNS:
                stop_reason = "stalled"
                LOG.warning("Stopping: %d consecutive stalled runs", stalled_runs)
                break

        total_wall = time.perf_counter() - bench_start

        # Final quality (only when all operations completed)
        final_quality: dict[str, Any] = {}
        if compute_quality and stop_reason == "completed":
            baselines = _load_baselines(dataset)
            final_quality = _compute_final_quality(
                model, baselines, params.n_neighbors, data_loader.n_instances
            )
        elif compute_quality:
            LOG.info("Skipping final quality: stop_reason=%s", stop_reason)

    finally:
        model.close()
        close_estimator(ins_est)
        close_estimator(upd_est)
        close_estimator(emb_est)

    summary = _compute_summary(records)

    output = {
        "benchmark_type": "e2e",
        "dataset": dataset,
        "strategy": strategy_name,
        "weights": list(weights) if weights is not None else None,
        "target_latency": target_latency,
        "tag": tag,
        "git_hash": _git_hash(),
        "n_runs": len(records),
        "stop_reason": stop_reason,
        "total_wall_time_sec": total_wall,
        "cumulative_model_wall_time_sec": cumulative_wall_time,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "final_quality": final_quality if final_quality else None,
        "runs": records,
    }

    out_path = output_dir / "benchmark_results.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    LOG.info("Saved → %s", out_path)

    _print_summary(output)
    return out_path


def _print_summary(r: dict[str, Any]) -> None:
    print()
    print(f"{'─'*60}")
    print(
        f"  E2E: {r['dataset']}  strategy={r['strategy']}  "
        f"latency={r['target_latency']}s  tag={r['tag'] or '(none)'}  git={r['git_hash']}"
    )
    print(f"{'─'*60}")
    print(f"  Total wall time : {r['total_wall_time_sec']:.1f}s")
    print(f"  Model wall time : {r['cumulative_model_wall_time_sec']:.1f}s")
    print(f"  Stop reason     : {r['stop_reason']}")
    print(f"  Total runs      : {r['n_runs']}")
    sm = r.get("summary", {})
    for ph in ("insertion", "update", "embedding"):
        t = sm.get("phase_times_sec", {}).get(ph)
        frac = sm.get("phase_fractions", {}).get(ph)
        tp = sm.get("throughput_per_sec", {}).get(ph)
        t_s = f"{t:.1f}s" if t is not None else "n/a"
        frac_s = f"{frac*100:.1f}%" if frac is not None else "n/a"
        tp_s = f"{tp:,.0f} pts/s" if tp is not None else "n/a"
        print(f"  {ph:<12}: {t_s:>8}  ({frac_s})  {tp_s}")
    q = r.get("final_quality") or {}
    if q.get("nng_recall") is not None:
        print(
            f"  NNG recall      : {q['nng_recall']:.4f}"
            f"  (k={q.get('nng_recall_k', '?')}, n_check={q.get('nng_recall_n_check', '?')})"
        )
    if q.get("fidelity") is not None:
        line = f"  Fidelity        : {q['fidelity']:.4f}  (k={q['fidelity_k']})"
        if "fidelity_vs_baseline" in q:
            line += (
                f"  (baseline={q['fidelity_baseline_mean']:.4f}"
                f"  Δ={q['fidelity_vs_baseline']:+.4f})"
            )
        print(line)
    print(f"{'─'*60}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--npy-path", default=None, metavar="FILE")
    p.add_argument(
        "--strategy",
        default="weighted",
        choices=["weighted", "sequential"],
    )
    p.add_argument("--target-latency", type=float, default=10.0, metavar="SEC")
    p.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=None,
        metavar=("INS", "UPD", "EMB"),
        help="Time budget weights for weighted strategy (insertion update embedding). "
             "Default: WeightedStrategy defaults (1.0 12.0 1.0).",
    )
    p.add_argument("--tag", default="", help="Tag appended to output dir name (e.g. v1-opt)")
    p.add_argument("--skip-final-quality", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_e2e(
        dataset=args.dataset,
        npy_path=args.npy_path,
        strategy_name=args.strategy,
        target_latency=args.target_latency,
        tag=args.tag,
        compute_quality=not args.skip_final_quality,
        weights=tuple(args.weights) if args.weights is not None else None,
    )


if __name__ == "__main__":
    main()
