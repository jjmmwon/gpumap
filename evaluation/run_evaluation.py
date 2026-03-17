"""
Single-run progressive GPUMAP benchmark with sampled per-iteration evaluation.

Output directory:
    progressive_results/{dataset}_{strategy}_{latency}_{timestamp}/
        metadata.json
        benchmark_results.json
        run_records.json
        runs/run_000001.npz ...

`target_latency` decimal points are replaced with `p` in the default directory name.
For example: 10.0 -> 10p0, 5.5 -> 5p5.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from gpumap import GPUMAP  # noqa: E402

from evaluation.evaluate_emb_certainty import eval_emb_certainty  # noqa: E402
from evaluation.run_progressive_gpumap import (  # noqa: E402
    DEFAULT_MAX_RUNS,
    DEFAULT_MAX_STALLED_RUNS,
    DEFAULT_OUTPUT_ROOT,
    ActualOps,
    RunRecord,
    build_data_loader,
    build_params,
    build_persistent_estimators,
    build_strategy,
    capture_memory_snapshot,
    close_estimator,
    is_finished,
    plan_run,
    resolve_dataset_path,
    resolve_default_version,
    resolve_label_path,
    save_json,
    save_run_outputs,
    to_numpy,
)


LOG = logging.getLogger("sp")
ALL_STRATEGIES = ["weighted", "sequential", "phase_based"]
EMBEDDING_SIMILARITY_KS = (15, 30, 50)
_NNG_RECALL_FALLBACK_WARNED = False


@dataclass
class IterationEvaluation:
    computed: bool
    nng_recall: float | None
    nng_recall_similarity: float | None
    emb_similarity: float | None
    similarity_k: int | None
    embedding_similarity_k15: float | None
    embedding_similarity_k30: float | None
    embedding_similarity_k50: float | None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def latency_str(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lat = latency_str(args.target_latency)
    name = f"{args.dataset}_{args.strategy}_{lat}_{ts}"
    return DEFAULT_OUTPUT_ROOT / name


def run_npz_path(output_dir: Path, run_id: int) -> Path:
    return output_dir / "runs" / f"run_{run_id:06d}.npz"


def load_ground_truth_neighbors(
    dataset_name: str,
    k_values: tuple[int, ...] = EMBEDDING_SIMILARITY_KS,
) -> tuple[dict[int, np.ndarray], Path | None]:
    path = (
        Path(__file__).resolve().parent
        / "preprocess_results"
        / dataset_name
        / "ground_truth_neighbors.npz"
    )
    if not path.exists():
        return {}, None

    ground_truth_by_k: dict[int, np.ndarray] = {}
    with np.load(path) as data:
        for k in k_values:
            key = f"k_{k}"
            if key in data.files:
                ground_truth_by_k[k] = data[key].astype(np.int32)
    return ground_truth_by_k, path


def compute_knn_overlap(pred_knn: np.ndarray, ref_knn: np.ndarray) -> float | None:
    if pred_knn.size == 0 or ref_knn.size == 0:
        return None

    n_rows = min(int(pred_knn.shape[0]), int(ref_knn.shape[0]))
    k = min(int(pred_knn.shape[1]), int(ref_knn.shape[1]))
    if n_rows <= 0 or k <= 0:
        return None

    pred_eval = np.ascontiguousarray(pred_knn[:n_rows, :k], dtype=np.int32)
    ref_eval = np.ascontiguousarray(ref_knn[:n_rows, :k], dtype=np.int32)
    hit = (pred_eval[:, :, None] == ref_eval[:, None, :]).any(axis=2)
    return float(hit.mean())


def compute_exact_knn_indices(data: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    n_samples = int(data.shape[0])
    actual_k = min(int(k), n_samples - 1)
    if actual_k <= 0:
        return np.empty((n_samples, 0), dtype=np.int32)

    nn = NearestNeighbors(n_neighbors=actual_k + 1, algorithm="auto")
    nn.fit(np.ascontiguousarray(data, dtype=np.float32))
    indices = nn.kneighbors(return_distance=False)[:, 1:]
    return np.ascontiguousarray(indices, dtype=np.int32)


def compute_nng_recall(data: np.ndarray, nng: np.ndarray) -> float | None:
    global _NNG_RECALL_FALLBACK_WARNED

    if nng.size == 0:
        return None

    try:
        from evaluation.evaluate_nng_recall import evaluate_nng_recall

        return float(evaluate_nng_recall(data, nng))
    except ModuleNotFoundError as exc:
        if exc.name not in {"cuml", "cupy"}:
            raise
    except ImportError:
        pass

    if not _NNG_RECALL_FALLBACK_WARNED:
        LOG.warning(
            "cuml-based evaluate_nng_recall is unavailable. Falling back to sklearn exact KNN."
        )
        _NNG_RECALL_FALLBACK_WARNED = True

    exact_knn = compute_exact_knn_indices(data, int(nng.shape[1]))
    return compute_knn_overlap(nng, exact_knn)


def resolve_eval_run_ids(total_runs: int, step_size: int) -> list[int]:
    if total_runs <= 0:
        return []
    if step_size <= 1:
        return list(range(1, total_runs + 1))

    run_ids = {1, total_runs}
    run_ids.update(range(step_size, total_runs + 1, step_size))
    return sorted(run_ids)


def evaluate_iteration(
    data: np.ndarray,
    embedding: np.ndarray,
    nn_indices: np.ndarray,
    final_nn_indices: np.ndarray,
    ground_truth_neighbors_by_k: dict[int, np.ndarray],
) -> IterationEvaluation:
    n_inserted = int(nn_indices.shape[0])
    if n_inserted <= 1:
        return IterationEvaluation(
            computed=True,
            nng_recall=None,
            nng_recall_similarity=None,
            emb_similarity=None,
            similarity_k=None,
            embedding_similarity_k15=None,
            embedding_similarity_k30=None,
            embedding_similarity_k50=None,
        )

    graph_k = min(int(nn_indices.shape[1]), n_inserted - 1)
    if graph_k <= 0:
        return IterationEvaluation(
            computed=True,
            nng_recall=None,
            nng_recall_similarity=None,
            emb_similarity=None,
            similarity_k=None,
            embedding_similarity_k15=None,
            embedding_similarity_k30=None,
            embedding_similarity_k50=None,
        )

    nn_eval = np.ascontiguousarray(nn_indices[:, :graph_k], dtype=np.int32)
    data_eval = np.ascontiguousarray(data[:n_inserted], dtype=np.float32)
    embedding_eval = np.ascontiguousarray(embedding[:n_inserted], dtype=np.float32)

    if n_inserted < data.shape[0]:
        nng_recall = compute_nng_recall(data_eval, nn_eval)
    else:
        nng_recall = compute_knn_overlap(nn_eval, final_nn_indices)

    nng_recall_similarity = compute_knn_overlap(
        nn_eval,
        np.ascontiguousarray(final_nn_indices[:n_inserted], dtype=np.int32),
    )

    embedding_similarity: dict[int, float | None] = {
        15: None,
        30: None,
        50: None,
    }
    max_requested_k = min(max(EMBEDDING_SIMILARITY_KS), n_inserted - 1)
    if max_requested_k > 0:
        embedding_knn = compute_exact_knn_indices(embedding_eval, max_requested_k)
        for k in EMBEDDING_SIMILARITY_KS:
            ground_truth_neighbors = ground_truth_neighbors_by_k.get(k)
            if ground_truth_neighbors is None:
                continue
            eval_k = min(k, int(embedding_knn.shape[1]))
            if eval_k <= 0:
                continue
            score, _ = eval_emb_certainty(
                embedding_knn[:, :eval_k],
                ground_truth_neighbors[:n_inserted],
            )
            embedding_similarity[k] = float(score)

    legacy_similarity_k = 15 if embedding_similarity[15] is not None else None

    return IterationEvaluation(
        computed=True,
        nng_recall=nng_recall,
        nng_recall_similarity=nng_recall_similarity,
        emb_similarity=embedding_similarity[15],
        similarity_k=legacy_similarity_k,
        embedding_similarity_k15=embedding_similarity[15],
        embedding_similarity_k30=embedding_similarity[30],
        embedding_similarity_k50=embedding_similarity[50],
    )


def build_record_dict(
    record: RunRecord,
    evaluation: IterationEvaluation | None,
) -> dict[str, Any]:
    payload = asdict(record)
    payload["evaluation"] = asdict(evaluation) if evaluation is not None else None
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-run GPUMAP benchmark with sampled per-iteration evaluation."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=ALL_STRATEGIES,
        metavar="STRATEGY",
    )
    parser.add_argument("--target-latency", type=float, default=10.0, metavar="SEC")
    parser.add_argument(
        "--eval-step-size",
        type=int,
        default=1,
        metavar="N",
        help="Evaluate run 1, every Nth run, and always the final run.",
    )
    parser.add_argument("--npy-data-path", default=None, metavar="FILE")
    parser.add_argument("--output", default=None, metavar="DIR")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.eval_step_size < 1:
        raise ValueError("--eval-step-size must be >= 1")

    version = resolve_default_version()
    dataset_path = (
        Path(args.npy_data_path)
        if args.npy_data_path is not None
        else resolve_dataset_path(
            args.dataset, Path(__file__).resolve().parent / "data"
        )
    )
    label_path = resolve_label_path(args.dataset, dataset_path)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Output dir: %s", output_dir)
    LOG.info(
        "=== RUN | dataset=%s strategy=%s latency=%.1fs ===",
        args.dataset,
        args.strategy,
        args.target_latency,
    )

    data_loader = build_data_loader(
        args.dataset,
        npy_path=str(dataset_path),
        label_npy_path=str(label_path) if label_path is not None else None,
    )
    params = build_params(data_loader)
    insertion_estimator, update_estimator, embedding_estimator = (
        build_persistent_estimators(args.dataset)
    )
    strategy = build_strategy(
        args.strategy,
        data_loader.n_instances,
        params.n_neighbors,
        insertion_estimator,
        update_estimator,
        embedding_estimator,
    )

    ground_truth_neighbors_by_k, ground_truth_path = load_ground_truth_neighbors(
        args.dataset
    )
    if not ground_truth_neighbors_by_k:
        LOG.warning(
            "Ground truth neighbors for embedding similarity not found. "
            "embedding_similarity_k15/k30/k50 will be stored as null."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "dataset_name": args.dataset,
        "dataset_path": str(dataset_path),
        "label_path": str(label_path) if label_path is not None else None,
        "strategy": args.strategy,
        "target_latency": args.target_latency,
        "version": version,
        "timestamp": timestamp,
        "output_dir": str(output_dir),
        "evaluation_step_size": args.eval_step_size,
        "ground_truth_path": str(ground_truth_path) if ground_truth_path else None,
        "ground_truth_similarity_k_values": sorted(ground_truth_neighbors_by_k.keys()),
        "gpumap_params": {
            "n_neighbors": params.n_neighbors,
            "n_components": params.n_components,
            "min_dist": params.min_dist,
            "negative_sample_rate": params.negative_sample_rate,
            "max_epoch": params.max_epoch,
            "verbose": params.verbose,
            "n_instances": params.n_instances,
            "n_features": params.n_features,
        },
        "strategy_params": {
            "weights": list(getattr(strategy, "_weights", [])),
            "reserved": getattr(strategy, "_reserved", None),
            "default_n_ops": getattr(strategy, "default_n_ops", None),
            "confidence": getattr(strategy, "confidence", None),
            "min_insertion_ops": getattr(strategy, "min_insertion_ops", None),
        },
        "records_schema_version": 3,
        "command": " ".join(sys.argv),
        "started_at": datetime.now().isoformat(),
    }
    save_json(output_dir / "metadata.json", metadata)

    model = GPUMAP(
        data_loader=data_loader,
        params=params,
        strategy=strategy,
    )

    records: list[dict[str, Any]] = []
    benchmark_start = time.perf_counter()
    cumulative_wall_time = 0.0
    stop_reason = "max_runs_reached"
    stalled_runs = 0
    prev_state: tuple[int, int, int] | None = None
    try:
        for run_id in range(1, DEFAULT_MAX_RUNS + 1):
            planned_ops, estimated_times = plan_run(model, args.target_latency)

            run_started_at = time.perf_counter()
            result = model.run_with_n_ops(
                planned_ops.insertion,
                planned_ops.update,
                planned_ops.embedding,
            )
            wall_time = time.perf_counter() - run_started_at
            cumulative_wall_time += wall_time

            model._strategy.update(result)
            if result.insertion_completed:
                model._strategy.on_insertion_complete()

            embedding = to_numpy(model.get_embedding())
            nn_indices = to_numpy(model.get_neighbor_indices())

            base_record = RunRecord(
                run_id=run_id,
                planned_ops=planned_ops,
                actual_ops=ActualOps(
                    insertion=int(result.n_insertion_ops),
                    update=int(result.n_update_ops),
                    embedding=int(result.n_embedding_ops),
                ),
                insertion_time=float(result.insertion_time),
                update_time=float(result.update_time),
                embedding_time=float(result.embedding_time),
                wall_time=float(wall_time),
                n_inserted_instances=int(result.n_inserted_instances),
                insertion_completed=bool(result.insertion_completed),
                update_queue_size=int(result.update_queue_size),
                embedding_queue_size=[int(result.embedding_queue_size)],
                cumulative_wall_time=float(cumulative_wall_time),
                estimated_times=estimated_times,
                memory=capture_memory_snapshot(),
            )

            records.append(build_record_dict(base_record, None))
            save_run_outputs(output_dir, run_id, embedding, nn_indices)

            for estimator in (
                insertion_estimator,
                update_estimator,
                embedding_estimator,
            ):
                estimator.log_run(
                    args.dataset,
                    data_loader.n_features,
                    int(result.n_inserted_instances),
                )

            LOG.info(
                "Run %d saved: inserted=%d, actual_ops=(%d,%d,%d), queues=(upd=%d, emb=%d)",
                run_id,
                result.n_inserted_instances,
                result.n_insertion_ops,
                result.n_update_ops,
                result.n_embedding_ops,
                result.update_queue_size,
                result.embedding_queue_size,
            )

            current_state = (
                int(result.n_inserted_instances),
                int(result.update_queue_size),
                int(result.embedding_queue_size),
            )
            actual_total_ops = (
                int(result.n_insertion_ops)
                + int(result.n_update_ops)
                + int(result.n_embedding_ops)
            )
            if current_state == prev_state and actual_total_ops == 0:
                stalled_runs += 1
            else:
                stalled_runs = 0
            prev_state = current_state

            if is_finished(result):
                stop_reason = "completed"
                break

            if stalled_runs >= DEFAULT_MAX_STALLED_RUNS:
                stop_reason = "stalled"
                LOG.warning(
                    "Stopping after %d stalled runs with unchanged state=%s",
                    stalled_runs,
                    current_state,
                )
                break
    finally:
        model.close()
        close_estimator(insertion_estimator)
        close_estimator(update_estimator)
        close_estimator(embedding_estimator)

    benchmark_wall_time = float(time.perf_counter() - benchmark_start)
    eval_run_ids = resolve_eval_run_ids(len(records), args.eval_step_size)
    if eval_run_ids:
        LOG.info(
            "=== EVAL | selected_runs=%d step_size=%d ===",
            len(eval_run_ids),
            args.eval_step_size,
        )
        record_by_run_id = {int(record["run_id"]): record for record in records}

        original_nn_infos_path = (
            Path(__file__).resolve().parent
            / "preprocess_results"
            / args.dataset
            / "original_nn_infos.npz"
        )
        if original_nn_infos_path.exists():
            with np.load(original_nn_infos_path) as nn_data:
                final_nn_indices = np.ascontiguousarray(
                    nn_data["indices"][:, :15], dtype=np.int32
                )
        else:
            LOG.warning("original_nn_infos.npz not found: %s", original_nn_infos_path)
            raise FileNotFoundError(
                f"Ground truth neighbor info not found at {original_nn_infos_path}"
            )
        for run_id in eval_run_ids:
            npz_path = run_npz_path(output_dir, run_id)
            if not npz_path.exists():
                LOG.warning("Missing run artifact for eval: %s", npz_path)
                continue

            with np.load(npz_path) as run_data:
                embedding = np.ascontiguousarray(
                    run_data["embedding"], dtype=np.float32
                )
                nn_indices = np.ascontiguousarray(
                    run_data["nn_indices"], dtype=np.int32
                )

            evaluation = evaluate_iteration(
                data_loader.X,
                embedding,
                nn_indices,
                final_nn_indices,
                ground_truth_neighbors_by_k,
            )
            record_by_run_id[run_id]["evaluation"] = asdict(evaluation)
            LOG.info(
                "Run %d eval: recall=%s recall_similarity=%s emb_sim(k15,k30,k50)=(%s,%s,%s)",
                run_id,
                (
                    f"{evaluation.nng_recall:.6f}"
                    if evaluation.nng_recall is not None
                    else "null"
                ),
                (
                    f"{evaluation.nng_recall_similarity:.6f}"
                    if evaluation.nng_recall_similarity is not None
                    else "null"
                ),
                (
                    f"{evaluation.embedding_similarity_k15:.6f}"
                    if evaluation.embedding_similarity_k15 is not None
                    else "null"
                ),
                (
                    f"{evaluation.embedding_similarity_k30:.6f}"
                    if evaluation.embedding_similarity_k30 is not None
                    else "null"
                ),
                (
                    f"{evaluation.embedding_similarity_k50:.6f}"
                    if evaluation.embedding_similarity_k50 is not None
                    else "null"
                ),
            )

    metadata.update(
        {
            "completed_runs": len(records),
            "stop_reason": stop_reason,
            "total_benchmark_wall_time": benchmark_wall_time,
            "cumulative_model_wall_time": float(cumulative_wall_time),
            "finished_at": datetime.now().isoformat(),
        }
    )

    benchmark_results = {
        "dataset": args.dataset,
        "strategy": args.strategy,
        "target_latency": args.target_latency,
        "version": version,
        "evaluation_step_size": args.eval_step_size,
        "ground_truth_similarity_k_values": sorted(ground_truth_neighbors_by_k.keys()),
        "runs": records,
    }

    save_json(output_dir / "run_records.json", records)
    save_json(output_dir / "benchmark_results.json", benchmark_results)
    save_json(output_dir / "metadata.json", metadata)
    LOG.info("Done. Results at: %s", output_dir)


if __name__ == "__main__":
    main()
