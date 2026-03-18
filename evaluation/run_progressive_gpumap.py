from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np

try:
    import tomllib
except ImportError:  # pragma: no cover - python < 3.11 fallback
    tomllib = None

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from gpumap.data_loader import DataLoader, NumpyDataLoader, PreloadDataLoader

try:
    import cupy as cp
except ImportError:  # pragma: no cover - runtime fallback
    cp = None

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

from gpumap import (  # noqa: E402
    GPUMAP,
    GPUMAPParams,
    DefaultLinearEstimator,
    PersistentEstimator,
    SequentialPriorityStrategy,
    WeightedStrategy,
)
from gpumap.schedule.weighted import FocusedStrategy


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "progressive_results"
DEFAULT_MAX_RUNS = 10000
DEFAULT_MAX_STALLED_RUNS = 5
DEFAULT_ESTIMATOR_TYPE = "persistent_linear"


@dataclass
class PlannedOps:
    insertion: int
    update: int
    embedding: int


@dataclass
class ActualOps:
    insertion: int
    update: int
    embedding: int


@dataclass
class EstimatedTimes:
    insertion: float
    update: float
    embedding: float


@dataclass
class MemorySnapshot:
    gpu_used_mb: float
    gpu_total_mb: float
    cpu_rss_mb: float


@dataclass
class RunRecord:
    run_id: int
    planned_ops: PlannedOps
    actual_ops: ActualOps
    insertion_time: float
    update_time: float
    embedding_time: float
    wall_time: float
    n_inserted_instances: int
    insertion_completed: bool
    update_queue_size: int
    embedding_queue_size: list[int]
    cumulative_wall_time: float
    estimated_times: EstimatedTimes
    memory: MemorySnapshot


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return sanitized.strip("-") or "unknown"


def resolve_dataset_path(dataset_name: str, data_root: Path) -> Path:
    candidates = [
        data_root / f"{dataset_name}.npy",
        data_root / dataset_name / f"{dataset_name}.npy",
        data_root / dataset_name / f"{dataset_name}_X.npy",
    ]
    for path in candidates:
        if path.is_file():
            return path

    matches = sorted(data_root.glob(f"**/{dataset_name}.npy"))
    if len(matches) == 1:
        return matches[0]

    matches = sorted(data_root.glob(f"**/{dataset_name}_X.npy"))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not resolve dataset '{dataset_name}' under {data_root}."
    )


def resolve_label_path(dataset_name: str, dataset_path: Path) -> Path | None:
    candidates = [
        dataset_path.with_name(f"{dataset_name}_y.npy"),
        dataset_path.with_name(f"{dataset_name}_Y.npy"),
        dataset_path.with_name("labels.npy"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"progressive_gpumap.{output_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "progressive.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def resolve_default_version() -> str:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    if tomllib is None or not pyproject_path.is_file():
        return "default"
    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return "default"

    version = data.get("project", {}).get("version")
    return version if isinstance(version, str) and version else "default"


def build_params(data_loader: DataLoader) -> GPUMAPParams:
    params = GPUMAPParams()
    params.n_instances = data_loader.n_instances
    params.n_features = data_loader.n_features
    return params


def build_data_loader(
    dataset: str, npy_path: str | None = None, label_npy_path: str | None = None
) -> DataLoader:
    if npy_path:
        return NumpyDataLoader(
            dataset_name=dataset, npy_path=npy_path, label_npy_path=label_npy_path
        )

    return PreloadDataLoader(dataset)


def resolve_estimator_db_path(dataset_name: str, estimator_type: str) -> Path:
    db_dir = DEFAULT_OUTPUT_ROOT / "_estimator_cache"
    db_name = "_".join(
        [
            sanitize_path_component(dataset_name),
            sanitize_path_component(estimator_type),
        ]
    )
    return db_dir / f"{db_name}.sqlite3"


def build_persistent_estimators(
    dataset_name: str,
) -> tuple[PersistentEstimator, PersistentEstimator, PersistentEstimator]:
    db_path = resolve_estimator_db_path(dataset_name, DEFAULT_ESTIMATOR_TYPE)

    return (
        PersistentEstimator(DefaultLinearEstimator(), str(db_path), "insertion"),
        PersistentEstimator(DefaultLinearEstimator(), str(db_path), "update"),
        PersistentEstimator(DefaultLinearEstimator(), str(db_path), "embedding"),
    )


def build_strategy(
    strategy_name: str,
    n_instances: int,
    n_neighbors: int,
    insertion_estimator: Any,
    update_estimator: Any,
    embedding_estimator: Any,
):
    common_kwargs = {
        "n_instances": n_instances,
        "n_neighbors": n_neighbors,
        "insertion_estimator": insertion_estimator,
        "update_estimator": update_estimator,
        "embedding_estimator": embedding_estimator,
    }

    if strategy_name == "weighted":
        return WeightedStrategy(**common_kwargs)
    if strategy_name == "sequential":
        return SequentialPriorityStrategy(**common_kwargs)
    if strategy_name == "insertion_focused":
        return FocusedStrategy(**common_kwargs, focus="insertion", focus_rate=0.5)
    if strategy_name == "update_focused":
        return FocusedStrategy(**common_kwargs, focus="update", focus_rate=0.5)
    if strategy_name == "embedding_focused":
        return FocusedStrategy(**common_kwargs, focus="embedding", focus_rate=0.5)
    raise ValueError(f"Unsupported strategy: {strategy_name}")


def safe_predict_time(estimator: Any, ops: int, quantile: float) -> float:
    if ops <= 0:
        return 0.0
    try:
        return float(estimator.predict_time(int(ops), quantile))
    except ValueError:
        return 0.0


def to_numpy(array: Any) -> np.ndarray:
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def capture_memory_snapshot() -> MemorySnapshot:
    gpu_used_mb = -1.0
    gpu_total_mb = -1.0
    if cp is not None:
        try:
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            gpu_total_mb = float(total_bytes / (1024**2))
            gpu_used_mb = float((total_bytes - free_bytes) / (1024**2))
        except Exception:
            gpu_used_mb = -1.0
            gpu_total_mb = -1.0

    cpu_rss_mb = -1.0
    if psutil is not None:
        try:
            cpu_rss_mb = float(
                psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            )
        except Exception:
            cpu_rss_mb = -1.0

    return MemorySnapshot(
        gpu_used_mb=gpu_used_mb,
        gpu_total_mb=gpu_total_mb,
        cpu_rss_mb=cpu_rss_mb,
    )


def plan_run(model: GPUMAP, target_latency: float) -> tuple[PlannedOps, EstimatedTimes]:
    strategy = model._strategy
    allocation = strategy.allocate(target_latency)

    n_insertion_ops = min(
        int(allocation[0]),
        model.data_loader.n_instances - model.n_inserted_instances,
    )
    n_update_ops = int(allocation[1])

    n_embedding_ops = int(allocation[2])

    planned_ops = PlannedOps(
        insertion=n_insertion_ops,
        update=n_update_ops,
        embedding=n_embedding_ops,
    )
    estimated_times = EstimatedTimes(
        insertion=safe_predict_time(
            strategy.insertion_estimator, planned_ops.insertion, strategy.confidence
        ),
        update=safe_predict_time(
            strategy.update_estimator, planned_ops.update, strategy.confidence
        ),
        embedding=safe_predict_time(
            strategy.embedding_estimator, planned_ops.embedding, strategy.confidence
        ),
    )
    return planned_ops, estimated_times


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_run_outputs(
    output_dir: Path,
    run_id: int,
    embedding: np.ndarray,
    nn_indices: np.ndarray,
) -> None:
    run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = f"run_{run_id:06d}"
    np.savez_compressed(
        run_dir / f"{stem}.npz",
        embedding=np.ascontiguousarray(embedding, dtype=np.float32),
        nn_indices=np.ascontiguousarray(nn_indices, dtype=np.int32),
    )


def close_estimator(estimator: Any) -> None:
    close = getattr(estimator, "close", None)
    if callable(close):
        close()


def is_finished(result: Any) -> bool:
    return result.embedding_completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run progressive GPUMAP and persist per-run artifacts."
    )
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("target_latency", type=float)
    parser.add_argument(
        "strategy",
        type=str,
        choices=[
            "weighted",
            "sequential",
            "insertion_focused",
            "update_focused",
            "embedding_focused",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = resolve_default_version()

    dataset_path = resolve_dataset_path(args.dataset_name, DEFAULT_DATA_ROOT)
    label_path = resolve_label_path(args.dataset_name, dataset_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = "_".join(
        [
            sanitize_path_component(args.dataset_name),
            sanitize_path_component(args.strategy),
            sanitize_path_component(f"{args.target_latency:g}"),
            sanitize_path_component(version),
            timestamp,
        ]
    )
    output_dir = DEFAULT_OUTPUT_ROOT / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=False)

    logger = setup_logger(output_dir)
    logger.info("Loading dataset from %s", dataset_path)
    data_loader = build_data_loader(
        args.dataset_name,
        npy_path=str(dataset_path),
        label_npy_path=str(label_path) if label_path is not None else None,
    )
    logger.info(
        "Dataset loaded: n_instances=%d, n_features=%d",
        data_loader.n_instances,
        data_loader.n_features,
    )

    params = build_params(data_loader)
    insertion_estimator, update_estimator, embedding_estimator = (
        build_persistent_estimators(args.dataset_name)
    )
    strategy = build_strategy(
        args.strategy,
        data_loader.n_instances,
        params.n_neighbors,
        insertion_estimator,
        update_estimator,
        embedding_estimator,
    )

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_path": str(dataset_path),
        "label_path": str(label_path) if label_path is not None else None,
        "strategy": args.strategy,
        "target_latency": args.target_latency,
        "version": version,
        "timestamp": timestamp,
        "output_dir": str(output_dir),
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
        "estimator_kind": {
            "insertion": "persistent_linear",
            "update": "persistent_linear",
            "embedding": "persistent_linear",
        },
        "estimator_db_path": str(
            resolve_estimator_db_path(args.dataset_name, DEFAULT_ESTIMATOR_TYPE)
        ),
        "defaults_policy": {
            "cli_args": ["dataset_name", "target_latency", "strategy"],
            "gpumap_params": "GPUMAPParams constructor defaults",
            "strategy_params": f"{args.strategy} constructor defaults",
            "estimators": {
                "insertion": "PersistentEstimator(DefaultLinearEstimator)",
                "update": "PersistentEstimator(DefaultLinearEstimator)",
                "embedding": "PersistentEstimator(DefaultLinearEstimator)",
            },
        },
        "records_schema_version": 1,
        "command": " ".join(sys.argv),
        "started_at": datetime.now().isoformat(),
    }
    save_json(output_dir / "metadata.json", metadata)

    model = GPUMAP(
        data_loader=data_loader,
        params=params,
        strategy=strategy,
    )

    records: list[RunRecord] = []
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

            embedding = to_numpy(model.get_embedding())
            nn_indices = to_numpy(model.get_neighbor_indices())

            record = RunRecord(
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
            records.append(record)

            save_run_outputs(output_dir, run_id, embedding, nn_indices)
            for estimator in (
                insertion_estimator,
                update_estimator,
                embedding_estimator,
            ):
                estimator.log_run(
                    args.dataset_name,
                    data_loader.n_features,
                    int(result.n_inserted_instances),
                )
            logger.info(
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
                logger.warning(
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

    metadata.update(
        {
            "completed_runs": len(records),
            "stop_reason": stop_reason,
            "total_benchmark_wall_time": float(time.perf_counter() - benchmark_start),
            "cumulative_model_wall_time": float(cumulative_wall_time),
            "finished_at": datetime.now().isoformat(),
        }
    )
    save_json(output_dir / "run_records.json", [asdict(item) for item in records])
    save_json(output_dir / "metadata.json", metadata)
    logger.info("Finished. output_dir=%s stop_reason=%s", output_dir, stop_reason)


if __name__ == "__main__":
    main()
