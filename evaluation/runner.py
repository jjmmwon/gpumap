"""Shared utilities for building and running GPUMAP models in evaluation scripts."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from gpumap.data_loader import DataLoader, NumpyDataLoader, PreloadDataLoader

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import psutil
except ImportError:
    psutil = None

from gpumap import (  # noqa: E402
    GPUMAP,
    GPUMAPParams,
    DefaultLinearEstimator,
    PersistentEstimator,
    SequentialPriorityStrategy,
    WeightedStrategy,
)


RAW_DATA_ROOT = Path(__file__).resolve().parent / "data"
SCALED_DATA_ROOT = Path(__file__).resolve().parent / "data_scaled"
# All evaluation entry points load StandardScaler-scaled data by default so that
# baselines (KNN ground truth, recall) are computed in the same space GPUMAP runs
# on. See evaluation/bench/build_scaled_data.py to (re)populate SCALED_DATA_ROOT.
DEFAULT_DATA_ROOT = SCALED_DATA_ROOT
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
class EstimatedTimes:
    insertion: float
    update: float
    embedding: float


@dataclass
class MemorySnapshot:
    gpu_used_mb: float
    gpu_total_mb: float
    cpu_rss_mb: float


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
    weights: tuple[float, float, float] | None = None,
):
    common_kwargs = {
        "n_instances": n_instances,
        "n_neighbors": n_neighbors,
        "insertion_estimator": insertion_estimator,
        "update_estimator": update_estimator,
        "embedding_estimator": embedding_estimator,
    }
    if strategy_name == "weighted":
        if weights is not None:
            common_kwargs["weights"] = weights
        return WeightedStrategy(**common_kwargs)
    if strategy_name == "sequential":
        return SequentialPriorityStrategy(**common_kwargs)
    raise ValueError(f"Unsupported strategy: {strategy_name}")


def _safe_predict_time(estimator: Any, ops: int, quantile: float) -> float:
    if ops <= 0:
        return 0.0
    try:
        return float(estimator.predict_time(int(ops), quantile))
    except ValueError:
        return 0.0


def plan_run(model: GPUMAP, target_latency: float) -> tuple[PlannedOps, EstimatedTimes]:
    strategy = model._strategy
    allocation = strategy.allocate(target_latency)

    planned_ops = PlannedOps(
        insertion=min(
            int(allocation[0]),
            model.data_loader.n_instances - model.n_inserted_instances,
        ),
        update=int(allocation[1]),
        embedding=int(allocation[2]),
    )
    estimated_times = EstimatedTimes(
        insertion=_safe_predict_time(
            strategy.insertion_estimator, planned_ops.insertion, strategy.confidence
        ),
        update=_safe_predict_time(
            strategy.update_estimator, planned_ops.update, strategy.confidence
        ),
        embedding=_safe_predict_time(
            strategy.embedding_estimator, planned_ops.embedding, strategy.confidence
        ),
    )
    return planned_ops, estimated_times


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
            pass

    cpu_rss_mb = -1.0
    if psutil is not None:
        try:
            cpu_rss_mb = float(
                psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            )
        except Exception:
            pass

    return MemorySnapshot(
        gpu_used_mb=gpu_used_mb,
        gpu_total_mb=gpu_total_mb,
        cpu_rss_mb=cpu_rss_mb,
    )


def close_estimator(estimator: Any) -> None:
    close = getattr(estimator, "close", None)
    if callable(close):
        close()


def is_finished(result: Any) -> bool:
    return result.embedding_completed
