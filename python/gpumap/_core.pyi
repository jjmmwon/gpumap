from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

class GPUMAPParams:
    n_neighbors: int
    n_components: int
    min_dist: float
    n_epochs: int
    negative_sample_rate: int
    max_epoch: int
    a: float
    b: float
    alpha: float
    verbose: bool
    # Accepts/returns a metric name string ("euclidean", "cosine" --
    # umap-learn style); an int or gpumap.Metric (enum.IntEnum) also works
    # for the setter. See cpp/bindings/binding.cu for why this isn't a
    # py::enum_-bound type.
    metric: str
    n_instances: int
    n_features: int

    def __init__(self) -> None: ...

class GPUMAPResult:
    n_insertion_ops: int
    n_update_ops: int
    n_embedding_ops: int
    insertion_time: float
    update_time: float
    embedding_time: float
    insertion_completed: bool
    update_completed: bool
    embedding_completed: bool
    n_inserted_instances: int
    n_update_changed_edges: int
    embedding_pending_edges: int
    insertion_queue_size: int
    update_queue_size: int
    embedding_queue_size: int

GPUMAPIterationResult: TypeAlias = GPUMAPResult

class GPUMAP:
    def __init__(self, config: GPUMAPParams) -> None: ...
    def initialize(self, n_instances: int, n_features: int) -> None: ...
    def append_data(self, x: np.ndarray) -> None: ...
    def run(
        self,
        n_jobs_insertion: int,
        n_jobs_update: int,
        n_jobs_embedding: int,
    ) -> GPUMAPResult: ...
    def close(self) -> None: ...
    def get_embedding(self) -> Any: ...
    def get_neighbor_indices(self) -> Any: ...
    def get_neighbor_dists(self) -> Any: ...
