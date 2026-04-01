from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

from .estimator import Estimator, MeanRateEstimator

if TYPE_CHECKING:
    from .. import _core


def _jitter(base: int) -> int:
    lo = max(1, base // 10)
    return base + int(np.random.randint(-lo, lo))


def _estimate_or_fallback(
    estimator: Estimator,
    budget_sec: float,
    confidence: float,
    default_ops: int,
) -> int:
    if budget_sec <= 0.0:
        return 0
    try:
        ops = max(0, int(estimator.estimate_ops(budget_sec, confidence)))
        # If estimator returns 0 despite positive budget (e.g. due to corrupt DB
        # samples causing near-zero or negative slope), fall back to default so
        # that the caller always makes forward progress and never spin-loops.
        return ops if ops > 0 else _jitter(default_ops)
    except ValueError:
        return _jitter(default_ops)


def clip_ops(n_ops: int, ops_range: tuple[int, int]) -> int:
    return max(ops_range[0], min(n_ops, ops_range[1]))


class BaseStrategy(ABC):
    """
    Base class for all allocation strategies.

    Tracks insertion/update/embedding progress and splits a time budget
    proportionally by weights. Subclasses implement ``allocate()``.

    Progress values are in [0.0, 1.0]:
      - insertion_progress: inserted / total instances
      - update_progress:    1 - update_queue / inserted
      - embedding_progress: 1 - embedding_queue / (inserted * n_neighbors * 3)

    Parameters
    ----------
    weights:
        (insertion, update, embedding) relative weights. The insertion weight
        is set to 0 automatically when on_insertion_complete() is called.
    reserved:
        Fraction of budget kept as idle headroom to prevent 100% utilisation.
    min_insertion_ops:
        Floor applied to insertion n_ops while insertion is still active.
    """

    insertion_ops_range: tuple[int, int] = (0, 0)
    update_ops_range: tuple[int, int] = (0, 0)
    embedding_ops_range: tuple[int, int] = (0, 0)

    _n_instances: int = 0
    _n_inserted: int = 0

    insertion_progress: float = 0.0
    update_progress: float = 0.0
    embedding_progress: float = 0.0

    def __init__(
        self,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        reserved: float = 0.2,
        insertion_estimator: Optional[Estimator] = None,
        update_estimator: Optional[Estimator] = None,
        embedding_estimator: Optional[Estimator] = None,
        n_instances: int = 0,
        n_neighbors: int = 15,
        default_n_ops: int = 100,
        confidence: float = 0.95,
        min_insertion_ops: Optional[int] = None,
    ):
        self._weights = list(weights)
        self._reserved = reserved

        self.insertion_estimator = insertion_estimator or MeanRateEstimator()
        self.update_estimator = update_estimator or MeanRateEstimator()
        self.embedding_estimator = embedding_estimator or MeanRateEstimator()

        self._n_instances = n_instances
        self._n_neighbors = n_neighbors
        self.default_n_ops = default_n_ops
        self.confidence = confidence

        self.min_insertion_ops = min_insertion_ops or int(0.01 * n_instances)
        self.max_uint32 = 2**32 - 1

        self._result_history: list[_core.GPUMAPResult] = []

        self.completeness = [False, False, False]  # insertion, update, embedding

    @abstractmethod
    def allocate(self, target_latency: float) -> tuple[int, int, int]: ...

    def update(self, result: _core.GPUMAPResult) -> None:
        self._result_history.append(result)

        if not self.completeness[0] and result.n_insertion_ops > 0:
            self.insertion_estimator.add_sample(
                result.n_insertion_ops, result.insertion_time
            )
        if result.n_update_ops > 0:
            self.update_estimator.add_sample(result.n_update_ops, result.update_time)
        if result.n_embedding_ops > 0:
            self.embedding_estimator.add_sample(
                result.n_embedding_ops, result.embedding_time
            )
        self._track_progress(result)

    def on_operation_complete(self, operation: str) -> None:
        op_dict = {"insertion": 0, "update": 1, "embedding": 2}
        if operation not in op_dict:
            raise ValueError(f"Unknown operation: {operation}")
        self.completeness[op_dict[operation]] = True
        self._weights[op_dict[operation]] = 0.0

    def _track_progress(self, result: _core.GPUMAPResult) -> None:
        self._n_inserted = result.n_inserted_instances
        self.insertion_progress = (
            self._n_inserted / self._n_instances if self._n_instances > 0 else 0.0
        )
        self.update_progress = (
            1.0 - (result.update_queue_size / self._n_inserted)
            if self._n_inserted > 0
            else 0.0
        )
        embedding_pending_edges = result.embedding_pending_edges
        embedding_remaining_edges = result.embedding_queue_size
        self.embedding_progress = (
            1.0 - (embedding_remaining_edges / embedding_pending_edges)
            if embedding_pending_edges > 0
            else 0.0
        )

    def get_progress(self) -> tuple[float, float, float]:
        return (self.insertion_progress, self.update_progress, self.embedding_progress)

    def _split(self, total: float) -> tuple[float, float, float]:
        weight_sum = sum(self._weights)
        denom = weight_sum * (1 + self._reserved)
        # denom = sum(self._weights) + self._reserved
        if denom <= 0:
            return (0.0, 0.0, 0.0)
        a, b, c = (total * w / denom for w in self._weights)
        return (a, b, c)
