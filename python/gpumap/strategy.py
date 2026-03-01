from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from . import _core
from .estimator import DefaultLinearEstimator, Estimator, MeanRateEstimator


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
        return max(0, int(estimator.estimate_ops(budget_sec, confidence)))
    except ValueError:
        return _jitter(default_ops)


class BaseOpsStrategy(ABC):
    """Maps a target latency budget to an ops allocation each iteration."""

    @abstractmethod
    def allocate(self, target_latency: float) -> tuple[int, int, int]: ...

    @abstractmethod
    def update(self, result: _core.GPUMAPResult) -> None: ...

    @abstractmethod
    def on_insertion_complete(self) -> None: ...

    def set_total_instances(self, n: int) -> None:
        """Optional hook — called once with the total dataset size."""
        pass


class BaseProgStrategy(ABC):
    """
    Progress values are in [0.0, 1.0]:
      - insertion_progress: inserted / total instances
      - update_progress:    1 - update_queue / inserted  (converges toward 1)
      - embedding_progress: 1 - embedding_queue / (inserted * n_neighbors * 3)
    """

    _n_instances: int = 0
    _n_inserted: int = 0

    insertion_progress: float = 0.0
    update_progress: float = 0.0
    embedding_progress: float = 0.0

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
        embedding_remaining_edges = sum(result.embedding_queue_level_sizes)
        self.embedding_progress = (
            1.0 - (embedding_remaining_edges / embedding_pending_edges)
            if embedding_pending_edges > 0
            else 0.0
        )

    def get_progress(self) -> tuple[float, float, float]:
        return (
            self.insertion_progress,
            self.update_progress,
            self.embedding_progress,
        )


class DefaultWeightedStrategy(BaseProgStrategy, BaseOpsStrategy):
    """
    Splits the time budget proportionally by weights across three phases.

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

    def __init__(
        self,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        reserved: float = 0.1,
        insertion_estimator: Optional[Estimator] = None,
        update_estimator: Optional[Estimator] = None,
        embedding_estimator: Optional[Estimator] = None,
        n_instances: int = 0,
        default_n_ops: int = 100,
        confidence: float = 0.95,
        min_insertion_ops: int = 500,
    ):
        self._weights = list(weights)
        self._reserved = reserved

        self.insertion_estimator = insertion_estimator or MeanRateEstimator()
        self.update_estimator = update_estimator or MeanRateEstimator()
        self.embedding_estimator = embedding_estimator or MeanRateEstimator()

        self._n_instances = n_instances
        self.default_n_ops = default_n_ops
        self.confidence = confidence
        self.min_insertion_ops = min_insertion_ops
        self._insertion_done = False
        self._result_history: list[_core.GPUMAPResult] = []

    def _split(self, total: float) -> tuple[float, float, float]:
        denom = sum(self._weights) + self._reserved
        if denom <= 0:
            return (0.0, 0.0, 0.0)
        a, b, c = (total * w / denom for w in self._weights)
        return (a, b, c)

    def allocate(self, target_latency: float) -> tuple[int, int, int]:
        ins_t, upd_t, emb_t = self._split(max(0.0, target_latency))

        n_ins = _estimate_or_fallback(
            self.insertion_estimator, ins_t, self.confidence, self.default_n_ops
        )
        if not self._insertion_done:
            n_ins = max(n_ins, self.min_insertion_ops)

        n_upd = _estimate_or_fallback(
            self.update_estimator, upd_t, self.confidence, self.default_n_ops
        )
        n_emb = _estimate_or_fallback(
            self.embedding_estimator, emb_t, self.confidence, self.default_n_ops
        )
        return (n_ins, n_upd, n_emb)

    def update(self, result: _core.GPUMAPResult) -> None:
        self._result_history.append(result)

        if not self._insertion_done and result.n_insertion_ops > 0:
            self.insertion_estimator.add_sample(
                result.n_insertion_ops, result.insertion_time
            )
        self.update_estimator.add_sample(result.n_update_ops, result.update_time)
        self.embedding_estimator.add_sample(
            result.n_embedding_ops, result.embedding_time
        )
        self._track_progress(result)

    def on_insertion_complete(self) -> None:
        self._insertion_done = True
        self._weights[0] = 0.0


class SequentialPriorityStrategy(BaseProgStrategy, BaseOpsStrategy):
    """
    Insertion → Update → Embedding cascade.

    The full budget is offered to insertion first. The predicted insertion time
    is subtracted; the remainder cascades to update then embedding. Falls back
    to `default_n_ops` (±10 % jitter) while estimators are warming up.
    """

    def __init__(
        self,
        insertion_estimator: Optional[Estimator] = None,
        update_estimator: Optional[Estimator] = None,
        embedding_estimator: Optional[Estimator] = None,
        default_n_ops: int = 100,
        confidence: float = 0.95,
        min_insertion_ops: int = 500,
    ):
        self.insertion_estimator = insertion_estimator or MeanRateEstimator()
        self.update_estimator = update_estimator or DefaultLinearEstimator()
        self.embedding_estimator = embedding_estimator or DefaultLinearEstimator()
        self.default_n_ops = default_n_ops
        self.confidence = confidence
        self.min_insertion_ops = min_insertion_ops
        self._insertion_done = False

    def allocate(self, target_latency: float) -> tuple[int, int, int]:
        remaining = max(0.0, target_latency)

        if not self._insertion_done:
            try:
                n_ins = max(
                    self.min_insertion_ops,
                    self.insertion_estimator.estimate_ops(remaining, self.confidence),
                )
                remaining = max(
                    0.0,
                    remaining
                    - self.insertion_estimator.predict_time(n_ins, self.confidence),
                )
            except ValueError:
                n_ins = _jitter(self.default_n_ops)
        else:
            n_ins = 0

        try:
            n_upd = self.update_estimator.estimate_ops(remaining, self.confidence)
            remaining = max(
                0.0,
                remaining - self.update_estimator.predict_time(n_upd, self.confidence),
            )
        except ValueError:
            n_upd = _jitter(self.default_n_ops) if remaining > 0 else 0

        try:
            n_emb = self.embedding_estimator.estimate_ops(remaining, self.confidence)
        except ValueError:
            n_emb = _jitter(self.default_n_ops) if remaining > 0 else 0

        return (n_ins, n_upd, n_emb)

    def update(self, result: _core.GPUMAPResult) -> None:
        if not self._insertion_done and result.n_insertion_ops > 0:
            self.insertion_estimator.add_sample(
                result.n_insertion_ops, result.insertion_time
            )
        self.update_estimator.add_sample(result.n_update_ops, result.update_time)
        self.embedding_estimator.add_sample(
            result.n_embedding_ops, result.embedding_time
        )
        self._track_progress(result)

    def on_insertion_complete(self) -> None:
        self._insertion_done = True
