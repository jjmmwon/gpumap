from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .estimator import Estimator, DefaultLinearEstimator, MeanRateEstimator
from .scheduler import Scheduler, DefaultScheduler


@dataclass
class OpsAllocation:
    n_insertion_ops: int
    n_update_ops: int
    n_embedding_ops: int


class BaseOpsPolicy(ABC):
    """
    Abstract policy that maps a target latency budget to an ops allocation.

    Implementations decide how to divide `target_latency` (seconds) among
    insertion, update, and embedding kernels each iteration.
    """

    @abstractmethod
    def allocate(self, target_latency: float) -> OpsAllocation:
        """Return n_ops for each kernel given the latency budget."""
        ...

    @abstractmethod
    def update(self, result) -> None:
        """Update internal state from the result of the last run() call."""
        ...

    @abstractmethod
    def on_insertion_complete(self) -> None:
        """Called once when all data has been inserted."""
        ...


class DefaultWeightedPolicy(BaseOpsPolicy):
    """
    Weighted time-budget split across all three kernels.

    The latency budget is split proportionally by weights (one per kernel),
    with MeanRateEstimator used for insertion and linear estimators for
    update/embedding.
    """

    def __init__(
        self,
        scheduler: Optional[Scheduler] = None,
        insertion_estimator: Optional[Estimator] = None,
        update_estimator: Optional[Estimator] = None,
        embedding_estimator: Optional[Estimator] = None,
        default_n_ops: int = 100,
    ):
        self.scheduler = scheduler or DefaultScheduler()
        self.insertion_estimator = insertion_estimator or MeanRateEstimator()
        self.update_estimator = update_estimator or DefaultLinearEstimator()
        self.embedding_estimator = embedding_estimator or DefaultLinearEstimator()
        self.default_n_ops = default_n_ops

    def allocate(self, target_latency: float) -> OpsAllocation:
        time_slices = self.scheduler.schedule(target_latency)

        try:
            n_ins = max(500, self.insertion_estimator.estimate_ops(time_slices[0]))
        except ValueError:
            n_ins = self._jitter(self.default_n_ops)

        try:
            n_upd = self.update_estimator.estimate_ops(time_slices[1])
        except ValueError:
            n_upd = self._jitter(self.default_n_ops)

        try:
            n_emb = self.embedding_estimator.estimate_ops(time_slices[2])
        except ValueError:
            n_emb = self._jitter(self.default_n_ops)

        return OpsAllocation(n_ins, n_upd, n_emb)

    def update(self, result) -> None:
        # Mirror original: add insertion sample only while still inserting.
        if not result.insertion_completed:
            self.insertion_estimator.add_sample(
                result.n_insertion_ops, result.insertion_time
            )
        self.update_estimator.add_sample(result.n_update_ops, result.update_time)
        self.embedding_estimator.add_sample(
            result.n_embedding_ops, result.embedding_time
        )

    def on_insertion_complete(self) -> None:
        self.scheduler.set_insertion_weight(0)

    def _jitter(self, base: int) -> int:
        lo = max(1, base // 10)
        return base + int(np.random.randint(-lo, lo))


class SequentialPriorityPolicy(BaseOpsPolicy):
    """
    Insertion → Update → Embedding priority cascade.

    The full latency budget is offered to insertion first. The predicted time
    for the estimated insertion ops is subtracted; the remainder flows to
    update, and what is left flows to embedding. Once insertion is complete
    its ops are zeroed and the full budget cascades to update then embedding.

    When an estimator has too few samples to produce an estimate it falls back
    to `default_n_ops` (with ±10 % jitter) so the system is never idle while
    warming up.
    """

    def __init__(
        self,
        insertion_estimator: Optional[Estimator] = None,
        update_estimator: Optional[Estimator] = None,
        embedding_estimator: Optional[Estimator] = None,
        default_n_ops: int = 100,
        confidence: float = 0.95,
    ):
        self.insertion_estimator = insertion_estimator or MeanRateEstimator()
        self.update_estimator = update_estimator or DefaultLinearEstimator()
        self.embedding_estimator = embedding_estimator or DefaultLinearEstimator()
        self.default_n_ops = default_n_ops
        self.confidence = confidence
        self._insertion_done = False

    def allocate(self, target_latency: float) -> OpsAllocation:
        remaining = target_latency

        # --- Insertion ---
        if not self._insertion_done:
            try:
                n_ins = max(
                    500,
                    self.insertion_estimator.estimate_ops(remaining, self.confidence),
                )
                remaining -= self.insertion_estimator.predict_time(
                    n_ins, self.confidence
                )
                remaining = max(0.0, remaining)
            except ValueError:
                n_ins = self._jitter(self.default_n_ops)
                # No reliable time prediction yet; leave full budget for later stages.
        else:
            n_ins = 0

        # --- Update ---
        if remaining > 0:
            try:
                n_upd = self.update_estimator.estimate_ops(remaining, self.confidence)
                remaining -= self.update_estimator.predict_time(n_upd, self.confidence)
                remaining = max(0.0, remaining)
            except ValueError:
                n_upd = self._jitter(self.default_n_ops)
        else:
            n_upd = 0

        # --- Embedding ---
        if remaining > 0:
            try:
                n_emb = self.embedding_estimator.estimate_ops(
                    remaining, self.confidence
                )
            except ValueError:
                n_emb = self._jitter(self.default_n_ops)
        else:
            n_emb = 0

        return OpsAllocation(n_ins, n_upd, n_emb)

    def update(self, result) -> None:
        if not self._insertion_done and result.n_insertion_ops > 0:
            self.insertion_estimator.add_sample(
                result.n_insertion_ops, result.insertion_time
            )
        self.update_estimator.add_sample(result.n_update_ops, result.update_time)
        self.embedding_estimator.add_sample(
            result.n_embedding_ops, result.embedding_time
        )

    def on_insertion_complete(self) -> None:
        self._insertion_done = True

    def _jitter(self, base: int) -> int:
        lo = max(1, base // 10)
        return base + int(np.random.randint(-lo, lo))
