from __future__ import annotations

from typing import Optional

from .estimator import Estimator, MeanRateEstimator
from .strategy import BaseStrategy, _estimate_or_fallback, clip_ops


class SequentialPriorityStrategy(BaseStrategy):
    """
    Insertion → Update → Embedding cascade.

    The full budget is offered to insertion first. The predicted insertion time
    is subtracted; the remainder cascades to update then embedding. Falls back
    to `default_n_ops` (±10 % jitter) while estimators are warming up.
    """

    def __init__(
        self,
        n_instances: int,
        n_neighbors: int = 15,
        insertion_estimator: Optional[Estimator] = None,
        update_estimator: Optional[Estimator] = None,
        embedding_estimator: Optional[Estimator] = None,
        default_n_ops: int = 100,
        confidence: float = 0.95,
        min_insertion_ops: int = 500,
    ):
        super().__init__(
            weights=(1.0, 1.0, 1.0),
            reserved=0.1,
            insertion_estimator=insertion_estimator or MeanRateEstimator(),
            update_estimator=update_estimator or MeanRateEstimator(),
            embedding_estimator=embedding_estimator or MeanRateEstimator(),
            n_instances=n_instances,
            n_neighbors=n_neighbors,
            default_n_ops=default_n_ops,
            confidence=confidence,
            min_insertion_ops=min_insertion_ops,
        )

    def allocate(self, target_latency: float) -> tuple[int, int, int]:

        if not self.completeness[0]:  # insertion not complete
            n_ins = _estimate_or_fallback(
                self.insertion_estimator,
                target_latency,
                self.confidence,
                self.default_n_ops,
            )
            clipped_ins = clip_ops(n_ins, (self.min_insertion_ops, self.max_uint32))

            return (clipped_ins, 0, 0)

        if not self.completeness[1]:  # update not complete
            n_upd = _estimate_or_fallback(
                self.update_estimator,
                target_latency,
                self.confidence,
                self.default_n_ops,
            )
            clipped_upd = min(n_upd, self.max_uint32)
            self._update_counter += 1

            return (0, clipped_upd, 0)

        n_emb = _estimate_or_fallback(
            self.embedding_estimator,
            target_latency,
            self.confidence,
            self.default_n_ops,
        )
        clipped_emb = min(n_emb, self.max_uint32)

        return (0, 0, clipped_emb)
