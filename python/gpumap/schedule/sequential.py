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
        self._update_counter = 0

    def allocate(self, target_latency: float) -> tuple[int, int, int]:

        if not self.completeness[0]:  # insertion not complete
            return (self.max_uint32, 0, 0)

        if not self.completeness[1]:  # update not complete

            return (0, self.max_uint32, 0)

        return (0, 0, self.max_uint32)
