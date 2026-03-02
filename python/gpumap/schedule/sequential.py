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
        remaining = max(0.0, target_latency)
        clipped_ins = 0
        clipped_upd = 0
        clipped_emb = 0

        if not self._insertion_done:
            n_ins = _estimate_or_fallback(
                self.insertion_estimator, remaining, self.confidence, self.default_n_ops
            )
            clipped_ins = clip_ops(n_ins, self.insertion_ops_range)
            try:
                predicted_ins_time = self.insertion_estimator.predict_time(
                    clipped_ins, self.confidence
                )
            except ValueError:
                predicted_ins_time = 0.0
            remaining = max(0.0, remaining - predicted_ins_time)

        if self.update_progress < 1.0:
            n_upd = _estimate_or_fallback(
                self.update_estimator, remaining, self.confidence, self.default_n_ops
            )
            clipped_upd = clip_ops(n_upd, self.update_ops_range)
            try:
                predicted_upd_time = self.update_estimator.predict_time(
                    clipped_upd, self.confidence
                )
            except ValueError:
                predicted_upd_time = 0.0
            remaining = max(0.0, remaining - predicted_upd_time)

        if self.embedding_progress < 1.0:
            # Guard against update eating the entire remaining budget: always
            # give embedding at least 10 % of target_latency so the queue can
            # drain even when the update estimator over-predicts.
            emb_budget = remaining if remaining > 0 else target_latency * 0.1
            n_emb = _estimate_or_fallback(
                self.embedding_estimator, emb_budget, self.confidence, self.default_n_ops
            )
            clipped_emb = clip_ops(n_emb, self.embedding_ops_range)

        return (clipped_ins, clipped_upd, clipped_emb)
