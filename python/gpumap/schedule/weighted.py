from __future__ import annotations

from typing import Optional

from .estimator import Estimator, MeanRateEstimator
from .strategy import BaseStrategy, _estimate_or_fallback, clip_ops


class WeightedStrategy(BaseStrategy):
    """
    A simple default strategy using StrategyInterface with MeanRateEstimators and
    fixed ops ranges. The insertion ops range has a non-zero floor to encourage
    steady progress on insertion, which is important for overall convergence.
    """

    def __init__(
        self,
        n_instances: int,
        n_neighbors: int = 15,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        reserved: float = 0.1,
        insertion_estimator: Optional[Estimator] = None,
        update_estimator: Optional[Estimator] = None,
        embedding_estimator: Optional[Estimator] = None,
        default_n_ops: int = 100,
        confidence: float = 0.95,
        min_insertion_ops: int = 500,
    ):
        super().__init__(
            weights=weights,
            reserved=reserved,
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
        ins_t, upd_t, emb_t = self._split(max(0.0, target_latency))

        n_ins = _estimate_or_fallback(
            self.insertion_estimator, ins_t, self.confidence, self.default_n_ops
        )
        n_upd = _estimate_or_fallback(
            self.update_estimator, upd_t, self.confidence, self.default_n_ops
        )
        n_emb = _estimate_or_fallback(
            self.embedding_estimator, emb_t, self.confidence, self.default_n_ops
        )

        return (
            clip_ops(n_ins, self.insertion_ops_range),
            clip_ops(n_upd, self.update_ops_range),
            clip_ops(n_emb, self.embedding_ops_range),
        )


class FixedInsertionStrategy(WeightedStrategy):
    """
    Fixed insertion ops until all instances are inserted, then switch to
    weighted allocation for update and embedding.

    This is a simple strategy that ensures steady progress on insertion, which is
    important for overall convergence. The insertion ops can be tuned to balance
    insertion speed with responsiveness of update/embedding phases.
    """

    def __init__(
        self,
        n_instances: int,
        n_neighbors: int = 15,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        reserved: float = 0.1,
        insertion_estimator: Optional[Estimator] = None,
        update_estimator: Optional[Estimator] = None,
        embedding_estimator: Optional[Estimator] = None,
        default_n_ops: int = 100,
        confidence: float = 0.95,
        fixed_insertion_rate: float = 0.5,
    ):
        super().__init__(
            weights=weights,
            reserved=reserved,
            insertion_estimator=insertion_estimator or MeanRateEstimator(),
            update_estimator=update_estimator or MeanRateEstimator(),
            embedding_estimator=embedding_estimator or MeanRateEstimator(),
            n_instances=n_instances,
            n_neighbors=n_neighbors,
            default_n_ops=default_n_ops,
            confidence=confidence,
        )
        self._weights[0] = (
            (self._weights[1] + self._weights[2])
            * fixed_insertion_rate
            / (1 - fixed_insertion_rate)
        )
