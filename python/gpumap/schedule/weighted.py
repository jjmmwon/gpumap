from __future__ import annotations

from typing import Literal, Optional

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


class FocusedStrategy(WeightedStrategy):
    """
    Allocates a fixed fraction of the budget to one phase, with the remainder
    split among the other two phases by their relative weights.

    `focus` selects the prioritized phase: "insertion", "update", or "embedding".
    `focus_rate` sets what fraction of total time goes to that phase (0 < focus_rate < 1).
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
        focus: Literal["insertion", "update", "embedding"] = "insertion",
        focus_rate: float = 0.5,
        reserved: float = 0.1,
    ):
        super().__init__(
            insertion_estimator=insertion_estimator or MeanRateEstimator(),
            update_estimator=update_estimator or MeanRateEstimator(),
            embedding_estimator=embedding_estimator or MeanRateEstimator(),
            n_instances=n_instances,
            n_neighbors=n_neighbors,
            default_n_ops=default_n_ops,
            confidence=confidence,
            reserved=reserved,
        )
        if not (0.0 < focus_rate < 1.0):
            raise ValueError("focus_rate must be in the range (0, 1)")

        focus_index = {"insertion": 0, "update": 1, "embedding": 2}[focus]
        other_weight = (1.0 - focus_rate) / 2

        self._weights = [other_weight] * 3
        self._weights[focus_index] = focus_rate
