from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .estimator import Estimator, MeanRateEstimator
from .strategy import BaseStrategy, _estimate_or_fallback, clip_ops


@dataclass
class PhaseSpec:
    """
    Defines the ops weight distribution for a single phase.

    A phase becomes active when ``insertion_progress`` reaches ``threshold``.
    The strategy always uses the spec with the highest threshold that has been
    reached, so specs must cover [0.0, 1.0] by including at least one entry
    with ``threshold=0.0``.

    Attributes
    ----------
    threshold:
        Minimum ``insertion_progress`` value (in [0.0, 1.0]) required to
        enter this phase.
    weights:
        Relative time budget weights ``(insertion, update, embedding)`` for
        this phase. The insertion weight is automatically zeroed out once
        ``on_insertion_complete()`` is called, regardless of what is set here.
    """

    threshold: float
    weights: tuple[float, float, float]


DEFAULT_PHASE_SPECS: list[PhaseSpec] = [
    # Phase 0 — early: insertion dominates, embedding not yet meaningful
    PhaseSpec(threshold=0.0, weights=(2.5, 0.5, 0.0)),
    # Phase 1 — mid: insertion and update share budget, embedding warms up
    PhaseSpec(threshold=0.3, weights=(1.0, 1.0, 1.0)),
    # Phase 2 — late: insertion almost done, shift toward update + embedding
    PhaseSpec(threshold=0.6, weights=(0.5, 0.5, 2.0)),
]


class PhaseBasedStrategy(BaseStrategy):
    """
    Adjusts the (insertion, update, embedding) weight split based on how far
    insertion has progressed.

    Phases are defined as a sorted list of :class:`PhaseSpec` objects, each
    specifying a progress threshold and corresponding weights. The strategy
    picks the spec with the highest threshold that ``insertion_progress`` has
    reached, so the budget allocation shifts automatically as data flows in.

    Parameters
    ----------
    n_instances:
        Total number of instances that will be inserted.
    n_neighbors:
        k-NN graph degree (used to set ops range ceilings).
    phases:
        Ordered phase definitions. If *None*, :data:`DEFAULT_PHASE_SPECS` is
        used. Must contain at least one entry with ``threshold=0.0`` so that
        a phase is always active. The list is sorted by threshold internally.
    reserved:
        Fraction of the total budget kept as idle headroom.
    insertion_estimator, update_estimator, embedding_estimator:
        Rate estimators for each operation type. Defaults to
        :class:`MeanRateEstimator` when not provided.
    default_n_ops:
        Fallback ops count (±10 % jitter) used while estimators are
        warming up.
    confidence:
        Percentile used by estimators when predicting durations.
    min_insertion_ops:
        Minimum ops floor applied to insertion while it is still active.
    """

    def __init__(
        self,
        n_instances: int,
        n_neighbors: int = 15,
        phases: Optional[list[PhaseSpec]] = None,
        reserved: float = 0.1,
        insertion_estimator: Optional[Estimator] = None,
        update_estimator: Optional[Estimator] = None,
        embedding_estimator: Optional[Estimator] = None,
        default_n_ops: int = 100,
        confidence: float = 0.95,
        min_insertion_ops: int = 500,
    ):
        resolved = sorted(
            phases if phases is not None else DEFAULT_PHASE_SPECS,
            key=lambda p: p.threshold,
        )
        if not resolved or resolved[0].threshold > 0.0:
            raise ValueError(
                "phases must contain at least one PhaseSpec with threshold=0.0 "
                "so that a phase is always active."
            )
        self._phases = resolved

        super().__init__(
            weights=self._phases[0].weights,
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

    def _active_phase(self) -> PhaseSpec:
        """Return the spec with the highest threshold that has been reached."""
        active = self._phases[0]
        for spec in self._phases:
            if self.insertion_progress >= spec.threshold:
                active = spec
        return active

    def allocate(self, target_latency: float) -> tuple[int, int, int]:
        phase = self._active_phase()
        self._weights = list(phase.weights)
        if self._insertion_done:
            self._weights[0] = 0.0

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
