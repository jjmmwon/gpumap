from __future__ import annotations

from .gpumap import GPUMAP
from ._core import GPUMAPParams
from .schedule import (
    BaseStrategy,
    WeightedStrategy,
    SequentialPriorityStrategy,
    PhaseSpec,
    DEFAULT_PHASE_SPECS,
    PhaseBasedStrategy,
    Estimator,
    MeanRateEstimator,
    DefaultLinearEstimator,
    PersistentEstimator,
)

__all__ = [
    "GPUMAP",
    "GPUMAPParams",
    # strategies
    "BaseStrategy",
    "WeightedStrategy",
    "SequentialPriorityStrategy",
    "PhaseSpec",
    "DEFAULT_PHASE_SPECS",
    "PhaseBasedStrategy",
    # estimators
    "Estimator",
    "MeanRateEstimator",
    "DefaultLinearEstimator",
    "PersistentEstimator",
]
