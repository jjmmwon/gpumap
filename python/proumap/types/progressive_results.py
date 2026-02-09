from dataclasses import dataclass
from typing import Optional
import numpy as np
import cupy as cp


@dataclass
class ProgressiveResults:
    """
    Data class to hold the results of the progressive UMAP algorithm.

    Attributes:
        embeddings (list): List of embeddings at each progressive step.
        n_steps (int): Number of progressive steps taken.
    """

    step: int
    metadata: dict
    embeddings: Optional[np.ndarray | cp.ndarray] = None
