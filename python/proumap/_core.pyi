from __future__ import annotations
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class UMAPConfig:
    """
    Configuration parameters for the UMAP algorithm.

    Attributes:
        n_neighbors (int): Number of neighboring points used in UMAP.
        n_components (int): Dimensionality of the embedding.
        metric (str): Metric used to compute distances in the input space.
        n_epochs (int): Number of training epochs for UMAP.
        min_dist (float): Minimum distance between points in the embedding.
        init (str): Initialization method for UMAP embedding.
        negative_sample_rate (int): Number of negative samples per positive sample.
        verbose (bool): Whether to print progress messages.
        random_state (int): Seed for random number generator.
    """

    n_neighbors: int = 15
    n_components: int = 2
    # metric: str = "euclidean"
    min_dist: float = 0.1
    n_epochs: int = 200
    # init: str = "spectral"
    negative_sample_rate: int = 5
    verbose: bool = False
    # random_state: int = 42

@dataclass(frozen=True)
class ProUMAPConfig(UMAPConfig):
    """
    Configuration parameters for the ProUMAP algorithm.

    Attributes:
        n_neighbors (int): Number of neighboring points used in UMAP.
        n_components (int): Dimensionality of the embedding.
        max_instances (int): Maximum number of instances to process.
        metric (str): Metric used to compute distances in the input space.
        n_epochs (int): Number of training epochs for UMAP.
        min_dist (float): Minimum distance between points in the embedding.
        init (str): Initialization method for UMAP embedding.
        negative_sample_rate (int): Number of negative samples per positive sample.
        verbose (bool): Whether to print progress messages.
        random_state (int): Seed for random number generator.

        batch_size_hint (int): Hint for the batch size in the progressive algorithm.
    """

    # parameters for progressive algorithm
    max_instances: int = 1000000
    batch_size_hint: int = 1024
    n_features: int = 0

class ProUMAP:
    def __init__(self, config: ProUMAPConfig) -> None: ...
    def close(self) -> None: ...
    def append_data(self, x: np.ndarray) -> None: ...
    def run_nng_build(self) -> None: ...
    def run_optimize_embeddings(self) -> None: ...
