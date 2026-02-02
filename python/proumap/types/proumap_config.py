from dataclasses import dataclass


@dataclass(frozen=True)
class ProUMAPConfig:
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

    n_neighbors: int = 15
    n_components: int = 2
    max_instances: int = 1000000
    metric: str = "euclidean"
    n_epochs: int = 200
    min_dist: float = 0.1
    init: str = "spectral"
    negative_sample_rate: int = 5
    verbose: bool = False
    random_state: int = 42

    # parameters for progressive algorithm
    batch_size_hint: int = 1024
