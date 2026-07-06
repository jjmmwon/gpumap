import numpy as np
import cupy as cp


def compute_knn_indices_and_distances(data: np.ndarray, k: int):
    """
    Compute the k-nearest neighbors (indices and distances) for the given data.

    Returns
    -------
    indices : (n_samples, k) — self excluded
    distances : (n_samples, k) — self excluded
    """
    from cuml.neighbors import NearestNeighbors

    data_device = cp.asarray(data, dtype=cp.float32)
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(data_device)
    distances, indices = knn.kneighbors(data_device, return_distance=True)
    return (
        cp.asnumpy(indices).astype(np.int32)[:, 1:],
        cp.asnumpy(distances).astype(np.float32)[:, 1:],
    )


def compute_umap_embedding(
    data: np.ndarray,
    precomputed_knn: tuple[np.ndarray, np.ndarray],
    n_components: int = 2,
) -> np.ndarray:
    """Compute a UMAP embedding using precomputed KNN."""
    from cuml.manifold import UMAP

    knn_indices, knn_distances = precomputed_knn
    data_device = cp.asarray(data, dtype=cp.float32)
    umap = UMAP(
        n_neighbors=15,
        n_components=n_components,
        precomputed_knn=(knn_indices, knn_distances),
    )
    embedding_device = umap.fit_transform(data_device)
    return cp.asnumpy(embedding_device).astype(np.float32)
