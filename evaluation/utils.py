import numpy as np
import cupy as cp
from sklearn.metrics import pairwise_distances


def calculate_pairwise_distances(data: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise distance np.matrix for the given data.

    Parameters
    ----------
    data : (n_samples, n_features)

    Returns
    -------
    (n_samples, n_samples) distance matrix
    """
    from cuml.metrics import pairwise_distances

    data_device = cp.asarray(data, dtype=cp.float32)

    pwd = pairwise_distances(data_device, metric="euclidean").astype(np.float32)
    return cp.asnumpy(pwd)


def compute_knn_indices_and_distances(data: np.ndarray, k: int):
    """
    Compute the k-nearest neighbors (indices and distances) for the given data.
    Parameters
    ----------
    data : (n_samples, n_features)
    k : int

    Returns
    -------
    indices : (n_samples, k) array of neighbor indices (excluding self-neighbor)
    distances : (n_samples, k) array of neighbor distances (excluding self-distance)
    """
    from cuml.neighbors import NearestNeighbors

    data_device = cp.asarray(data, dtype=cp.float32)

    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(data_device)
    distances, indices = knn.kneighbors(data_device, return_distance=True)

    return (
        cp.asnumpy(indices).astype(np.int32)[:, 1:],  # Exclude self-neighbor
        cp.asnumpy(distances).astype(np.float32)[:, 1:],  # Exclude self-distance
    )


def get_nn_infos(
    pairwise_distances: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the k-nearest neighbors with ranking for each point based on the pairwise distance matrix.

    Parameters
    ----------
    pairwise_distances : (n_samples, n_samples)
        The pairwise distance matrix.

    Returns
    -------
    sorted_indices : (n_samples, n_samples)
        Indices of neighbors sorted by distance for each point.
    sorted_distances : (n_samples, n_samples)
        Distances to neighbors sorted by distance for each point.
    ranking : (n_samples, n_samples)
        Ranking of neighbors for each point (0 for closest, 1 for second closest, etc.).
    """

    sorted_indices = np.argsort(pairwise_distances, axis=1)
    sorted_distances = np.take_along_axis(pairwise_distances, sorted_indices, axis=1)
    ranking = np.argsort(sorted_indices, axis=1)

    return sorted_indices, sorted_distances, ranking


def compute_umap_embedding(
    data: np.ndarray,
    precomputed_knn: tuple[np.ndarray, np.ndarray],
    n_components: int = 2,
) -> np.ndarray:
    """
    Compute a UMAP embedding for the given data.

    Parameters
    ----------
    precomputed_knn : tuple[np.ndarray, np.ndarray]
        A tuple containing the precomputed k-nearest neighbors (indices and distances).
    n_components : int, default=2
        The number of dimensions for the embedding.
    Returns
    -------
    (n_samples, n_components) UMAP embedding
    """
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


def compute_ground_truth_neighbors(
    exact_emb_neighbors: list[np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Compute the ground truth neighbors for each point based on the exact neighbors from multiple embeddings.
    Parameters
    ----------
    exact_emb_neighbors : list[np.ndarray]
        A list of arrays, where each array contains the exact neighbors for each point from a different
        embedding (shape: [n_samples, k]).
    k : int
        The number of neighbors to consider from each embedding.

    Returns
    -------
        (n_samples, k * n_embeddings + 1) array containing the unique ground truth
        neighbors for each point, along with the count of unique neighbors in the last column.

    """
    n_instances = exact_emb_neighbors[0].shape[0]
    n_runs = len(exact_emb_neighbors)

    # (n_instances, n_runs * k) — Python loop 없이 한 번에 concat
    all_neighbors = np.concatenate(exact_emb_neighbors, axis=1)

    result = np.full((n_instances, k * n_runs + 1), -1, dtype=np.int32)

    for i in range(n_instances):
        unique = np.unique(all_neighbors[i])  # C 레벨 sort + dedup
        result[i, : len(unique)] = unique
        result[i, -1] = len(unique)

    return result
