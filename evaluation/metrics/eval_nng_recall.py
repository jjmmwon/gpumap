import numpy as np
import cupy as cp
from cuml.neighbors import NearestNeighbors


def _validate_input(data: np.ndarray, nng: np.ndarray):
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if nng.ndim != 2:
        raise ValueError("NNG must be a 2D array")


def evaluate_nng_recall(data: np.ndarray, nng: np.ndarray) -> float:
    """
    Compute recall for a single embedding.

    Parameters:
    data (np.ndarray): The input data (shape: [n_samples, n_features]).
    nng (np.ndarray): The nearest neighbor graph (shape: [n_samples, n_neighbors]).

    Returns:
    float: The recall value.
    """
    _validate_input(data, nng)

    n_samples, k = nng.shape

    X = cp.asarray(data[:n_samples], dtype=cp.float32)
    pred = cp.asarray(nng, dtype=cp.int32)

    exact = (
        NearestNeighbors(n_neighbors=k + 1, algorithm="brute")
        .fit(X)
        .kneighbors(X, return_distance=False)
    )[:, 1:].astype(
        cp.int32
    )  # Exclude self-neighbor

    base = cp.int64(n_samples)
    rows = cp.arange(n_samples, dtype=cp.int64)[:, None]

    pred_keys = rows * base + pred.astype(cp.int64)
    exact_keys = rows * base + exact.astype(cp.int64)

    hit = cp.isin(pred_keys, exact_keys)

    return float(hit.mean())
