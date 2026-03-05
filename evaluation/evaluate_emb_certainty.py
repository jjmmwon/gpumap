import numpy as np


def fill_unembedded_by_density(
    incomplete_embedding: np.ndarray,
    n_instances: int,
    seed: int = 0,
    grid_size: int = 128,
) -> np.ndarray:
    """
    Parameters
    ----------
    incomplete_embedding : (n_subsamples, 2)
    n_instances         : int  number of total instances (including unembedded ones)
    seed                : int  random seed for reproducibility
    grid_size           : int  grid resolution (grid_size × grid_size)

    Returns
    -------
    (n_instances, 2)
    """
    filled_embedding = np.full((n_instances, 2), np.nan, dtype=np.float32)
    n_embedded = incomplete_embedding.shape[0]
    filled_embedding[:n_embedded] = incomplete_embedding

    n_unembedded = n_instances - n_embedded
    if n_unembedded <= 0:
        return filled_embedding

    rng = np.random.default_rng(seed)

    # 1. Grid bounds
    mins = incomplete_embedding.min(axis=0)
    maxs = incomplete_embedding.max(axis=0)

    # 2. Count instances in each grid cell
    hist, xedges, yedges = np.histogram2d(
        incomplete_embedding[:, 0],
        incomplete_embedding[:, 1],
        bins=grid_size,
        range=[[mins[0], maxs[0]], [mins[1], maxs[1]]],
    )

    # 3. Turn counts into probabilities (for sampling)
    weights = hist.ravel()
    weights = weights / weights.sum()

    # 4. Sample grid cells according to density
    flat_idx = rng.choice(grid_size * grid_size, size=n_unembedded, p=weights)
    ix, iy = np.unravel_index(flat_idx, (grid_size, grid_size))

    # 5. 셀 center + Gaussian noise (std = cell 크기의 절반)
    cell_w = (maxs[0] - mins[0]) / grid_size
    cell_h = (maxs[1] - mins[1]) / grid_size
    cx = xedges[ix] + cell_w * 0.5
    cy = yedges[iy] + cell_h * 0.5
    cx += rng.normal(0.0, cell_w * 0.5, size=n_unembedded)
    cy += rng.normal(0.0, cell_h * 0.5, size=n_unembedded)

    filled_embedding[n_embedded:] = np.stack([cx, cy], axis=1).astype(np.float32)
    return filled_embedding


def eval_in_progress_emb_certainty(
    in_progress_embedding: np.ndarray, ground_truth: np.ndarray
):
    filled = fill_unembedded_by_density(in_progress_embedding, ground_truth.shape[0])
    return eval_emb_certainty(filled, ground_truth)


def eval_emb_certainty(
    emb_knn_indices: np.ndarray, ground_truth_sets: np.ndarray
) -> tuple[float, np.ndarray]:
    n, k = emb_knn_indices.shape

    assert ground_truth_sets.shape[0] == (n)

    accuracy = np.zeros(n, dtype=np.float32)

    for i, gt in enumerate(ground_truth_sets):
        gt_valid = gt[: gt[-1]]
        true_positives = np.isin(gt_valid, emb_knn_indices[i]).sum()
        accuracy[i] = true_positives / k

    return np.mean(accuracy), accuracy
