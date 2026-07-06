import numpy as np


def eval_emb_certainty(
    emb_knn_indices: np.ndarray, ground_truth_sets: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Compute fidelity of an embedding against ground-truth neighbor sets.

    Parameters
    ----------
    emb_knn_indices : (n, k)  — KNN indices in embedding space
    ground_truth_sets : (n, max_unique+1) — last column = per-point valid count

    Returns
    -------
    mean_accuracy : float
    per_point_accuracy : (n,) float32
    """
    n, k = emb_knn_indices.shape
    assert ground_truth_sets.shape[0] == n

    accuracy = np.zeros(n, dtype=np.float32)
    for i, gt in enumerate(ground_truth_sets):
        gt_valid = gt[: gt[-1]]
        accuracy[i] = np.isin(gt_valid, emb_knn_indices[i]).sum() / k

    return float(np.mean(accuracy)), accuracy
