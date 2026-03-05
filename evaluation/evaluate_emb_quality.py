import numpy as np


def _lookup_ranks_topK(
    ref_topK: np.ndarray,
    query_knn: np.ndarray,
    fallback_rank: int,
) -> np.ndarray:
    """Row-wise rank lookup in top-K list. Missing entries -> fallback_rank."""
    n, k = query_knn.shape
    ranks = np.full((n, k), fallback_rank, dtype=np.int32)
    for i in range(n):
        rank_map = {int(idx): rank for rank, idx in enumerate(ref_topK[i], start=1)}
        ranks[i] = [rank_map.get(int(idx), fallback_rank) for idx in query_knn[i]]
    return ranks


def approx_trustworthiness_continuity(
    orig_nn_indices: np.ndarray,
    emb_nn_indices: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """
    n×n ranking matrix 없이 근사 Trustworthiness/Continuity를 계산한다.

    top-K 내에서만 rank lookup을 수행하고, 범위 밖은 rank=K+1로 간주한다.
    입력 nn_indices는 self를 포함하지 않는다고 가정한다.
    """

    n = orig_nn_indices.shape[0]
    max_K = min(orig_nn_indices.shape[1], emb_nn_indices.shape[1])

    if (2 * n - 3 * k - 1) <= 0:
        raise ValueError(
            f"Invalid (n, k) for normalization: n={n}, k={k}, "
            "requires (2*n - 3*k - 1) > 0."
        )

    emb_knn = emb_nn_indices[:, :k]  # (n, k)
    orig_knn = orig_nn_indices[:, :k]  # (n, k)
    orig_topK = orig_nn_indices
    emb_topK = emb_nn_indices

    fallback_rank = max_K + 1

    # --- Trustworthiness ---
    orig_rank_of_emb_knn = _lookup_ranks_topK(orig_topK, emb_knn, fallback_rank)
    intrusions = orig_rank_of_emb_knn > k
    trust_penalty = (orig_rank_of_emb_knn[intrusions] - k).sum(dtype=np.float64)

    # --- Continuity ---
    emb_rank_of_orig_knn = _lookup_ranks_topK(emb_topK, orig_knn, fallback_rank)
    extrusions = emb_rank_of_orig_knn > k
    cont_penalty = (emb_rank_of_orig_knn[extrusions] - k).sum(dtype=np.float64)

    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return float(1.0 - norm * trust_penalty), float(1.0 - norm * cont_penalty)


def lcmc(
    orig_nn_indices: np.ndarray,
    emb_nn_indices: np.ndarray,
    k: int,
) -> float:
    """
    LCMC(k) = Q_NX(k) - k/(n-1)
    Q_NX(k) = (1/(n*k)) * sum_i |N_k^orig(i) ∩ N_k^emb(i)|
    """
    n = orig_nn_indices.shape[0]

    orig_knn = orig_nn_indices[:, :k]
    emb_knn = emb_nn_indices[:, :k]

    # matches[i, r, c] = (orig_knn[i, r] == emb_knn[i, c]) -> (n, k, k)
    matches = orig_knn[:, :, None] == emb_knn[:, None, :]
    intersections = matches.any(axis=2).sum(axis=1)  # (n,)

    q_nx = float(intersections.sum(dtype=np.float64)) / (n * k)
    return float(q_nx - (k / (n - 1)))


def evaluate_TNC(
    orig_nn_indices: np.ndarray,
    emb_nn_indices: np.ndarray,
    k_list: list[int] = [15, 30, 50],
) -> list[dict]:
    """
    여러 k에 대해 Trustworthiness / Continuity를 계산한다.

    approx=True이면 top-K 기반 근사 계산을 사용한다.
    """

    results = []
    for _k in k_list:
        t, c = approx_trustworthiness_continuity(orig_nn_indices, emb_nn_indices, _k)
        results.append({"k": _k, "trustworthiness": t, "continuity": c})
    return results


def evaluate_LCMC(
    orig_nn_indices: np.ndarray,
    emb_nn_indices: np.ndarray,
    k_list: list[int] = [15, 30, 50],
) -> list[dict]:
    """
    여러 k에 대해 LCMC를 계산한다.

    Returns
    -------
    [{"k": k, "lcmc": float}, ...]
    """
    return [
        {"k": _k, "lcmc": lcmc(orig_nn_indices, emb_nn_indices, _k)} for _k in k_list
    ]
