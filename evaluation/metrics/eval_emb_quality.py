import numpy as np
from zadu import ZADU


def evaluate_embedding_quality(embedding: np.ndarray, data: np.ndarray):
    """
    Evaluate the quality of an embedding using ZADU.

    Parameters:
    embedding (np.ndarray): The low-dimensional embedding (shape: [n_samples, n_components]).
    data (np.ndarray): The original high-dimensional data (shape: [n_samples, n_features]).

    Returns:
    float: The ZADU score indicating the quality of the embedding.
    """
    k = [15, 30, 50]  # Example values for k
    tnc_spec = [{"id": "tnc", "params": {"k": _k}} for _k in k]
    nh_spec = [{"id": "nh", "params": {"k": _k}} for _k in k]

    spec = tnc_spec + nh_spec

    zadu_obj = ZADU(spec, data)

    scores = zadu_obj.measure(embedding)

    results = [{"spec": spec_, "score": score_} for spec_, score_ in zip(spec, scores)]

    return results
