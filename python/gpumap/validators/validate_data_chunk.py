import numpy as np


def validate_data_chunk(data_chunk: np.ndarray, feature_dim: int) -> int:
    shape = data_chunk.shape

    if len(shape) != 2:
        raise ValueError("data_chunk must be 2D array")

    if feature_dim is not None and shape[1] != feature_dim:
        raise ValueError(
            f"data_chunk feature dimension {shape[1]} does not match "
            f"previous feature dimension {feature_dim}"
        )

    return shape[1]
