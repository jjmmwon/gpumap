from typing import Optional

import numpy as np
import cupy as cp


def validate_data_chunk(
    data_chunk: np.ndarray | cp.ndarray, feature_dim: Optional[int]
) -> int:

    shape = data_chunk.shape

    if len(shape) != 2:
        raise ValueError("data_chunk must be 2D array")

    if feature_dim is not None and shape[1] != feature_dim:
        raise ValueError(
            f"data_chunk feature dimension {shape[1]} does not match "
            f"previous feature dimension {feature_dim}"
        )

    return shape[1]
