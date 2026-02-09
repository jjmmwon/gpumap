import numpy as np


def _as_f32_c_contig_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("Input array must be 2D")
    if not x.flags["C_CONTIGUOUS"] or x.dtype != np.float32:
        x = np.ascontiguousarray(x, dtype=np.float32)

    return x
