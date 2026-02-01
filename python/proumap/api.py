from __future__ import annotations

from typing import Any
import numpy as np
import cupy as cp

from . import _backend


def print_device_matrix_f32(x: Any, *, max_rows: int = 4, max_cols: int = 8) -> None:

    if hasattr(x, "__cuda_array_interface__"):
        x_d = x
    else:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        x_d = cp.asarray(x)

    if x_d.dtype != cp.float32:
        x_d = x_d.astype(cp.float32, copy=False)
    if x_d.ndim != 2:
        raise ValueError("x must be 2D")
    if not x_d.flags.c_contiguous:
        x_d = cp.ascontiguousarray(x_d)

    _backend.print_device_matrix_f32(x_d, int(max_rows), int(max_cols))


def add_f32(a: Any, b: Any) -> cp.ndarray:

    if hasattr(a, "__cuda_array_interface__"):
        a_d = a
    else:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim != 1:
            raise ValueError("a must be 1D")
        a_d = cp.asarray(a)

    if hasattr(b, "__cuda_array_interface__"):
        b_d = b
    else:
        b = np.asarray(b, dtype=np.float32)
        if b.ndim != 1:
            raise ValueError("b must be 1D")
        b_d = cp.asarray(b)

    if a_d.dtype != cp.float32:
        a_d = a_d.astype(cp.float32, copy=False)
    if b_d.dtype != cp.float32:
        b_d = b_d.astype(cp.float32, copy=False)

    if a_d.ndim != 1 or b_d.ndim != 1:
        raise ValueError("a, b must be 1D")
    if a_d.size != b_d.size:
        raise ValueError("a, b must have same length")

    if not a_d.flags.c_contiguous:
        a_d = cp.ascontiguousarray(a_d)
    if not b_d.flags.c_contiguous:
        b_d = cp.ascontiguousarray(b_d)

    out = cp.empty_like(a_d)
    _backend.add_f32(a_d, b_d, out)
    return out
