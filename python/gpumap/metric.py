"""Distance metric selector for GPUMAPParams.metric.

The primary way to set the metric is a plain string, umap-learn style:

    params.metric = "cosine"   # or "euclidean" (default)

`Metric` here is an optional, type-checked alternative for the same purpose
(`params.metric = Metric.Cosine`) -- a real enum.IntEnum on the pure-Python
side, values matching gpumap::Metric in cpp/include/gpumap/gpumap.cuh. Both
forms (and a plain int) are accepted by `_core.GPUMAPParams.metric`'s setter;
see cpp/bindings/binding.cu for why it isn't a py::enum_-bound type
(nvcc/pybind11 toolchain incompatibility, unrelated to Metric's definition).
"""

from __future__ import annotations

import enum


class Metric(enum.IntEnum):
    Euclidean = 0
    Cosine = 1
