from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from . import _core

# Import CuPy after loading gpumap._core to avoid mixing CUDA libs
# from different package managers (e.g., pip nvidia-* vs conda CUDA libs).
import cupy as cp


from gpumap.validators import validate_data_chunk
from gpumap.utils import _as_f32_c_contig_2d

from .schedule import Estimator, BaseStrategy, WeightedStrategy
from .data_loader import DataLoader


def find_ab_params(spread, min_dist):
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


class GPUMAP:
    def __init__(
        self,
        data_loader: DataLoader,
        params: _core.GPUMAPParams,
        strategy: BaseStrategy | None = None,
        insertion_estimator: Estimator | None = None,
        update_estimator: Estimator | None = None,
        embedding_estimator: Estimator | None = None,
        default_n_ops: int = 100,
    ):

        self.data_loader = data_loader
        self._params = params

        if self.params.min_dist > 1.0:
            raise ValueError("min_dist must be <= 1.0")
        self.params.a, self.params.b = find_ab_params(1.0, self.params.min_dist)

        try:
            self._impl = _core.GPUMAP(self.params)
        except MemoryError:
            # The first CUDA allocation via RAFT/RMM can transiently fail in
            # some environments (e.g. WSL2) before the CUDA runtime is fully
            # initialised. Retry once; it always succeeds on the second attempt.
            self._impl = _core.GPUMAP(self.params)
        self._closed = False

        if strategy is not None:
            self._strategy = strategy
        else:
            self._strategy = WeightedStrategy(
                n_instances=data_loader.n_instances,
                n_neighbors=params.n_neighbors,
                insertion_estimator=insertion_estimator,
                update_estimator=update_estimator,
                embedding_estimator=embedding_estimator,
                default_n_ops=default_n_ops,
            )

        self._inserted_instances = 0
        self._impl.initialize(self.data_loader.n_instances, self.data_loader.n_features)

    @property
    def params(self) -> _core.GPUMAPParams:
        return self._params

    @property
    def n_inserted_instances(self) -> int:
        return self._inserted_instances

    def close(self):
        if not self._closed:
            self._impl.close()
            self._closed = True

    def run_with_n_ops(
        self, n_insertion_ops: int, n_update_ops: int, n_embedding_ops: int
    ) -> _core.GPUMAPIterationResult:
        if n_insertion_ops > 0:
            data_chunk = self.data_loader.get_next_chunk(n_insertion_ops)
            try:
                validate_data_chunk(data_chunk, self._params.n_features)
                data_chunk = _as_f32_c_contig_2d(data_chunk)
            except ValueError as e:
                print(f"Error validating data chunk: {e}")

            self._impl.append_data(data_chunk)

        res = self._impl.run(n_insertion_ops, n_update_ops, n_embedding_ops)
        self._inserted_instances = res.n_inserted_instances

        self._strategy.update(res)
        if res.insertion_completed:
            self._strategy.on_operation_complete("insertion")
        if res.update_completed:
            self._strategy.on_operation_complete("update")
        if res.embedding_completed:
            self._strategy.on_operation_complete("embedding")

        return res

    def run(self, target_latency: float = 10.0) -> _core.GPUMAPIterationResult:
        allocation = self._strategy.allocate(target_latency)
        n_insertion_ops = min(
            allocation[0],
            self.data_loader.n_instances - self._inserted_instances,
        )
        n_update_ops = allocation[1]

        n_embedding_ops = allocation[2]

        if n_insertion_ops > 0:
            data_chunk = self.data_loader.get_next_chunk(n_insertion_ops)
            try:
                validate_data_chunk(data_chunk, self._params.n_features)
                data_chunk = _as_f32_c_contig_2d(data_chunk)
            except ValueError as e:
                print(f"Error validating data chunk: {e}")

            self._impl.append_data(data_chunk)

        res = self._impl.run(n_insertion_ops, n_update_ops, n_embedding_ops)

        self._inserted_instances = res.n_inserted_instances

        self._strategy.update(res)
        if res.insertion_completed:
            self._strategy.on_operation_complete("insertion")
        if res.update_completed:
            self._strategy.on_operation_complete("update")
        if res.embedding_completed:
            self._strategy.on_operation_complete("embedding")

        return res

    def get_embedding(self) -> cp.ndarray:
        embedding_view = cp.asarray(self._impl.get_embedding())
        return embedding_view[: self._inserted_instances]

    def get_neighbor_indices(self) -> cp.ndarray:
        neighbor_indices_view = cp.asarray(self._impl.get_neighbor_indices())
        return neighbor_indices_view[
            : self._inserted_instances, : self._params.n_neighbors
        ]

    def get_neighbor_dists(self) -> cp.ndarray:
        neighbor_dists_view = cp.asarray(self._impl.get_neighbor_dists())
        return neighbor_dists_view[
            : self._inserted_instances, : self._params.n_neighbors
        ]
