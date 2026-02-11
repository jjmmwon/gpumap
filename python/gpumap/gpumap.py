from __future__ import annotations

import numpy as np
import cupy as cp

from gpumap.validators import validate_data_chunk
from gpumap.utils import _as_f32_c_contig_2d

from . import _core
from .scheduler import Scheduler, DefaultScheduler
from .estimator import Estimator, DefaultLinearEstimator
from .data_loader import DataLoader

class GPUMAP:
    def __init__(self, data_loader: DataLoader,
                 config: _core.GPUMAPParams, scheduler: Scheduler = DefaultScheduler(),
                 insertion_estimator: Estimator = DefaultLinearEstimator(),
                 update_estimator: Estimator = DefaultLinearEstimator(),
                 embedding_estimator: Estimator = DefaultLinearEstimator(),
                 default_n_ops:int = 500):

        self.data_loader = data_loader
        self._config = config
        self.scheduler = scheduler
        self._impl = _core.GPUMAP(self.config)
        self._closed = False

        self.insertion_estimator = insertion_estimator
        self.update_estimator = update_estimator
        self.embedding_estimator = embedding_estimator

        self.default_n_ops = default_n_ops

        # TODO
        self._impl.initialize(self.data_loader.n_instances, self.data_loader.n_features)

    @property
    def config(self) -> _core.GPUMAPParams:
        return self._config

    def close(self):
        if not self._closed:
            # TODO
            self._impl.close()
            self._closed = True

    def run_with_n_ops(self, n_insertion_ops: int, n_update_ops: int, n_embedding_ops: int) -> _core.GPUMAPIterationResult:
        if n_insertion_ops > 0:
            data_chunk = self.data_loader.get_next_chunk(n_insertion_ops)
            try:
                validate_data_chunk(data_chunk, self._config.n_features)
                data_chunk = _as_f32_c_contig_2d(data_chunk)
            except ValueError as e:
                print(f"Error validating data chunk: {e}")

            # TODO
            self._impl.append_data(data_chunk)

        res = self._impl.run(n_insertion_ops, n_update_ops, n_embedding_ops)
        return res

    def run(self, target_latency: float = 10.0) -> _core.GPUMAPIterationResult:
        time_slices = self.scheduler.schedule(target_latency)

        try:
            n_insertion_ops = self.insertion_estimator.estimate_ops(time_slices[0])
        except ValueError:
            n_insertion_ops = self.default_n_ops + np.random.randint(-self.default_n_ops//10, self.default_n_ops//10)

        try:
            n_update_ops = self.update_estimator.estimate_ops(time_slices[1])
        except ValueError:
            n_update_ops = self.default_n_ops + np.random.randint(-self.default_n_ops//10, self.default_n_ops//10)

        try:
            n_embedding_ops = self.embedding_estimator.estimate_ops(time_slices[2])
        except ValueError:
            n_embedding_ops = self.default_n_ops + np.random.randint(-self.default_n_ops//10, self.default_n_ops//10)

        if n_insertion_ops > 0:
            data_chunk = self.data_loader.get_next_chunk(n_insertion_ops)
            try:
                validate_data_chunk(data_chunk, self._config.n_features)
                data_chunk = _as_f32_c_contig_2d(data_chunk)
            except ValueError as e:
                print(f"Error validating data chunk: {e}")

            # TODO
            self._impl.append_data(data_chunk)

        # TODO
        res = self._impl.run(n_insertion_ops, n_update_ops, n_embedding_ops)

        if res.insertion_completed:
            self.scheduler.set_insertion_weight(0)
        else:
            self.insertion_estimator.add_sample(res.n_insertion_ops, res.insertion_time)

        self.update_estimator.add_sample(res.n_update_ops, res.update_time)
        self.embedding_estimator.add_sample(res.n_embedding_ops, res.embedding_time)

        return res

    def get_embedding(self) -> cp.ndarray:
        embedding_view = cp.asarray(self._impl.get_embedding())
        return embedding_view

    def get_neighbor_indices(self) -> cp.ndarray:
        neighbor_indices_view = cp.asarray(self._impl.get_neighbor_indices())
        return neighbor_indices_view

    def get_neighbor_dists(self) -> cp.ndarray:
        neighbor_dists_view = cp.asarray(self._impl.get_neighbor_dists())
        return neighbor_dists_view



