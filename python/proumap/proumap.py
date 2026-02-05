from __future__ import annotations

import numpy as np
import cupy as cp

from proumap.validators import validate_data_chunk
from proumap.utils import _as_f32_c_contig_2d

from . import _core


class ProUMAP:
    """
    Progressive UMAP main class.
    Args:
        config (Optional[ProUMAPConfig]): Configuration for ProUMAP. If None,
            default configuration will be used.
    """

    def __init__(self, config: _core.ProUMAPConfig):
        self._config = config
        self._impl = _core.ProUMAP(self.config)

        self._closed = False

    @property
    def config(self) -> _core.ProUMAPConfig:
        return self._config

    def close(self):
        if not self._closed:
            self._impl.close()
            self._closed = True

    def append(self, data_chunk: np.ndarray):
        try:
            validate_data_chunk(data_chunk, self._config.n_features)
            data_chunk = _as_f32_c_contig_2d(data_chunk)
        except ValueError as e:
            print(f"Error validating data chunk: {e}")
            return

        self._impl.append_data(data_chunk)

    def run_nng_build(self, chunk_size: int) -> None:
        self._impl.run_nng_build()

    def run_optimize_embeddings(self, chunk_size: int) -> None:
        self._impl.run_optimize_embeddings()
