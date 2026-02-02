from typing import Optional

import numpy as np
import cupy as cp

from proumap.types import ProUMAPConfig
from proumap.validators import validate_data_chunk
from . import _core


class ProUMAP:
    """
    Progressive UMAP main class.
    Args:
        config (Optional[ProUMAPConfig]): Configuration for ProUMAP. If None,
            default configuration will be used.
    """

    def __init__(self, config: Optional[ProUMAPConfig] = None):
        self.config = config or ProUMAPConfig()
        self._impl: _core.ProUMAPImpl = _core.ProUMAPImpl(self.config)

        self._feature_dim: Optional[int] = None

        self._closed = False

    def close(self):
        if not self._closed:
            self._impl.close()
            self._closed = True

    def append(self, data_chunk: np.ndarray | cp.ndarray):
        try:
            self._feature_dim = validate_data_chunk(data_chunk, self._feature_dim)
        except ValueError as e:
            print(f"Error validating data chunk: {e}")
            return

        self._impl.append(data_chunk)

    def run_nn(self, chunk_size: int) -> None:
        self._impl.run_nn(chunk_size)

    def run_embedding(self, chunk_size: int) -> None:
        self._impl.run_embedding(chunk_size)

    def get_embeddings(self) -> np.ndarray | cp.ndarray:
        return self._impl.get_embeddings()

    def get_current_status(self):
        return self._impl.get_current_status()
