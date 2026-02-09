import numpy as np
from sklearn.datasets import fetch_openml

class DataLoader:
    def __init__(self):
        raise NotImplementedError("init not implemented")

    def get_next_chunk(self, chunk_size: int):
        raise NotImplementedError("get_next_chunk not implemented")


def load_mnist_numpy():
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    return X.astype(np.float32) / 255.0, y.astype(np.int64)

class PreloadMNISTDataLoader(DataLoader):
    def __init__(self):
        self.X, _ = load_mnist_numpy()
        self.n_instances = len(self.X)
        self.n_dims = self.X.shape[1]
        self._current_index = 0

    def get_next_chunk(self, chunk_size: int):
        chunk_size = min(chunk_size, self.n_instances - self._current_index)
        data_chunk = self.X[self._current_index:self._current_index + chunk_size]
        self._current_index += chunk_size
        return data_chunk
