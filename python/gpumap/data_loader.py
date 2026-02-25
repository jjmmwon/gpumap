import numpy as np
from sklearn.datasets import fetch_openml


class DataLoader:
    n_instances: int
    n_features: int

    def __init__(self):
        raise NotImplementedError("init not implemented")

    def get_next_chunk(self, chunk_size: int) -> np.ndarray:
        raise NotImplementedError("get_next_chunk not implemented")


def load_mnist_numpy():
    X = np.load("data/mnist_X.npy")
    y = np.load("data/mnist_y.npy")
    return X, y


class PreloadMNISTDataLoader(DataLoader):
    def __init__(self):
        self.X, self.y = load_mnist_numpy()
        self.n_instances = len(self.X)
        self.n_features = self.X.shape[1]
        self._current_index = 0

    def get_next_chunk(self, chunk_size: int) -> np.ndarray:
        chunk_size = min(chunk_size, self.n_instances - self._current_index)
        data_chunk = self.X[self._current_index : self._current_index + chunk_size]
        self._current_index += chunk_size
        return data_chunk
