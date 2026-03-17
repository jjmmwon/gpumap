import os
import warnings
import numpy as np
from sklearn.datasets import fetch_openml


class DataLoader:
    n_instances: int
    n_features: int
    X: np.ndarray
    y: np.ndarray | None
    _current_index: int = 0

    def __init__(self):
        raise NotImplementedError("init not implemented")

    def get_next_chunk(self, chunk_size: int) -> np.ndarray:
        chunk_size = min(chunk_size, self.n_instances - self._current_index)
        data_chunk = self.X[self._current_index : self._current_index + chunk_size]
        self._current_index += chunk_size
        return data_chunk


def load_numpy_data(file_path: str) -> np.ndarray:
    if os.path.exists(file_path):
        data = np.load(file_path)
        return data
    else:
        raise ValueError(f"File not found: {file_path}")


PRELOAD_DATASETS = ["mnist", "fashion_mnist", "wine_reviews", "svhn"]


class PreloadDataLoader(DataLoader):
    def __init__(self, dataset_name: str):
        if dataset_name not in PRELOAD_DATASETS:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

        file_dir_path = (
            f"{os.path.dirname(os.path.realpath(__file__))}/data/{dataset_name}"
        )
        X_path = f"{file_dir_path}/{dataset_name}_X.npy"
        y_path = f"{file_dir_path}/{dataset_name}_y.npy"

        self.X = load_numpy_data(X_path)
        self.y = load_numpy_data(y_path) if os.path.exists(y_path) else None
        self.n_instances = len(self.X)
        self.n_features = self.X.shape[1]
        self._current_index = 0


class NumpyDataLoader(DataLoader):
    def __init__(
        self, dataset_name: str, npy_path: str, label_npy_path: str | None = None
    ):
        self.dataset_name = dataset_name
        self.X = load_numpy_data(npy_path)
        self.y = load_numpy_data(label_npy_path) if label_npy_path else None
        self.n_instances = len(self.X)
        self.n_features = self.X.shape[1]
        self._current_index = 0
