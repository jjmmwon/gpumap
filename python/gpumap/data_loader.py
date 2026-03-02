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


def load_numpy_data(dataset_name: str):
    file_dir_path = f"{os.path.dirname(os.path.realpath(__file__))}/data/{dataset_name}"
    file_path = f"{file_dir_path}/{dataset_name}_X.npy"
    y_path = f"{file_dir_path}/{dataset_name}_y.npy"
    if os.path.exists(file_path):
        X = np.load(file_path)
        if os.path.exists(y_path):
            y = np.load(y_path)
        else:
            y = None
        return X, y
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")


PRELOAD_DATASETS = ["mnist", "gist", "wine_reviews", "covertype"]


class PreloadDataLoader(DataLoader):
    def __init__(self, dataset_name: str):
        if dataset_name not in PRELOAD_DATASETS:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

        self.X, self.y = load_numpy_data(dataset_name)
        self.n_instances = len(self.X)
        self.n_features = self.X.shape[1]
        self._current_index = 0
