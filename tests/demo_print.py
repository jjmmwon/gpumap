import numpy as np

from proumap import ProUMAP
from proumap._core import ProUMAPConfig


def main() -> None:
    pumap_cfg = ProUMAPConfig()
    pumap_cfg.n_neighbors = 15
    pumap_cfg.n_components = 2
    pumap_cfg.min_dist = 0.1
    pumap_cfg.n_epochs = 200
    pumap_cfg.negative_sample_rate = 5
    pumap_cfg.verbose = True
    pumap_cfg.max_instances = 100000
    pumap_cfg.batch_size_hint = 1024
    pumap_cfg.n_features = 10

    model = ProUMAP(pumap_cfg)

    for i in range(10):
        data_chunk = np.random.rand(1024, 10).astype(np.float32)

        model.append(data_chunk)
        data = model.get_data_view()

        print(f"Data chunk tail:\n{data_chunk[-5:]}")
        print(f"Data tail: \n{data[-5:]}")
        print(f"Data shape: {data.shape}")
        print(data.device)


if __name__ == "__main__":
    main()
