import numpy as np
import cupy as cp
from gpumap import GPUMAP, GPUMAPConfig
from gpumap.data_loader import PreloadMNISTDataLoader
from gpumap.scheduler import DefaultScheduler
from gpumap.estimator import DefaultLinearEstimator


if __name__ == "__main__":
    data_loader = PreloadMNISTDataLoader()
    print(f"Loaded {data_loader.n_instances} samples with {data_loader.n_dims} features")


    # GPUMAP 설정
    config = GPUMAPConfig(
        n_neighbors=15,
        n_components=2,
        n_epochs=200,
        min_dist=0.1,
        random_state=42,
        verbose=True,
        n_instances=data_loader.n_instances,
        n_dims=data_loader.n_dims
    )

    gpumap = GPUMAP(
        data_loader=data_loader,
        config=config,
        scheduler=DefaultScheduler(),
        insertion_estimator=DefaultLinearEstimator(),
        update_estimator=DefaultLinearEstimator(),
        embedding_estimator=DefaultLinearEstimator()
    )

    print(f"Running progressive UMAP...")
    for i in range(10):
        print(f"  Iteration {i+1}...")
        result = gpumap.run(target_latency=10.0)
        print(f"Target latency: 10s")
        # printout the total time
        total_time = result.insertion_time + result.update_time + result.embedding_time
        print(f"  Total time: {total_time:.4f}s ")


    gpumap.close()

