from gpumap import GPUMAP, GPUMAPParams
from gpumap.data_loader import PreloadMNISTDataLoader
from gpumap.scheduler import DefaultScheduler
from gpumap.estimator import DefaultLinearEstimator


if __name__ == "__main__":
    data_loader = PreloadMNISTDataLoader()
    print(f"Loaded {data_loader.n_instances} samples with {data_loader.n_features} features")


    # GPUMAP 설정
    config = GPUMAPParams()
    config.n_neighbors = 15
    config.n_components = 2
    config.n_epochs = 200
    config.min_dist = 0.1
    config.verbose = True
    config.n_instances = data_loader.n_instances
    config.n_features = data_loader.n_features

    gpumap = GPUMAP(
        data_loader=data_loader,
        config=config,
        scheduler=DefaultScheduler(),
        insertion_estimator=DefaultLinearEstimator(),
        update_estimator=DefaultLinearEstimator(),
        embedding_estimator=DefaultLinearEstimator()
    )

    print(f"Running GPUMAP...")
    for i in range(10):
        print(f"  Iteration {i+1}...")
        result = gpumap.run(target_latency=10.0)
        print(f"Target latency: 10s")
        # printout the total time
        total_time = result.insertion_time + result.update_time + result.embedding_time
        print(f"  Total time: {total_time:.4f}s ")


    gpumap.close()

