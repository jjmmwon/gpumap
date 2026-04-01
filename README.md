# GPUMAP

GPU-accelerated progressive dimensionality reduction based on UMAP, implemented in CUDA C++ with Python bindings via pybind11.

## Directory Structure

```
gpumap/
├── env.yml                # conda environment (CUDA 12.6, RAPIDS 26.02, pybind11, …)
├── CMakeLists.txt         # top-level CMake
├── pyproject.toml         # scikit-build-core config
├── cpp/                   # CUDA C++ source and headers
├── python/gpumap/         # Python API + pybind11 bindings
├── evaluation/            # benchmark and evaluation scripts
└── scripts/               # dev utilities (clangd patch, etc.)
```

## Requirements

- NVIDIA GPU with a recent driver installed on the host
- conda (Miniconda or Anaconda)

## Environment Setup

```bash
conda env create -f env.yml
conda activate gpumap
```

## Build & Install

### Option A — pip install (scikit-build-core)

The simplest path: builds the CUDA extension and installs the Python package in one step.

```bash
pip install -U pip wheel setuptools
pip install -e . -v
```

### Option B — CMake build

Use this when you need explicit control over the build (e.g., for IDE integration or debugging).

```bash
# Configure
cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
cmake --build build -j

# Install Python package pointing at the build tree
pip install -e . -v
```

After the CMake build you can patch `build/compile_commands.json` for clangd:

```bash
python scripts/patch_compile_commands.py
```

## Python API

### Parameters

```python
from gpumap import GPUMAPParams

params = GPUMAPParams()
params.n_instances   = 70000   # total number of data points
params.n_features    = 784     # input dimensionality
params.n_neighbors   = 15      # k for k-NN graph
params.n_components  = 2       # output dimensionality
params.min_dist      = 0.1     # controls compactness of the embedding
params.max_epoch     = 200     # number of optimization epochs per embedding step
```

### Data Loaders

```python
from gpumap.data_loader import NumpyDataLoader, PreloadDataLoader
import numpy as np

# Load from a .npy file
loader = NumpyDataLoader(
    dataset_name="my_dataset",
    npy_path="data/my_dataset_X.npy",
    label_npy_path="data/my_dataset_y.npy",   # optional
)

# Use a built-in dataset  ("mnist", "fashion_mnist", "wine_reviews", "svhn")
loader = PreloadDataLoader("mnist")

print(loader.n_instances, loader.n_features)
```

### Creating a GPUMAP Model

```python
from gpumap import GPUMAP, GPUMAPParams
from gpumap.data_loader import NumpyDataLoader

loader = NumpyDataLoader("mnist", "data/mnist_X.npy")

params = GPUMAPParams()
params.n_instances = loader.n_instances
params.n_features  = loader.n_features
params.n_neighbors = 15
params.n_components = 2

model = GPUMAP(data_loader=loader, params=params)
```

### Running the Progressive Pipeline

GPUMAP processes data incrementally. Each call to `run()` performs one cycle of
insertion → graph update → embedding optimization, proportionally divided within
the given time budget.

```python
import numpy as np

while True:
    result = model.run(target_latency=10.0)   # aim for ~10 s per iteration

    print(
        f"inserted={result.n_inserted_instances} "
        f"ins_ops={result.n_insertion_ops} "
        f"upd_ops={result.n_update_ops} "
        f"emb_ops={result.n_embedding_ops}"
    )

    # Retrieve current embedding (CuPy array, shape: [n_inserted, n_components])
    embedding = model.get_embedding()          # cupy.ndarray
    embedding_np = embedding.get()             # transfer to NumPy

    if result.embedding_completed:
        break

model.close()
```

### Retrieving Results

```python
import cupy as cp

# 2-D embedding coordinates — shape (n_inserted, n_components)
embedding = model.get_embedding()        # cp.ndarray on GPU
embedding_np = embedding.get()           # np.ndarray on CPU

# k-NN graph — shape (n_inserted, n_neighbors)
nn_indices = model.get_neighbor_indices().get()   # np.ndarray, dtype int32
nn_dists   = model.get_neighbor_dists().get()     # np.ndarray, dtype float32
```

### Manual Op Control

If you need fine-grained control over how many operations run per iteration,
use `run_with_n_ops()` instead of `run()`:

```python
result = model.run_with_n_ops(
    n_insertion_ops=500,    # data points to insert this step
    n_update_ops=2000,      # k-NN graph update steps
    n_embedding_ops=5000,   # embedding optimization steps
)
```

### Scheduling Strategies

The strategy controls how the time budget is split between the three operations.

```python
from gpumap import GPUMAP, GPUMAPParams, WeightedStrategy, SequentialPriorityStrategy

# WeightedStrategy (default) — runs all three operations in parallel,
# weighted by their relative costs.
strategy = WeightedStrategy(
    n_instances=loader.n_instances,
    n_neighbors=15,
)

# SequentialPriorityStrategy — finishes insertion before update,
# finishes update before embedding.
strategy = SequentialPriorityStrategy(
    n_instances=loader.n_instances,
    n_neighbors=15,
)

model = GPUMAP(data_loader=loader, params=params, strategy=strategy)
```

### Full Minimal Example

```python
from gpumap import GPUMAP, GPUMAPParams
from gpumap.data_loader import NumpyDataLoader

loader = NumpyDataLoader("mnist", "data/mnist_X.npy", "data/mnist_y.npy")

params = GPUMAPParams()
params.n_instances  = loader.n_instances
params.n_features   = loader.n_features
params.n_neighbors  = 15
params.n_components = 2
params.min_dist     = 0.1

model = GPUMAP(data_loader=loader, params=params)

try:
    while True:
        result = model.run(target_latency=10.0)
        embedding_np = model.get_embedding().get()
        print(f"[{result.n_inserted_instances}/{loader.n_instances}] "
              f"emb shape: {embedding_np.shape}")
        if result.embedding_completed:
            break
finally:
    model.close()
```