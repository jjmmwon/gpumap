#pragma once

#include "kernels/buffers.cuh"
#include "raft/core/device_mdarray.hpp"
#include "rmm/device_uvector.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <raft/core/resources.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <thread>

namespace gpumap {

// Ensures the CUDA runtime and RMM per-device resource map are initialized
// before the first device/pinned allocation in NNDBuffers.
// The first cudaMalloc in some environments can transiently fail while the
// driver is still finalizing context setup. Retry a tiny alloc/free probe here
// so later RMM allocations do not become the first CUDA call in the process.
struct CudaWarmup {
  CudaWarmup() {
    constexpr int kMaxAttempts = 4;
    constexpr auto kRetryDelay = std::chrono::milliseconds(50);

    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
      cudaSetDevice(device);
    }
    cudaGetLastError(); // clear any pending CUDA error

    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
      cudaFree(0);
      cudaGetLastError();

      void *ptr = nullptr;
      if (cudaMalloc(&ptr, 1) == cudaSuccess) {
        if (ptr != nullptr)
          cudaFree(ptr);
        break;
      }

      cudaGetLastError();
      std::this_thread::sleep_for(kRetryDelay);
    }
    cudaGetLastError(); // clear any pending CUDA error

    // Pre-initialize the RMM per-device resource map (CPU-only operation).
    (void)rmm::mr::get_current_device_resource();

    // WSL2 WDDM에서 이전 인스턴스가 cudaFreeHost()한 pinned memory가
    // 드라이버 레이어에 실제로 반납되기 전에 새 cudaMallocHost()를 시도하면
    // std::bad_alloc이 발생한다. 1바이트 alloc/free로 pinned memory
    // subsystem을 미리 건드려 WDDM 레이어의 상태를 안정화한다.
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
      void *pinned = nullptr;
      if (cudaMallocHost(&pinned, 1) == cudaSuccess) {
        if (pinned != nullptr)
          cudaFreeHost(pinned);
        break;
      }

      cudaGetLastError();
      std::this_thread::sleep_for(kRetryDelay);
    }
    cudaGetLastError(); // clear any error from the probe
  }
};

template <typename Data_t = float, typename Idx_t = uint32_t> struct Storage {
  Idx_t n_instances;
  Idx_t n_features;
  Idx_t n_neighbors;
  Idx_t n_components;

  // current number of data rows stored in data matrix and neighbor graphs
  Idx_t n_inserted_instances = 0;

  // Completion flags — set to true once and never reset.
  bool insertion_completed = false;
  bool update_completed = false;
  bool embedding_completed = false;

  Idx_t seed;

  // configs for nndescent
  float termination_threshold_nnd;

  // Declared before any RMM-backed member so warmup runs before the first
  // device or pinned allocation in this struct.
  CudaWarmup cuda_warmup;

  // high dim raw vectors
  rmm::device_uvector<Data_t> data;
  Idx_t data_block_size;

  // neighbor graph matrices
  rmm::device_uvector<Idx_t> neighbor_indices;
  Idx_t neighbor_indices_block_size;

  rmm::device_uvector<Data_t> neighbor_dists;
  Idx_t neighbor_dists_block_size;

  // Embedding matrices
  rmm::device_uvector<Data_t> embedding;
  rmm::device_scalar<float> max_w;

  // Owns the raft::resources for nnd_buffer's lifetime.
  // Declared before nnd_buffer so it is constructed first and destroyed last.
  raft::resources raft_res;

  // Buffers used during nndescent algorithm.
  NNDBuffers<Data_t, int> nnd_buffer;

  void allocate_matrices() {
    data_block_size = n_features + 1; // +1 for l2 norm
    data = rmm::device_uvector<Data_t>(n_instances *
                                           data_block_size, // +1 for l2 norm
                                       rmm::cuda_stream_default);

    neighbor_indices_block_size = n_neighbors + 1;
    // + 1 for storing whether an instance is dirty or not (for kernel size
    // computation)

    neighbor_indices = rmm::device_uvector<Idx_t>(
        n_instances * neighbor_indices_block_size, rmm::cuda_stream_default);

    neighbor_dists_block_size = n_neighbors + 1;
    // + 1 for storing sigma_i

    neighbor_dists = rmm::device_uvector<Data_t>(
        n_instances * neighbor_dists_block_size, rmm::cuda_stream_default);

    embedding = rmm::device_uvector<Data_t>(n_instances * n_components,
                                            rmm::cuda_stream_default);
  };

  Storage() = delete;
  Storage(Idx_t n_instances_, Idx_t n_features_, Idx_t n_neighbors_,
          Idx_t n_components_ = 2, Idx_t seed_ = 42, Idx_t n_samples_nnd_ = 150,
          float termination_threshold_nnd_ = 0.0001f)
      : n_instances(n_instances_), n_features(n_features_),
        n_neighbors(n_neighbors_), n_components(n_components_), seed(seed_),
        termination_threshold_nnd(termination_threshold_nnd_),
        data(0, rmm::cuda_stream_default),
        neighbor_indices(0, rmm::cuda_stream_default),
        neighbor_dists(0, rmm::cuda_stream_default),
        embedding(0, rmm::cuda_stream_default),
        max_w(0.0f, rmm::cuda_stream_default), cuda_warmup(), raft_res(),
        nnd_buffer(raft_res, n_instances_, n_neighbors_) {
    allocate_matrices();
  }

  ~Storage() = default;
};

} // namespace gpumap
