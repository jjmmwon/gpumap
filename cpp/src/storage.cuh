#pragma once

#include "rmm/device_uvector.hpp"
#include <cstddef>

namespace gpumap {

template <typename Data_t = float, typename Idx_t = size_t> struct Storage {
  Idx_t capacity;
  Idx_t n_features;
  Idx_t n_neighbors;

  // current number of data rows stored in data matrix and neighbor graphs
  Idx_t n_data_rows;
  Idx_t n_rows;

  size_t seed;

  // configs for nndescent
  size_t max_iter_nnd;
  size_t n_samples_nnd;
  float termination_threshold_nnd;

  // Data processing buffers
  rmm::device_uvector<Data_t> data;

  // neighbor graph matrices
  rmm::device_uvector<Idx_t> neighbor_indices;
  size_t neighbor_indices_block_size;

  rmm::device_uvector<Data_t> neighbor_dists;
  size_t neighbor_dists_block_size;

  // Embedding matrices
  rmm::device_uvector<Data_t> embedding;

  void allocate_matrices() {
    data = rmm::device_uvector<Data_t>(
        static_cast<size_t>(capacity) *
            static_cast<size_t>(n_features + 1), // +1 for l2 norm
        rmm::cuda_stream_default);

    neighbor_indices_block_size = static_cast<size_t>(n_neighbors) * 2 + 1;
    // * 2 for storing candidates
    // + 1 for storing whether an instance is dirty or not (for kernel size computation)
    
    neighbor_indices = rmm::device_uvector<Idx_t>(
        static_cast<size_t>(capacity) * neighbor_indices_block_size * sizeof(Idx_t), rmm::cuda_stream_default);
    

    neighbor_dists_block_size = static_cast<size_t>(n_neighbors) * 2 + 1;
    // * 2 for storing candidates 
    // + 1 for storing sigma_i

    neighbor_dists = rmm::device_uvector<Data_t>(
        static_cast<size_t>(capacity) * neighbor_dists_block_size * sizeof(Data_t), rmm::cuda_stream_default);
    
    embedding = rmm::device_uvector<Data_t>(static_cast<size_t>(capacity) *
                                                static_cast<size_t>(2),
                                            rmm::cuda_stream_default);
  };

  Storage() = delete;
  Storage(Idx_t capacity_, Idx_t n_features_, Idx_t n_neighbors_,
          size_t seed_ = 42, size_t max_iter_nnd_ = 30,
          size_t n_samples_nnd_ = 150,
          float termination_threshold_nnd_ = 0.0001f)
      : capacity(capacity_), n_features(n_features_), n_neighbors(n_neighbors_),
      seed(seed_),
        max_iter_nnd(max_iter_nnd_), n_samples_nnd(n_samples_nnd_),
        termination_threshold_nnd(termination_threshold_nnd_),
        data(0, rmm::cuda_stream_default),
        neighbor_indices(0, rmm::cuda_stream_default),
        neighbor_dists(0, rmm::cuda_stream_default),
        embedding(0, rmm::cuda_stream_default) {
    allocate_matrices();
  }

  ~Storage() = default;
};

} // namespace gpumap
