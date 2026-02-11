#pragma once

#include "rmm/device_uvector.hpp"
#include <cstddef>

namespace gpumap {

template <typename Data_t = float, typename Idx_t = uint32_t> struct Storage {
  Idx_t n_instances;
  Idx_t n_features;
  Idx_t n_neighbors;
  Idx_t n_components;

  // current number of data rows stored in data matrix and neighbor graphs

  Idx_t n_inserted_instances = 0;

  Idx_t seed;

  // configs for nndescent
  Idx_t max_iter_nnd;
  Idx_t n_samples_nnd;
  float termination_threshold_nnd;

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

  void allocate_matrices() {
    data_block_size = n_features + 1; // +1 for l2 norm
    data = rmm::device_uvector<Data_t>(
        static_cast<size_t>(n_instances) * data_block_size, // +1 for l2 norm
        rmm::cuda_stream_default);

    neighbor_indices_block_size = n_neighbors * 2 + 1;
    // * 2 for storing candidates
    // + 1 for storing whether an instance is dirty or not (for kernel size computation)

    neighbor_indices = rmm::device_uvector<Idx_t>(
        n_instances * neighbor_indices_block_size, rmm::cuda_stream_default);


    neighbor_dists_block_size = n_neighbors * 2 + 1;
    // * 2 for storing candidates
    // + 1 for storing sigma_i

    neighbor_dists = rmm::device_uvector<Data_t>(
        n_instances * neighbor_dists_block_size, rmm::cuda_stream_default);

    embedding = rmm::device_uvector<Data_t>(n_instances * n_components,
                                            rmm::cuda_stream_default);
  };

  Storage() = delete;
  Storage(Idx_t n_instances_, Idx_t n_features_, Idx_t n_neighbors_,
          Idx_t n_components_ = 2,
          Idx_t seed_ = 42, Idx_t max_iter_nnd_ = 30,
          Idx_t n_samples_nnd_ = 150,
          float termination_threshold_nnd_ = 0.0001f)
      : n_instances(n_instances_), n_features(n_features_), n_neighbors(n_neighbors_),
      n_components(n_components_),
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
