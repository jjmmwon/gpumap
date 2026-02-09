#pragma once

#include "storage/matrix2D.cuh"

namespace gpumap {
namespace storage {
template <typename Data_t, typename Idx_t = uint32_t> struct Storage {

  Idx_t max_rows{};
  Idx_t n_features{};
  Idx_t n_neighbors{};

  DeviceMatrix2D<Data_t, Idx_t> data;
  DeviceMatrix2D<Idx_t, Idx_t> neighbors_indices;
  DeviceMatrix2D<Data_t, Idx_t> neighbors_dists;
  DeviceMatrix2D<Data_t, Idx_t> embedding;
  // Other matrices or graph as needed

  void allocate_matrices() {
    data = DeviceMatrix2D<Data_t, Idx_t>(max_rows, n_features,
                                         rmm::cuda_stream_default);
    neighbors_indices = DeviceMatrix2D<Idx_t, Idx_t>(max_rows, n_neighbors,
                                                     rmm::cuda_stream_default);
    neighbors_dists = DeviceMatrix2D<Data_t, Idx_t>(max_rows, n_neighbors,
                                                    rmm::cuda_stream_default);
    embedding = DeviceMatrix2D<Data_t, Idx_t>(max_rows, n_features,
                                              rmm::cuda_stream_default);
  };

  Storage() = default;
  Storage(Idx_t max_rows_, Idx_t n_features_, Idx_t n_neighbors_)
      : max_rows(max_rows_), n_features(n_features_),
        n_neighbors(n_neighbors_) {
    allocate_matrices();
  }

  // Move-only
  Storage(Storage &&) = default;
  Storage &operator=(Storage &&) = default;
  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;

  ~Storage() = default;
};

} // namespace storage
} // namespace gpumap
