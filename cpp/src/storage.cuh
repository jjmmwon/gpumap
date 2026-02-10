#pragma once

#include "rmm/device_uvector.hpp"

namespace gpumap {

template <typename Data_t, typename Idx_t = size_t> struct Matrix {
  Idx_t n_rows{};
  Idx_t n_cols{};

  rmm::device_uvector<Data_t> buf;

  Matrix(Idx_t n_rows_, Idx_t n_cols_)
      : n_rows(n_rows_), n_cols(n_cols_),
        buf(static_cast<size_t>(n_rows_) * static_cast<size_t>(n_cols_),
            rmm::cuda_stream_default) {}

  Data_t *data() { return buf.data(); }

  void update_n_rows(Idx_t new_n_rows) { n_rows = new_n_rows; }

  ~Matrix() = default;
};

template <typename Data_t = float, typename Idx_t = size_t> struct Storage {
  Idx_t capacity;
  Idx_t n_features;
  Idx_t n_neighbors;

  Idx_t n_data_rows;
  Idx_t n_rows;

  // Data processing buffers
  rmm::device_uvector<Data_t> data;

  // neighbor graph matrices
  rmm::device_uvector<Idx_t> neighbor_indices;
  rmm::device_uvector<Data_t> neighbor_dists;

  // Embedding matrices
  rmm::device_uvector<Data_t> embedding;

  void allocate_matrices() {
    data = rmm::device_uvector<Data_t>(
        static_cast<size_t>(capacity) *
            static_cast<size_t>(n_features + 1), // +1 for l2 norm
        rmm::cuda_stream_default);

    neighbor_indices = rmm::device_uvector<Idx_t>(
        static_cast<size_t>(capacity) * static_cast<size_t>(n_neighbors),
        rmm::cuda_stream_default);
    neighbor_dists = rmm::device_uvector<Data_t>(
        static_cast<size_t>(capacity) * static_cast<size_t>(n_neighbors),
        rmm::cuda_stream_default);

    embedding = rmm::device_uvector<Data_t>(static_cast<size_t>(capacity) *
                                                static_cast<size_t>(2),
                                            rmm::cuda_stream_default);
  };

  Storage() = default;
  Storage(Idx_t capacity_, Idx_t n_features_, Idx_t n_neighbors_)
      : capacity(capacity_), n_features(n_features_),
        n_neighbors(n_neighbors_) {
    allocate_matrices();
  }

  ~Storage() = default;
};

} // namespace gpumap
