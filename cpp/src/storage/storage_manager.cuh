#pragma once

#include "matrix2D.cuh"
#include "matrix_utils.cuh"
#include <cstdint>
#include <memory>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace proumap {
namespace storage {

template <typename Data_t, typename Idx_t = uint32_t> class StorageManager {
private:
  // RAFT resource handle
  std::unique_ptr<raft::resources> res_;

  // RMM stream pool for efficient stream management
  std::unique_ptr<rmm::cuda_stream_pool> stream_pool_;
  rmm::cuda_stream_view primary_stream_;

public:
  Idx_t max_rows;
  Idx_t n_features;
  Idx_t n_neighbors;
  Idx_t n_rows = 0;

  // Four matrices for different purposes
  DeviceMatrix2D<Data_t, Idx_t> progressive_data;  // Progressive data matrix
  DeviceMatrix2D<Idx_t, Idx_t> neighbor_indices;   // Neighbor indices
  DeviceMatrix2D<float, Idx_t> neighbor_distances; // Neighbor distances
  DeviceMatrix2D<float, Idx_t> embedding_matrix;   // Current embeddings

  StorageManager() = default;

  StorageManager(Idx_t max_rows_, Idx_t n_features_, Idx_t n_neighbors_,
                 size_t stream_pool_size = 4)
      : max_rows(max_rows_), n_features(n_features_),
        n_neighbors(n_neighbors_) {
    initialize_resources(stream_pool_size);
    allocate_matrices();
  }

  ~StorageManager() = default;

  // Move-only semantics for resource management
  StorageManager(const StorageManager &) = delete;
  StorageManager &operator=(const StorageManager &) = delete;
  StorageManager(StorageManager &&) = default;
  StorageManager &operator=(StorageManager &&) = default;

  // Initialize RAFT resources and RMM streams
  void initialize_resources(size_t stream_pool_size = 4) {
    // Create stream pool
    stream_pool_ = std::make_unique<rmm::cuda_stream_pool>(stream_pool_size);
    primary_stream_ = stream_pool_->get_stream();

    // Initialize RAFT resources with our stream
    res_ = std::make_unique<raft::resources>();
    raft::resource::set_cuda_stream(*res_, primary_stream_);
  }

  // Allocate all matrices
  void allocate_matrices() {
    if (!res_) {
      throw std::runtime_error("Resources not initialized");
    }

    progressive_data =
        DeviceMatrix2D<Data_t, Idx_t>(max_rows, n_features, primary_stream_);
    neighbor_indices = DeviceMatrix2D<Idx_t, Idx_t>(max_rows, n_neighbors * 2,
                                                    primary_stream_);
    neighbor_distances =
        DeviceMatrix2D<Data_t, Idx_t>(max_rows, n_neighbors, primary_stream_);
    embedding_matrix =
        DeviceMatrix2D<Data_t, Idx_t>(max_rows, 2, primary_stream_);
  }

  // Get RAFT resources
  raft::resources &get_resources() {
    if (!res_)
      throw std::runtime_error("Resources not initialized");
    return *res_;
  }

  const raft::resources &get_resources() const {
    if (!res_)
      throw std::runtime_error("Resources not initialized");
    return *res_;
  }

  // Get primary stream
  rmm::cuda_stream_view get_primary_stream() const { return primary_stream_; }

  // Get additional stream from pool
  rmm::cuda_stream_view get_stream() const {
    if (!stream_pool_)
      throw std::runtime_error("Stream pool not initialized");
    return stream_pool_->get_stream();
  }

  // Get matrix views for computations
  auto get_embedding_view() const { return embedding_matrix.view(); }
  auto get_data_view() const { return progressive_data.view(); }
  auto get_neighbor_indices_view() const { return neighbor_indices.view(); }
  auto get_neighbor_distances_view() const { return neighbor_distances.view(); }

  // Utility methods
  Idx_t capacity() const { return max_rows; }
  Idx_t size() const { return n_rows; }
  Idx_t features() const { return n_features; }
  Idx_t neighbors() const { return n_neighbors; }

  bool is_full() const { return n_rows >= max_rows; }
  bool is_empty() const { return n_rows == 0; }

  // Synchronize all operations
  void synchronize() { raft::resource::sync_stream(*res_); }
};

} // namespace storage
} // namespace proumap