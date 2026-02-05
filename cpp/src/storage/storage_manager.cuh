#pragma once

#include "matrix2D.cuh"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace proumap {
namespace storage {

template <typename Data_t, typename Idx_t = uint32_t> class StorageManager {
private:
  // ---- Small enums to index arrays ----
  enum class StreamSlot : uint8_t {
    Primary = 0,
    DataManage = 1,
    NNG = 2,
    Embedding = 3,
    Count
  };

  enum class ResSlot : uint8_t {
    DataManage = 0,
    NNG = 1,
    Embedding = 2,
    Count
  };

  static constexpr size_t kNumStreams = static_cast<size_t>(StreamSlot::Count);
  static constexpr size_t kNumRes = static_cast<size_t>(ResSlot::Count);

  // ---- RAFT resources (per stream) ----
  std::array<std::unique_ptr<raft::resources>, kNumRes> res_{};

  // ---- Stream pool + streams ----
  std::unique_ptr<rmm::cuda_stream_pool> stream_pool_;
  std::array<rmm::cuda_stream_view, kNumStreams> streams_{};

  // ---- Helpers ----
  static constexpr size_t idx(StreamSlot s) { return static_cast<size_t>(s); }
  static constexpr size_t idx(ResSlot r) { return static_cast<size_t>(r); }

  raft::resources &get_res_(ResSlot r) {
    auto &p = res_[idx(r)];
    if (!p)
      throw std::runtime_error("Resources not initialized");
    return *p;
  }
  const raft::resources &get_res_(ResSlot r) const {
    auto &p = res_[idx(r)];
    if (!p)
      throw std::runtime_error("Resources not initialized");
    return *p;
  }

  rmm::cuda_stream_view get_stream_(StreamSlot s) const {
    if (!stream_pool_)
      throw std::runtime_error("Stream pool not initialized");
    return streams_[idx(s)];
  }

  // ---- Initialization ----
  void initialize_resources(size_t stream_pool_size_) {
    stream_pool_ = std::make_unique<rmm::cuda_stream_pool>(stream_pool_size);

    // Streams: primary + 3 worker streams
    for (size_t i = 0; i < kNumStreams; ++i) {
      streams_[i] = stream_pool_->get_stream();
    }

    // Resources: bind each resource to a specific stream
    auto bind = [&](ResSlot r, StreamSlot s) {
      res_[idx(r)] = std::make_unique<raft::resources>();
      raft::resource::set_cuda_stream(*res_[idx(r)], streams_[idx(s)]);
    };

    bind(ResSlot::DataManage, StreamSlot::DataManage);
    bind(ResSlot::NNG, StreamSlot::NNG);
    bind(ResSlot::Embedding, StreamSlot::Embedding);
  }

  void allocate_matrices() {
    if (!stream_pool_)
      throw std::runtime_error("Resources not initialized");

    const auto primary = streams_[idx(StreamSlot::Primary)];

    progressive_data =
        DeviceMatrix2D<Data_t, Idx_t>(max_rows, n_features, primary);

    neighbor_indices =
        DeviceMatrix2D<Idx_t, Idx_t>(max_rows, n_neighbors * 2, primary);

    neighbor_distances =
        DeviceMatrix2D<Data_t, Idx_t>(max_rows, n_neighbors, primary);

    embedding_matrix = DeviceMatrix2D<Data_t, Idx_t>(max_rows, 2, primary);
  }

public:
  // ---- Public configuration ----
  Idx_t max_rows{};
  Idx_t n_features{};
  Idx_t n_neighbors{};
  size_t stream_pool_size{4};

  // ---- Device matrices ----
  DeviceMatrix2D<Data_t, Idx_t> progressive_data; // (max_rows x n_features)
  DeviceMatrix2D<Idx_t, Idx_t>
      neighbor_indices; // (max_rows x (2*k)) or as you decide
  DeviceMatrix2D<Data_t, Idx_t> neighbor_distances; // (max_rows x k)
  DeviceMatrix2D<Data_t, Idx_t>
      embedding_matrix; // (max_rows x 2) (or n_components)

  StorageManager() = default;

  StorageManager(Idx_t max_rows_, Idx_t n_features_, Idx_t n_neighbors_,
                 size_t stream_pool_size_ = 4)
      : max_rows(max_rows_), n_features(n_features_), n_neighbors(n_neighbors_),
        stream_pool_size(stream_pool_size_) {
    initialize_resources(stream_pool_size_);
    allocate_matrices();
  }

  // Move-only
  StorageManager(const StorageManager &) = delete;
  StorageManager &operator=(const StorageManager &) = delete;
  StorageManager(StorageManager &&) = default;
  StorageManager &operator=(StorageManager &&) = default;

  ~StorageManager() = default;

  // ---- Resource accessors (kept for convenience) ----
  raft::resources &get_data_manage_resources() {
    return get_res_(ResSlot::DataManage);
  }
  const raft::resources &get_data_manage_resources() const {
    return get_res_(ResSlot::DataManage);
  }

  raft::resources &get_nng_build_resources() { return get_res_(ResSlot::NNG); }
  const raft::resources &get_nng_build_resources() const {
    return get_res_(ResSlot::NNG);
  }

  raft::resources &get_embedding_resources() {
    return get_res_(ResSlot::Embedding);
  }
  const raft::resources &get_embedding_resources() const {
    return get_res_(ResSlot::Embedding);
  }

  // ---- Stream accessors ----
  rmm::cuda_stream_view get_primary_stream() const {
    return get_stream_(StreamSlot::Primary);
  }

  // get any additional stream from pool (not one of the reserved ones)
  rmm::cuda_stream_view get_stream() const {
    if (!stream_pool_)
      throw std::runtime_error("Stream pool not initialized");
    return stream_pool_->get_stream();
  }

  // ---- Views ----
  auto get_data_view() const { return progressive_data.view(); }
  auto get_neighbor_indices_view() const { return neighbor_indices.view(); }
  auto get_neighbor_distances_view() const { return neighbor_distances.view(); }
  auto get_embedding_view() const { return embedding_matrix.view(); }

  // ---- Utility ----
  bool is_full() const { return progressive_data.rows() >= max_rows; }
  bool is_empty() const { return progressive_data.rows() == 0; }

  Idx_t data_size() const { return progressive_data.rows(); }
  Idx_t neighbor_size() const { return neighbor_indices.rows(); }
  Idx_t embedding_size() const { return embedding_matrix.rows(); }

  void synchronize() {
    for (auto &r : res_) {
      if (r)
        raft::resource::sync_stream(*r);
    }
  }
};

} // namespace storage
} // namespace proumap
