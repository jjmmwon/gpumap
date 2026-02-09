// include/gpumap/gpumap.cuh
#pragma once

#include <cstdint>
#include <memory>

// Public return types (lightweight views)
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>

namespace gpumap {

struct GPUMAPParams{
  size_t n_neighbors = 15;
  size_t n_components = 2;
  float min_dist = 0.1f;
  size_t n_epochs = 200;
  size_t negative_sample_rate = 5;
  bool verbose = false;
  size_t n_instances = 0;
  size_t n_features = 0;

  GPUMAPParams() = default;

  GPUMAPParams(size_t n_instances_, size_t n_features_)
      : n_instances(n_instances_), n_features(n_features_) {}

  // size_t batch_size_hint = 1024;
};

struct GPUMAPResult {
  size_t n_insertion_ops;
  size_t n_update_ops;
  size_t n_embedding_ops;

  float insertion_time;
  float update_time;
  float embedding_time;

  bool insertion_completed;
};

// Public, non-template API (bind this from pybind).
// Internals (StorageManager, streams, RMM, etc.) stay in src via PImpl.
class GPUMAP {
public:
  using Data_t = float;
  using Idx_t = size_t;
  using DataView =
      raft::device_matrix_view<const Data_t, Idx_t, raft::row_major>;
  using EmbedView =
      raft::device_matrix_view<const Data_t, Idx_t, raft::row_major>;

  GPUMAP(const GPUMAPParams &params);

  // Move-only
  GPUMAP(const GPUMAP &) = delete;
  GPUMAP &operator=(const GPUMAP &) = delete;
  GPUMAP(GPUMAP &&) noexcept;
  GPUMAP &operator=(GPUMAP &&) noexcept;

  ~GPUMAP();

  // Host pointer input: binding layer should guarantee contiguous layout.
  // (If you want to accept stride/ld, add it here explicitly.)

  void initialize(size_t n_instances, size_t n_features);
  void append_data(const float *ptr, size_t n_new_instances);
  GPUMAPResult run(size_t n_insertion_ops, size_t n_update_ops, size_t embedding_ops);
  void close();

  EmbedView get_embedding() const;

  // Get current configuration
  // const GPUMAPParams &get_config() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace gpumap
