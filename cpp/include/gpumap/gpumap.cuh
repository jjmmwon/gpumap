// include/gpumap/gpumap.cuh
#pragma once

#include <cstdint>
#include <memory>

// Public return types (lightweight views)
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>

namespace gpumap {

struct UMAPParams {
  size_t n_neighbors = 15;
  size_t n_components = 2;
  float min_dist = 0.1f;
  size_t n_epochs = 200;
  size_t negative_sample_rate = 5;
  bool verbose = false;
};

struct GPUMAPParams : public UMAPParams {
  size_t n_instances;
  size_t n_features;

  GPUMAPParams(size_t n_instances_, size_t n_features_)
      : n_instances(n_instances_), n_features(n_features_) {}

  // size_t batch_size_hint = 1024;
};

// Public, non-template API (bind this from pybind).
// Internals (StorageManager, streams, RMM, etc.) stay in src via PImpl.
class GPUMAP {
public:
  using Data_t = float;
  using Idx_t = uint32_t;
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
  void append_data(const float *ptr, Idx_t n_new_rows, Idx_t n_new_cols);

  void run_nng_build();
  void run_embedding_optimization();

  DataView get_data_view() const;
  EmbedView get_embedding_view() const;

  // Get current configuration
  const GPUMAPParams &get_config() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace gpumap
