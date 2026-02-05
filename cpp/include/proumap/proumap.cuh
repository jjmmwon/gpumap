// include/proumap/proumap.cuh
#pragma once

#include <cstdint>
#include <memory>

// Public return types (lightweight views)
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>

namespace proumap {

struct UMAPConfig {
  int n_neighbors = 15;
  int n_components = 2;
  float min_dist = 0.1f;
  int n_epochs = 200;
  int negative_sample_rate = 5;
  bool verbose = false;
};

struct ProUMAPConfig : public UMAPConfig {
  int max_instances = 1000000;
  int batch_size_hint = 1024;
  int n_features = 0;
};

// Public, non-template API (bind this from pybind).
// Internals (StorageManager, streams, RMM, etc.) stay in src via PImpl.
class ProUMAP {
public:
  using Data_t = float;
  using Idx_t = uint32_t;
  using DataView =
      raft::device_matrix_view<const Data_t, Idx_t, raft::row_major>;
  using EmbedView =
      raft::device_matrix_view<const Data_t, Idx_t, raft::row_major>;

  ProUMAP();
  ProUMAP(const ProUMAPConfig &config);

  // Move-only
  ProUMAP(const ProUMAP &) = delete;
  ProUMAP &operator=(const ProUMAP &) = delete;
  ProUMAP(ProUMAP &&) noexcept;
  ProUMAP &operator=(ProUMAP &&) noexcept;

  ~ProUMAP();

  // Host pointer input: binding layer should guarantee contiguous layout.
  // (If you want to accept stride/ld, add it here explicitly.)
  void append_data(const float *ptr, Idx_t n_new_rows, Idx_t n_new_cols);

  void run_nng_build();
  void run_optimize_embeddings();

  DataView get_data_view() const;
  EmbedView get_embedding_view() const;

  // Get current configuration
  const ProUMAPConfig &get_config() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace proumap
