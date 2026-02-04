#pragma once

#include "storage/storage_manager.cuh"
#include "stroage/matrix2D.cuh"

namespace proumap {
struct UMAPConfig {
  int n_neighbors = 15;
  int n_components = 2;
  float min_dist = 0.1f;
  int n_epochs = 200;
  float learning_rate = 1.0f;
  bool verbose = false;
};

template <typename Data_t = float, typename Idx_t = uint32_t> class ProUMAP {
public:
  UMAPConfig config;
  int max_instances = 1000000;
  int batch_size_hint = 1024;
  int n_features;
  storage::StorageManager<Data_t, Idx_t> storage_manager;

  ProUMAP() = default;
  ProUMAP(const UMAPConfig &config_, int max_instances_, int batch_size_hint_,
          int n_features_)
      : config(config_), max_instances(max_instances_),
        batch_size_hint(batch_size_hint_), n_features(n_features_),
        storage_manager(max_instances_, n_features_, config.n_neighbors, 4) {}

  // Move-only semantics for
  ProUMAP(const ProUMAP &) = delete;
  ProUMAP &operator=(const ProUMAP &) = delete;
  ProUMAP(ProUMAP &&) = default;
  ProUMAP &operator=(ProUMAP &&) = default;

  ~ProUMAP() = default;

  auto get_embedding_view() const
      -> raft::device_matrix_view<Data_t, Idx_t, raft::row_major> {
    return storage_manager.get_embedding_view();
  }

  auto get_data_view() const
      -> raft::device_matrix_view<Data_t, Idx_t, raft::row_major> {
    return storage_manager.get_data_view();
  }
};
} // namespace proumap