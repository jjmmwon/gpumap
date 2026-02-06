#pragma once

#include "proumap/proumap.cuh"
#include "storage/matrix_utils.cuh"
#include "storage/storage_manager.cuh"

namespace proumap {

struct ProUMAP::Impl {
  using Data_t = float;
  using Idx_t = uint32_t;

  ProUMAPConfig config;

  // queue obj
  storage::StorageManager<Data_t, Idx_t> storage_manager;

  Impl(const ProUMAPConfig &config_)
      : config(config_),
        storage_manager(config_.max_instances, config_.n_features,
                        config_.n_neighbors, 4) {}

  void append_data(const Data_t *ptr, Idx_t n_new_rows, Idx_t n_new_cols) {
    storage::utils::append_host(storage_manager.get_data_manage_resources(),
                                storage_manager.progressive_data, ptr,
                                n_new_rows);
  }

  void run_nng_build(int n_items) {
    // NNG build implementation placeholder
  }
  void run_embedding_optimization(int n_items) {
    // Embedding optimization implementation placeholder
  }

  auto get_data_view() const
      -> raft::device_matrix_view<const Data_t, Idx_t, raft::row_major> {
    return storage_manager.get_data_view();
  }

  auto get_embedding_view() const
      -> raft::device_matrix_view<const Data_t, Idx_t, raft::row_major> {
    return storage_manager.get_embedding_view();
  }
};
} // namespace proumap
