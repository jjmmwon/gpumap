#pragma once

#include "proumap/proumap.cuh"
#include "proumap_queue.cuh"
#include "proumap_storage.cuh"

namespace proumap {

struct ProUMAP::Impl {
  using Data_t = float;
  using Idx_t = uint32_t;

  ProUMAPConfig config;

  storage::Storage<Data_t, Idx_t> storage;

  Impl(const ProUMAPConfig &config_)
      : config(config_), storage(config_.max_instances, config_.n_features,
                                 config_.n_neighbors) {}

  void append_data(const Data_t *ptr, Idx_t n_new_rows, Idx_t n_new_cols) {
    storage.data.append_device(ptr, n_new_rows, rmm::cuda_stream_default);
  }

  void run_nng_build() {
    // NNG build implementation placeholder
  }
  void run_embedding_optimization() {
    // Embedding optimization implementation placeholder
  }

  auto get_data_view() const
      -> raft::device_matrix_view<const Data_t, Idx_t, raft::row_major> {
    return storage.data.view();
  }

  auto get_embedding_view() const
      -> raft::device_matrix_view<const Data_t, Idx_t, raft::row_major> {
    return storage.embedding.view();
  }
};
} // namespace proumap
