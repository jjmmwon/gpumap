#pragma once

#include "nndescent/buffers.cuh"
#include "nndescent/update_detail.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <tuple>
#include <vector>

namespace gpumap {

namespace nnd = nndescent;

template <typename Data_t, typename Idx_t, typename Index_t = int>
void update_kernel(

    nnd::NNDBuffers<Data_t, Index_t> &buffers, std::vector<Idx_t> batch_indices,
    std::vector<Idx_t> &affected_indices,
    std::vector<std::tuple<Idx_t, Idx_t, float, float, float>> &new_edges,
    cudaStream_t update_stream,

    Data_t *input_data, Idx_t *neighbor_indices, Data_t *neighbor_distances,
    uint32_t data_block_size, uint32_t neighbor_block_size,
    uint32_t neighbor_dists_block_size,

    uint32_t n_instances, uint32_t n_features, uint32_t n_neighbors,
    size_t n_inserted_instances,

    uint32_t max_iter = 30, float termination_threshold = 0.001f,
    size_t seed = 42

) {
  if (batch_indices.empty()) {
    return;
  }

  raft::resources res;
  raft::resource::set_cuda_stream(res, update_stream);

  nndescent::UpdateAlgo<Data_t, Idx_t> updater(
      buffers, batch_indices, input_data, neighbor_indices, neighbor_distances,
      data_block_size, neighbor_block_size, neighbor_dists_block_size,
      n_instances, n_features, n_neighbors, n_inserted_instances, max_iter,
      termination_threshold);

  updater.run(update_stream, affected_indices, new_edges);

  // neighbor_changed_indices를 embedding_queue에 추가
}
} // namespace gpumap
