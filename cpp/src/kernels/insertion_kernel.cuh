
#pragma once

#include "nndescent/buffers.cuh"
#include "nndescent/insertion_detail.cuh"
#include "raft/util/cuda_rt_essentials.hpp"
#include <cstdint>
#include <vector>

namespace gpumap {
template <typename Data_t, typename Idx_t>
void insertion_kernel(
    nndescent::NNDBuffers<Data_t, int> &buffers,
    const std::vector<Idx_t> &batch_indices,

    Data_t *input_data, Idx_t *neighbor_indices, Data_t *neighbor_distances,
    Data_t *embedding,

    uint32_t data_block_size, uint32_t neighbor_indices_block_size,
    uint32_t neighbor_dists_block_size, uint32_t n_components,
    uint32_t n_instances, uint32_t n_features, uint32_t n_neighbors,
    uint32_t n_inserted_instances, std::vector<Idx_t> &affected_indices,
    std::vector<std::tuple<Idx_t, Idx_t, float, float, float>> &new_edges,
    cudaStream_t insertion_stream, size_t seed = 42) {

  nndescent::InsertionAlgo<Data_t, Idx_t> runner(
      buffers, batch_indices, input_data, neighbor_indices, neighbor_distances,
      embedding, data_block_size, neighbor_indices_block_size,
      neighbor_dists_block_size, n_components, n_instances, n_features,
      n_neighbors, n_inserted_instances, /*max_iter=*/3,
      /*termination_threshold=*/0.001f, seed);

  runner.run(insertion_stream, affected_indices, new_edges);
}

} // namespace gpumap
