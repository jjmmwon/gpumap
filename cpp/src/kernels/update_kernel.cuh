#pragma once

#include "nndescent/update.cuh"
#include "raft/util/cuda_rt_essentials.hpp"
#include <__clang_cuda_runtime_wrapper.h>

namespace gpumap {
template <typename Data_t, typename Idx_t>
void update_kernel(
    std::vector<size_t> batch_indices, Data_t *input_data,
    Idx_t *neighbor_indices, Data_t *neighbor_distances,
    size_t neighbor_indices_block_size, size_t neighbors,
    size_t total_instances, size_t n_features, size_t seed,

    std::vector<std::tuple<size_t, size_t, size_t>> affected_edges,
    cudaStream_t update_stream) {
  // TODO:
  // update nndescent for batch_indices
  // update affected_edges based on updated neighbor_indices and
  // neighbor_distances
}
} // namespace gpumap