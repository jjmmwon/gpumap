
#pragma once

#include "nndescent/initialize.cuh"
#include "raft/util/cuda_rt_essentials.hpp"
#include <__clang_cuda_runtime_wrapper.h>
#include <vector>

namespace gpumap {

template <typename Data_t, typename Idx_t>
void insertion_kernel(Data_t *input_data, Idx_t *neighbor_indices,
                      float *neighbor_distances, Data_t *embedding,

                      Idx_t n_neighbors, size_t n_instances,

                      const std::vector<size_t> &batch_indices,
                      std::vector<size_t> &affected_indices,
                      std::vector<std::tuple<size_t, size_t, float>> &new_edges,

                      cudaStream_t *insertion_stream,
                      cudaStream_t *update_stream,
                      cudaStream_t *embedding_stream) {}

template <typename Data_t, typename Idx_t>
void init_nndescent(Data_t *input_data, Idx_t *neighbor_indices,
                    size_t n_instances, size_t total_instances,
                    size_t n_neighbors, size_t n_features,
                    unsigned long long seed, cudaStream_t stream) {

  // Initialize random neighbors
  const int threads_per_block = 256;
  const int init_random_neighbors_blocks =
      (n_instances + threads_per_block - 1) / threads_per_block;

  nndescent::initialize_random_neighbors_kernel<Idx_t>
      <<<init_random_neighbors_blocks, threads_per_block, 0, stream>>>(
          neighbor_indices, n_instances, n_neighbors, total_instances, seed);
  RAFT_CUDA_TRY(cudaGetLastError());

  // Calculate L2 norms squared
  const int calculate_l2_norms_blocks = static_cast<int>(n_instances);
  const int shared_mem_size = threads_per_block * sizeof(Data_t);
  nndescent::preprocess_l2_norms_kernel<Data_t, Idx_t>
      <<<calculate_l2_norms_blocks, threads_per_block, shared_mem_size,
         stream>>>(input_data, n_instances, n_features);
  RAFT_CUDA_TRY(cudaGetLastError());

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

} // namespace gpumap
