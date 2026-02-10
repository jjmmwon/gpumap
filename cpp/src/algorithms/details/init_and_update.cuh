#pragma once

#include "nndescent.cuh"
#include "raft/util/cuda_rt_essentials.hpp"

namespace gpumap::detail {
void init_and_update_kernel() {
  // Implementation of the kernel
}

template <typename Data_t, typename Idx_t>
void init_nndescent(Data_t *input_data, Idx_t *neighbor_indices,
                    size_t n_instances, size_t total_instances,
                    size_t n_neighbors, size_t n_dim, unsigned long long seed,
                    cudaStream_t stream) {

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
  nndescent::calculate_l2_norms_sqr_kernel<Data_t, Idx_t>
      <<<calculate_l2_norms_blocks, threads_per_block, shared_mem_size,
         stream>>>(input_data, n_instances, n_dim);
  RAFT_CUDA_TRY(cudaGetLastError());

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}
} // namespace gpumap::detail