
#pragma once

#include "nndescent/initialize.cuh"
#include "queue.cuh"
#include "raft/util/cuda_rt_essentials.hpp"
#include <__clang_cuda_runtime_wrapper.h>
#include <vector>

namespace gpumap {
template <typename Data_t, typename Idx_t>
void init_nndescent(Data_t *input_data, Idx_t *neighbor_indices,
                    Data_t *neighbor_distances, Idx_t n_start, Idx_t n_end,
                    size_t neighbor_indices_block_size, size_t n_neighbors,
                    size_t total_instances, size_t n_features, size_t seed,
                    cudaStream_t stream) {

  size_t n_instances = n_end - n_start;
  // Initialize random neighbors
  const int threads_per_block = 256;
  const int init_random_neighbors_blocks =
      (n_instances + threads_per_block - 1) / threads_per_block;

  // Calculate L2 norms squared
  const int calculate_l2_norms_blocks = static_cast<int>(n_instances);
  const int shared_mem_size = threads_per_block * sizeof(Data_t);

  nndescent::preprocess_l2_norms_kernel<Data_t, Idx_t>
      <<<calculate_l2_norms_blocks, threads_per_block, shared_mem_size,
         stream>>>(input_data, n_start, n_end, n_features);
  RAFT_CUDA_TRY(cudaGetLastError());

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  // Initialize random neighbors
  nndescent::initialize_random_neighbors_kernel<Idx_t>
      <<<init_random_neighbors_blocks, threads_per_block, 0, stream>>>(
          neighbor_indices, neighbor_distances, n_start, n_end, n_neighbors,
          neighbor_indices_block_size, total_instances, seed);
  RAFT_CUDA_TRY(cudaGetLastError());

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Data_t, typename Idx_t>
void insertion_kernel(
    Data_t *input_data, Idx_t *neighbor_indices, Data_t *neighbor_distances,
    Data_t *embedding, size_t neighbor_indices_block_size, Idx_t n_neighbors,
    size_t n_instances, size_t total_instances, size_t n_features,
    const std::vector<size_t> &batch_indices,
    std::vector<size_t> &affected_indices,
    std::vector<std::tuple<size_t, size_t, size_t>> &new_edges,
    cudaStream_t insertion_stream, size_t seed = 42) {
  const size_t n_start = batch_indices.front();
  const size_t n_end = batch_indices.back() + 1;

  init_nndescent<Data_t, Idx_t>(
      input_data, neighbor_indices, neighbor_distances, n_start, n_end,
      neighbor_indices_block_size, n_neighbors, total_instances, n_features,
      seed, insertion_stream);

  RAFT_CUDA_TRY(cudaStreamSynchronize(insertion_stream));

  init_embedding<Data_t,
                 Idx_t>( // embedding을 batch_indices의 이웃의
                         // 평균으로 초기화 하고 distances를 기반으로
                         // simplicial set embedding 업데이트
      embedding, neighbor_indices, neighbor_distances, n_start, n_end,
      neighbor_indices_block_size, n_neighbors, n_instances, n_features,
      batch_indices, affected_indices, new_edges, insertion_stream);

  // affected_indices에 batch_indices의 이웃들을
  // unique하게 추가 new_edges에 batch_indices의 이웃 간의
  // 에지 (i, n_0), (i, n_k-1) 추가
}

} // namespace gpumap
