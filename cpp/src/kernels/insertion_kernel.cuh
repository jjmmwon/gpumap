
#pragma once

#include "nndescent/initialize.cuh"
#include "queue.cuh"
#include "raft/util/cuda_rt_essentials.hpp"
#include <vector>

namespace gpumap {
template <typename Data_t, typename Idx_t>
void init_nndescent(Data_t *input_data, Idx_t *neighbor_indices,
                    Data_t *neighbor_distances, Idx_t n_start, Idx_t n_end,
                    Idx_t neighbor_indices_block_size, Idx_t neighbor_dists_block_size,
                    Idx_t n_neighbors, Idx_t n_instances,
                    Idx_t n_inserted_instances, Idx_t n_features, size_t seed,
                    cudaStream_t stream) {

  size_t n_rows = n_end - n_start;
  // Initialize random neighbors
  const int threads_per_block = 256;
  const int init_random_neighbors_blocks =
      (n_rows + threads_per_block - 1) / threads_per_block;

  // Calculate L2 norms squared
  const int calculate_l2_norms_blocks = static_cast<int>(n_rows);
  const int shared_mem_size = threads_per_block * sizeof(Data_t);

  nndescent::preprocess_l2_norms_kernel<Data_t, Idx_t>
      <<<calculate_l2_norms_blocks, threads_per_block, shared_mem_size,
         stream>>>(input_data, n_start, n_end, n_features);
  RAFT_CUDA_TRY(cudaGetLastError());

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  // Initialize random neighbors
  nndescent::initialize_random_neighbors_kernel<Data_t, Idx_t, 32>
      <<<init_random_neighbors_blocks, threads_per_block, 0, stream>>>(
          neighbor_indices, neighbor_distances, n_start, n_end, n_neighbors,
          neighbor_indices_block_size, neighbor_dists_block_size, n_instances, n_inserted_instances, seed);
  RAFT_CUDA_TRY(cudaGetLastError());

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Data_t, typename Idx_t>
void insertion_kernel(
    Data_t *input_data, Idx_t *neighbor_indices, Data_t *neighbor_distances,
    Data_t *embedding, Idx_t neighbor_indices_block_size, Idx_t neighbor_dists_block_size,
    Idx_t n_neighbors,
    Idx_t n_instances, Idx_t n_inserted_instances, Idx_t n_features,
    const std::vector<Idx_t> &batch_indices,
    std::vector<Idx_t> &affected_indices,
    std::vector<std::tuple<Idx_t, Idx_t, Idx_t>> &new_edges,
    cudaStream_t insertion_stream, size_t seed = 42) {

  const Idx_t n_start = batch_indices.front();
  const Idx_t n_end = batch_indices.back() + 1;

  init_nndescent<Data_t, Idx_t>(
      input_data, neighbor_indices, neighbor_distances, n_start, n_end,
      neighbor_indices_block_size, neighbor_dists_block_size, n_neighbors,
      n_instances, n_inserted_instances, n_features,
      seed, insertion_stream);

  RAFT_CUDA_TRY(cudaStreamSynchronize(insertion_stream));

  // init_embedding<Data_t,
  //                Idx_t>( // embedding을 batch_indices의 이웃의
  //                        // 평균으로 초기화 하고 distances를 기반으로
  //                        // simplicial set embedding 업데이트
  //     embedding, neighbor_indices, neighbor_distances, n_start, n_end,
  //     neighbor_indices_block_size, n_neighbors, n_inserted_instances, n_features,
  //     batch_indices, affected_indices, new_edges, insertion_stream);

  // affected_indices에 batch_indices의 이웃들을
  // unique하게 추가 new_edges에 batch_indices의 이웃 간의
  // 에지 (i, n_0), (i, n_k-1) 추가
}

} // namespace gpumap
