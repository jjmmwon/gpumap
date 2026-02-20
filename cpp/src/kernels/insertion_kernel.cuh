
#pragma once

#include "nndescent/buffers.cuh"
#include "nndescent/insertion_detail.cuh"
#include "raft/util/cuda_rt_essentials.hpp"
#include <cstdint>
#include <vector>

namespace gpumap {
template <typename Data_t, typename Idx_t>
void insertion_kernel(nndescent::NNDBuffers<Data_t, int> &buffers,
                      const std::vector<Idx_t> &batch_indices,

                      Data_t *input_data, Idx_t *neighbor_indices,
                      Data_t *neighbor_distances, Data_t *embedding,

                      uint32_t data_block_size,
                      uint32_t neighbor_indices_block_size,
                      uint32_t neighbor_dists_block_size,
                      uint32_t embedding_block_size, uint32_t n_instances,
                      uint32_t n_features, uint32_t n_neighbors,
                      uint32_t n_inserted_instances,
                      std::vector<Idx_t> &affected_indices,
                      std::vector<std::tuple<Idx_t, Idx_t, Idx_t>> &new_edges,
                      cudaStream_t insertion_stream, size_t seed = 42) {

  nndescent::InsertionAlgo<Data_t, Idx_t> runner(
      buffers, batch_indices, input_data, neighbor_indices, neighbor_distances,
      embedding, data_block_size, neighbor_indices_block_size,
      neighbor_dists_block_size, embedding_block_size, n_instances, n_features,
      n_neighbors, n_inserted_instances, seed = seed);

  runner.run(insertion_stream, affected_indices, new_edges);

  RAFT_CUDA_TRY(cudaStreamSynchronize(insertion_stream));

  // init_embedding<Data_t,
  //                Idx_t>( // embedding을 batch_indices의 이웃의
  //                        // 평균으로 초기화 하고 distances를 기반으로
  //                        // simplicial set embedding 업데이트
  //     embedding, neighbor_indices, neighbor_distances, n_start, n_end,
  //     neighbor_indices_block_size, n_neighbors, n_inserted_instances,
  //     n_features, batch_indices, affected_indices, new_edges,
  //     insertion_stream);
}

} // namespace gpumap
