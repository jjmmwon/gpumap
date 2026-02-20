#pragma once

#include "buffers.cuh"

#include <utility>
#include <vector>

namespace gpumap::nndescent {

template <typename Data_t, typename Idx_t, typename Index_t = int>
class InsertionAlgo {
public:
  InsertionAlgo() = delete;
  InsertionAlgo(const InsertionAlgo &) = delete;
  InsertionAlgo &operator=(const InsertionAlgo &) = delete;

  InsertionAlgo(NNDBuffers<Data_t, Index_t> &buffers,
                std::vector<Idx_t> batch_indices,

                Data_t *input_data, Idx_t *neighbor_indices,
                Data_t *neighbor_distances, Data_t *embedding,
                uint32_t data_block_size, uint32_t neighbor_indices_block_size,
                uint32_t neighbor_dists_block_size,
                uint32_t embedding_block_size, uint32_t n_instances,
                uint32_t n_features, uint32_t n_neighbors,
                uint32_t n_inserted_instances, uint32_t max_iter = 30,
                float termination_threshold = 0.001f, size_t seed = 42)
      : buffers_(buffers), batch_indices_(std::move(batch_indices)),
        input_data(input_data), neighbor_indices(neighbor_indices),
        neighbor_distances(neighbor_distances), embedding(embedding),
        data_block_size_(data_block_size),
        neighbor_indices_block_size_(neighbor_indices_block_size),
        neighbor_dists_block_size_(neighbor_dists_block_size),
        embedding_block_size_(embedding_block_size), n_instances_(n_instances),
        n_features_(n_features), n_neighbors_(n_neighbors),
        n_inserted_instances_(n_inserted_instances), max_iter_(max_iter),
        termination_threshold_(termination_threshold), seed_(seed) {}

  void run(cudaStream_t const stream, std::vector<Idx_t> &affected_indices,
           std::vector<std::tuple<Idx_t, Idx_t, Idx_t>> &new_edges);
  void preprocess_l2_norms(cudaStream_t const stream);
  void initialize_random_neighbors(cudaStream_t const stream, size_t seed);
  void sample_forward_candidates(cudaStream_t const stream);
  void forward_nndescent(cudaStream_t const stream);
  void write_new_neighbors_info(cudaStream_t const stream);
  void initialize_embeddings();

private:
  NNDBuffers<Data_t, int> &buffers_;
  std::vector<Idx_t> batch_indices_;
  Data_t *input_data;
  Idx_t *neighbor_indices;
  Data_t *neighbor_distances;
  Data_t *embedding;

  uint32_t data_block_size_;
  uint32_t neighbor_indices_block_size_;
  uint32_t neighbor_dists_block_size_;
  uint32_t embedding_block_size_;

  uint32_t n_instances_;
  uint32_t n_features_;
  uint32_t n_neighbors_;
  uint32_t n_inserted_instances_;
  uint32_t max_iter_;
  float termination_threshold_;
  size_t seed_;
  cudaStream_t stream_{nullptr};
};
} // namespace gpumap::nndescent
