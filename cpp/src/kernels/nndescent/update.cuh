#pragma once

#include "buffers.cuh"
#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

namespace gpumap::nndescent {
template <typename Data_t, typename Idx_t, typename Index_t = int>
class UpdateAlgo {
public:
  UpdateAlgo() = delete;
  UpdateAlgo(const UpdateAlgo &) = delete;
  UpdateAlgo &operator=(const UpdateAlgo &) = delete;

  UpdateAlgo(NNDBuffers<Data_t, Index_t> &buffers,
             std::vector<Idx_t> batch_indices,

             Data_t *input_data, Idx_t *neighbor_indices,
             Data_t *neighbor_distances, uint32_t data_block_size,
             uint32_t neighbor_indices_block_size,
             uint32_t neighbor_dists_block_size, uint32_t n_instances,
             uint32_t n_features, uint32_t n_neighbors,
             uint32_t n_inserted_instances, uint32_t max_iter = 30,
             float termination_threshold = 0.001f)
      : buffers_(buffers), update_counter_{0},
        batch_indices_(std::move(batch_indices)),

        input_data(input_data), neighbor_indices(neighbor_indices),
        neighbor_distances(neighbor_distances),
        data_block_size_(data_block_size),
        neighbor_indices_block_size_(neighbor_indices_block_size),
        neighbor_dists_block_size_(neighbor_dists_block_size),

        n_instances_(n_instances), n_features_(n_features),
        n_neighbors_(n_neighbors), n_inserted_instances_(n_inserted_instances),
        max_iter_(max_iter), termination_threshold_(termination_threshold) {}

  std::vector<Idx_t> run(cudaStream_t const stream);
  void local_join(cudaStream_t const stream);
  void add_reverse_edges(Index_t *graph_ptr, Index_t *h_rev_graph_ptr,
                         Index_t *d_rev_graph_ptr, int2 *list_sizes,
                         cudaStream_t stream);

  void sample_graph_new();
  void sample_graph(bool sample_new);
  void update_graph(std::atomic<int64_t> &update_counter);
  void sort_lists();

private:
  NNDBuffers<Data_t, int> &buffers_;
  std::atomic<int64_t> update_counter_;
  // Row mapping for partial update mode: local row i -> global row
  // batch_indices_[i].
  std::vector<Idx_t> batch_indices_;

  Data_t *input_data;
  Idx_t *neighbor_indices;
  Data_t *neighbor_distances;

  uint32_t data_block_size_;
  uint32_t neighbor_indices_block_size_;
  uint32_t neighbor_dists_block_size_;

  uint32_t n_instances_;
  uint32_t n_features_;
  uint32_t n_neighbors_;
  uint32_t n_inserted_instances_;
  uint32_t max_iter_;
  float termination_threshold_;
};

} // namespace gpumap::nndescent