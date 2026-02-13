#pragma once

#include "components.cuh"
#include "utils.cuh"

#include <__clang_cuda_runtime_wrapper.h>
#include <atomic>
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <unordered_map>
#include <vector>

namespace gpumap::nndescent {

template <typename Data_t, typename Idx_t>
NNDGraph<Data_t, Idx_t>::NNDGraph(raft::resources const &res,
                                  const size_t n_row, const size_t n_neighbors,
                                  const size_t n_internal_neighbors,
                                  const size_t n_inserted_instances,
                                  const size_t num_samples)
    : n_row(n_row), n_neighbors(n_neighbors),
      n_inserted_instances_(n_inserted_instances),
      n_internal_neighbors(n_internal_neighbors), num_samples(num_samples),
      bloom_filter(n_row, 4, 4),
      h_dists{raft::make_host_matrix<Data_t, size_t, raft::row_major>(
          n_row, n_neighbors)},
      h_graph_new{raft::make_pinned_matrix<Idx_t, size_t, raft::row_major>(
          res, n_row, num_samples)},
      h_list_sizes_new{raft::make_pinned_vector<int2, size_t>(res, n_row)},
      h_graph_old{raft::make_pinned_matrix<Idx_t, size_t, raft::row_major>(
          res, n_row, num_samples)},
      h_list_sizes_old{raft::make_pinned_vector<int2, size_t>(res, n_row)} {
  h_graph = nullptr;
};

template <typename Data_t, typename Idx_t>
void NNDGraph<Data_t, Idx_t>::sample_graph_new(InternalID_t *new_neighbors,
                                               const size_t width) {
  std::fill_n(h_graph_new.data_handle(), n_row * num_samples,
              std::numeric_limits<Idx_t>::max());
#pragma omp parallel for
  for (size_t i = 0; i < n_row; ++i) {
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    auto list_new = h_graph_new.data_handle() + i * num_samples;

    for (size_t j = 0; j < width; ++j) {
      auto new_neighbor_id = new_neighbors[i * width + j].id();
      if ((size_t)new_neighbor_id >= n_inserted_instances_)
        break;
      if (bloom_filter.check(i, (Idx_t)new_neighbor_id))
        continue;
      bloom_filter.add(i, (Idx_t)new_neighbor_id);
      new_neighbors[i * width + j].mark_old();
      list_new[h_list_sizes_new.data_handle()[i].x++] = new_neighbor_id;
      if (h_list_sizes_new.data_handle()[i].x == num_samples)
        break;
    }
  };
};

template <typename Data_t, typename Idx_t>
void NNDGraph<Data_t, Idx_t>::sample_graph(bool sample_new) {
  std::fill_n(h_graph_old.data_handle(), n_row * num_samples,
              std::numeric_limits<Idx_t>::max());
  if (sample_new) {
    std::fill_n(h_graph_new.data_handle(), n_row * num_samples,
                std::numeric_limits<Idx_t>::max());
  };
#pragma omp parallel for
  for (size_t i = 0; i < n_row; i++) {
    h_list_sizes_old.data_handle()[i].x = 0;
    h_list_sizes_old.data_handle()[i].y = 0;
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    auto list = h_graph + i * n_internal_neighbors;
    auto list_old = h_graph_old.data_handle() + i * num_samples;
    auto list_new = h_graph_new.data_handle() + i * num_samples;
    for (int j = 0; j < n_internal_neighbors; j++) {
      auto neighbor = list[j];
      if ((size_t)neighbor.id() >= n_inserted_instances_)
        continue;
      if (!neighbor.is_new()) {
        if (h_list_sizes_old.data_handle()[i].x < num_samples) {
          list_old[h_list_sizes_old.data_handle()[i].x++] = neighbor.id();
        }
      } else if (sample_new) {
        if (h_list_sizes_new.data_handle()[i].x < num_samples) {
          list[j].mark_old();
          list_new[h_list_sizes_new.data_handle()[i].x++] = neighbor.id();
        }
      }
      if (h_list_sizes_old.data_handle()[i].x == num_samples &&
          h_list_sizes_new.data_handle()[i].x == num_samples) {
        break;
      }
    }
    if (h_list_sizes_old.data_handle()[i].x == num_samples &&
        h_list_sizes_new.data_handle()[i].x == num_samples) {
      break;
    }
  }
};

template <typename Data_t, typename Idx_t>
void NNDGraph<Data_t, Idx_t>::update_graph(
    const InternalID_t *new_neighbors, const Data_t *new_dists,
    const size_t width, std::atomic<int64_t> &update_counter) {}

// template <typename Data_t, typename Idx_t, size_t NUM_SAMPLES, size_t
// MAX_ITER>

// class NNDescentUpdate {
// public:
//   NNDescentUpdate(std::vector<size_t> batch_indices, Data_t *input_data,
//                   Idx_t *neighbor_indices, Data_t *neighbor_distances,
//                   BloomFilter<Idx_t> &bloom_filter, // for checking
//                   duplicates size_t neighbor_indices_block_size, size_t
//                   n_internal_neighbors, size_t n_inserted_instances,
//                   size_t n_features, size_t seed, cudaStream_t
//                   update_stream)
//       : batch_indices_(batch_indices), input_data_(input_data),
//         neighbor_indices_(neighbor_indices),
//         neighbor_distances_(neighbor_distances),
//         bloom_filter_(bloom_filter),
//         neighbor_indices_block_size_(neighbor_indices_block_size),
//         n_internal_neighbors_(n_internal_neighbors),
//         n_inserted_instances_(n_inserted_instances),
//         n_features_(n_features), seed_(seed),
//         update_stream_(update_stream)
//         {

//     initialize(); // load neighbor indices and distances to host
//   }

//   void
//   run_single(std::vector<size_t> batch_indices,
//              std::vector<std::tuple<size_t, size_t, size_t>>
//              &affected_edges);

// private:
//   std::vector<size_t> batch_indices_;
//   std::unordered_map<size_t, size_t> batch_indices_map_;
//   Data_t *input_data_;
//   Idx_t *neighbor_indices_;
//   Data_t *neighbor_distances_;
//   BloomFilter<Idx_t> &bloom_filter_;
//   size_t neighbor_indices_block_size_;
//   size_t n_neighbors_;
//   size_t n_internal_neighbors_;
//   size_t n_inserted_instances_;
//   size_t n_features_;
//   size_t seed_;
//   cudaStream_t update_stream_;
//   raft::resources res_;

//   std::atomic<int64_t> update_counter_;

//   // memory for host-side processing
//   raft::host_matrix<InternalID_t, size_t> h_graph;
//   raft::host_matrix<Data_t, size_t> h_dists;
//   raft::pinned_matrix<int, size_t> h_graph_new; // new neighbor indices
//   raft::pinned_vector<int2, size_t> h_lists_sizes_new;
//   raft::pinned_matrix<int, size_t> h_graph_old; // old neighbor indices
//   raft::pinned_vector<int2, size_t> h_lists_sizes_old;

//   raft::pinned_matrix<Idx_t, size_t> graph_host_buffer_;
//   raft::pinned_matrix<Data_t, size_t> dists_host_buffer_;

//   raft::pinned_matrix<Idx_t, size_t> h_rev_graph_new_;
//   raft::pinned_matrix<Idx_t, size_t> h_rev_graph_old_;

//   void initialize() {
//     raft::resource::set_cuda_stream(res_, update_stream_);

//     batch_indices_map_.clear();
//     for (size_t i = 0; i < batch_indices_.size(); ++i) {
//       batch_indices_map_[batch_indices_[i]] = i;
//     }

//     size_t n_rows = batch_indices_.size();

//     h_graph = raft::make_host_matrix<InternalID_t, size_t>(
//         n_rows, n_internal_neighbors_);

//     auto h_graph_tmp =
//         raft::make_host_matrix<Idx_t, size_t>(n_rows,
//         n_internal_neighbors_);

//     h_dists =
//         raft::make_host_matrix<Data_t, size_t>(n_rows,
//         n_internal_neighbors_);
//     h_graph_new =
//         raft::make_pinned_matrix<int, size_t>(res_, n_rows, NUM_SAMPLES);
//     h_lists_sizes_new = raft::make_pinned_vector<int2, size_t>(res_,
//     n_rows); h_graph_old =
//         raft::make_pinned_matrix<int, size_t>(res_, n_rows, NUM_SAMPLES);
//     h_lists_sizes_old = raft::make_pinned_vector<int2, size_t>(res_,
//     n_rows);

//     // Copy scattered rows from device to host (one async memcpy per row)
//     size_t row_bytes_idx = n_internal_neighbors_ * sizeof(Idx_t);
//     size_t row_bytes_dist = n_internal_neighbors_ * sizeof(Data_t);

//     for (size_t i = 0; i < n_rows; ++i) {
//       size_t global_row_idx = batch_indices_[i] *
//       neighbor_indices_block_size_; raft::copy(h_graph_tmp.data_handle()
//       +
//       i
//       * n_internal_neighbors_,
//                  neighbor_indices_ + global_row_idx,
//                  n_internal_neighbors_, update_stream_);
//       raft::copy(h_dists.data_handle() + i * n_internal_neighbors_,
//                  neighbor_distances_ + global_row_idx,
//                  n_internal_neighbors_, update_stream_);
//     }
//     cudaStreamSynchronize(update_stream_);

// #pragma omp parallel for
//     for (size_t i = 0; i < n_rows * n_internal_neighbors_; ++i) {
//       h_graph.data_handle()[i] =
//       InternalID_t(h_graph_tmp.data_handle()[i]);
//     }
//   };

//   void build() {
//     // graph_new
//     // graph_old
//     // fwd_sizes_new
//     // fwd_sizes_old
//     // rev_sizes_new
//     // rev_sizes_old
//     sample_graph(true);

//     for (size_t it = 0; it < MAX_ITER; ++it) {
//       raft::copy(d_list_sizes_new_) update_sample_graph();
//       add_reverse_edges();
//       local_join_graph();
//       update_neighbor_info_device();
//     }
//   };

//   void sample_graph(bool sample_new) {
//     size_t n_rows = batch_indices_.size();
//     std::fill_n(h_graph_old.data_handle(), n_rows * NUM_SAMPLES,
//                 std::numeric_limits<int>::max());
//     if (sample_new) {
//       std::fill_n(h_graph_new.data_handle(), n_rows * NUM_SAMPLES,
//                   std::numeric_limits<int>::max());
//     };

// #pragma omp parallel for
//     for (size_t i = 0; i < n_rows; ++i) {
//       h_lists_sizes_old.data_handle()[i].x = 0;
//       h_lists_sizes_old.data_handle()[i].y = 0;
//       h_lists_sizes_new.data_handle()[i].x = 0;
//       h_lists_sizes_new.data_handle()[i].y = 0;

//       auto list = h_graph.data_handle() + i * n_internal_neighbors_;
//       auto list_old = h_graph_old.data_handle() + i * NUM_SAMPLES;
//       auto list_new = h_graph_new.data_handle() + i * NUM_SAMPLES;

//       for (int j = 0; j < n_internal_neighbors_; ++j) {
//         InternalID_t neighbor_id = list[j];
//         if (neighbor_id.id() > n_inserted_instances_)
//           continue;
//         if (!neighbor_id.is_new()) {
//           if (h_lists_sizes_old.data_handle()[i].x < NUM_SAMPLES) {
//             list_old[h_lists_sizes_old.data_handle()[i].x++] =
//             neighbor_id.id();
//           }
//         } else if (sample_new) {
//           if (h_lists_sizes_new.data_handle()[i].x < NUM_SAMPLES) {
//             neighbor_id.mark_old();
//             list_new[h_lists_sizes_new.data_handle()[i].x++] =
//             neighbor_id.id();
//           }
//         }
//         if (h_lists_sizes_old.data_handle()[i].x == NUM_SAMPLES &&
//             h_lists_sizes_new.data_handle()[i].x == NUM_SAMPLES) {
//           break;
//         }
//       };
//       if (h_lists_sizes_old.data_handle()[i].x == NUM_SAMPLES &&
//           h_lists_sizes_new.data_handle()[i].x == NUM_SAMPLES) {
//         break;
//       }
//     };
//   }

//   void sample_graph_new() {

//   };

//   void update_graph(const InternalID_t *new_neighbors, const Data_t
//   *new_dists,
//                     std::atomic<int64_t> &update_counter) {
//     for (size_t i = 0; i < batch_indices_.size(); ++i) {
//       size_t global_row_idx = batch_indices_[i] *
//       neighbor_indices_block_size_;

//       auto h_graph = h_graph.data_handle() + i * n_internal_neighbors_;
//       auto dist_list = h_dists.data_handle() + i * n_internal_neighbors_;

//       Data_t worst = dist_list[n_internal_neighbors_ - 1];

//       for (size_t j = 0; j < NUM_SAMPLES; ++j) {
//         InternalID_t new_id = new_neighbors[i * NUM_SAMPLES + j];
//         Data_t new_dist = new_dists[i * NUM_SAMPLES + j];

//         if (new_dist == std::numeric_limits<Data_t>::max())
//           break;
//         if (static_cast<size_t>(new_id.id()) ==
//             batch_indices_[i]) // skip self-loops
//           continue;

//         if (new_dist >= worst)
//           break;

//         if (utils::insert_neighbor_sorted<Data_t, Idx_t>(
//                 reinterpret_cast<Idx_t *>(h_graph), dist_list,
//                 static_cast<int>(n_internal_neighbors_),
//                 static_cast<Idx_t>(new_id.id()), new_dist)) {
//           worst = dist_list[n_internal_neighbors_ - 1];
//           update_counter++;
//         }
//       }
//     }

//     void update_sample_graph();
//     void add_reverse_edges();
//     void local_join_graph();
//     void update_neighbor_info_device();

} // namespace gpumap::nndescent