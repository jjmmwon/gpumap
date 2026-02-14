#pragma once

#include "components.cuh"
#include "utils.cuh"

#include <__clang_cuda_runtime_wrapper.h>
#include <atomic>
#include <cstddef>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <limits>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <unordered_map>
#include <vector>

namespace gpumap::nndescent {

template <typename Data_t, typename Idx_t>
NNDGraph<Data_t, Idx_t>::NNDGraph(raft::resources const &res,
                                  const size_t n_row, const size_t n_neighbors,
                                  const size_t n_internal_neighbors,
                                  const size_t n_inserted_instances,
                                  const size_t num_samples, const size_t seed)
    : n_row(n_row), n_neighbors(n_neighbors),
      n_inserted_instances(n_inserted_instances),
      n_internal_neighbors(n_internal_neighbors), num_samples(num_samples),
      seed(seed), bloom_filter(n_row, 4, 4),

      h_dists{raft::make_host_matrix<Data_t, size_t, raft::row_major>(
          n_row, n_internal_neighbors)},
      h_indices_new{raft::make_pinned_matrix<Idx_t, size_t, raft::row_major>(
          res, n_row, num_samples)},
      h_list_sizes_new{raft::make_pinned_vector<int2, size_t>(res, n_row)},
      h_indices_old{raft::make_pinned_matrix<Idx_t, size_t, raft::row_major>(
          res, n_row, num_samples)},
      h_list_sizes_old{raft::make_pinned_vector<int2, size_t>(res, n_row)} {
  h_indices = nullptr;
};

template <typename Data_t, typename Idx_t>
void NNDGraph<Data_t, Idx_t>::sample_graph_new(InternalID_t *new_neighbors,
                                               const size_t width) {
  std::fill_n(h_indices_new.data_handle(), n_row * num_samples,
              std::numeric_limits<Idx_t>::max());
#pragma omp parallel for
  for (size_t i = 0; i < n_row; ++i) {
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    auto list_new = h_indices_new.data_handle() + i * num_samples;

    for (size_t j = 0; j < width; ++j) {
      auto new_neighbor_id = new_neighbors[i * width + j].id();
      if ((size_t)new_neighbor_id >= n_inserted_instances)
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
  std::fill_n(h_indices_old.data_handle(), n_row * num_samples,
              std::numeric_limits<Idx_t>::max());
  if (sample_new) {
    std::fill_n(h_indices_new.data_handle(), n_row * num_samples,
                std::numeric_limits<Idx_t>::max());
  };
#pragma omp parallel for
  for (size_t i = 0; i < n_row; i++) {
    h_list_sizes_old.data_handle()[i].x = 0;
    h_list_sizes_old.data_handle()[i].y = 0;
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    auto list = h_indices + i * n_internal_neighbors;
    auto list_old = h_indices_old.data_handle() + i * num_samples;
    auto list_new = h_indices_new.data_handle() + i * num_samples;
    for (int j = 0; j < n_internal_neighbors; j++) {
      auto neighbor = list[j];
      if ((size_t)neighbor.id() >= n_inserted_instances)
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
    const size_t width, std::atomic<int64_t> &update_counter) {

#pragma omp parallel for
  for (size_t i = 0; i < n_row; ++i) {
    for (size_t j = 0; j < width; ++j) {
      auto new_id = new_neighbors[i * width + j];
      auto new_dist = new_dists[i * width + j];

      if (new_dist == std::numeric_limits<Data_t>::max())
        break;
      if (static_cast<size_t>(new_id.id()) == i) // skip self-loops
        continue;

      auto list = h_indices + i * n_internal_neighbors;
      auto dist_list = h_dists.data_handle() + i * n_internal_neighbors;

      Data_t worst = dist_list[n_internal_neighbors - 1];

      if (new_dist >= worst)
        break;

      if (utils::insert_neighbor_sorted<Data_t, InternalID_t *>(
              list, dist_list, n_internal_neighbors, new_id, new_dist)) {
        worst = dist_list[n_internal_neighbors - 1];
        update_counter++;
      }
    }
  }
}

template <typename Data_t, typename Idx_t>
void NNDUpdate<Data_t, Idx_t>::update_neighbor_info() {
  cudaStream_t stream = raft::resource::get_cuda_stream(res_);

  // InternalID_t -> Idx_t 변환을 위한 임시 호스트 버퍼
  std::vector<Idx_t> h_tmp_indices(n_internal_neighbors_);

  for (size_t i = 0; i < n_row_; ++i) {
    size_t global_row_idx_indices =
        batch_indices_[i] * neighbor_indices_block_size_;
    size_t global_row_idx_dist = batch_indices_[i] * neighbor_dists_block_size_;

    // InternalID_t -> Idx_t 변환
    for (size_t j = 0; j < n_internal_neighbors_; ++j) {
      h_tmp_indices[j] = static_cast<Idx_t>(
          graph_.h_indices.data_handle()[i * num_samples_ + j].id());
    }

    // 변환된 Idx_t 값을 디바이스의 d_neighbor_indices_에 복사
    raft::copy(d_neighbor_indices_ + global_row_idx_indices,
               h_tmp_indices.data(), n_internal_neighbors_, stream);

    // 거리 정보를 디바이스의 d_neighbor_distances_에 복사
    raft::copy(d_neighbor_distances_ + global_row_idx_dist,
               graph_.h_dists.data_handle() + i * num_samples_,
               n_internal_neighbors_, stream);
  }

  raft::resource::sync_stream(res_, stream);
};

template <typename Data_t, typename Idx_t>
NNDUpdate<Data_t, Idx_t>::NNDUpdate(
    raft::resources const &res, std::vector<size_t> batch_indices,
    Data_t *input_data, Idx_t *neighbor_indices, Data_t *neighbor_distances,
    Idx_t data_block_size, Idx_t neighbor_indices_block_size,
    Idx_t neighbor_dists_block_size, BloomFilter<Idx_t> &bloom_filter,
    size_t n_features, size_t n_neighbors, size_t n_internal_neighbors,
    size_t n_inserted_instances, size_t num_samples,
    float termination_threshold, size_t max_iterations, size_t seed)
    : batch_indices_(batch_indices), d_input_data_(input_data),
      d_neighbor_indices_(neighbor_indices),
      d_neighbor_distances_(neighbor_distances),
      data_block_size_(data_block_size),
      neighbor_indices_block_size_(neighbor_indices_block_size),
      neighbor_dists_block_size_(neighbor_dists_block_size),
      bloom_filter_(bloom_filter), n_row_(batch_indices.size()),
      n_features_(n_features), n_neighbors_(n_neighbors),
      n_internal_neighbors_(n_internal_neighbors),
      n_inserted_instances_(n_inserted_instances), num_samples_(num_samples),
      termination_threshold_(termination_threshold),
      max_iterations_(max_iterations), seed_(seed), res_(res),
      graph_(res, n_row_, n_neighbors_, n_internal_neighbors_,
             n_inserted_instances_, num_samples_, seed_),
      update_counter_{0},

      indices_buffer_{
          raft::make_device_matrix<InternalID_t, size_t, raft::row_major>(
              res, n_row_, num_samples_)},
      dists_buffer_{raft::make_device_matrix<Data_t, size_t, raft::row_major>(
          res, n_row_, num_samples_)},
      indices_host_buffer_{
          raft::make_pinned_matrix<InternalID_t, size_t, raft::row_major>(
              res, n_row_, num_samples_)},
      dists_host_buffer_{
          raft::make_pinned_matrix<Data_t, size_t, raft::row_major>(
              res, n_row_, num_samples_)},
      d_locks_{raft::make_device_vector<int>(res, n_row_)},
      h_rev_indices_new_{
          raft::make_pinned_matrix<Idx_t, size_t, raft::row_major>(
              res, n_row_, num_samples_)},
      h_indices_old_{raft::make_pinned_matrix<Idx_t, size_t, raft::row_major>(
          res, n_row_, num_samples_)},
      h_rev_indices_old_{
          raft::make_pinned_matrix<Idx_t, size_t, raft::row_major>(
              res, n_row_, num_samples_)},
      d_list_sizes_new_{raft::make_device_vector<int2, size_t>(res, n_row_)},
      d_list_sizes_old_{raft::make_device_vector<int2, size_t>(res, n_row_)} {}

template <typename Data_t, typename Idx_t>
void NNDUpdate<Data_t, Idx_t>::initialize() {
  cudaStream_t stream = raft::resource::get_cuda_stream(res_);

  // Idx_t -> InternalID_t 변환을 위한 임시 호스트 버퍼
  std::vector<Idx_t> h_tmp_indices(neighbor_indices_block_size_);

#pragma omp parallel for
  for (size_t i = 0; i < n_row_; ++i) {
    size_t global_row_idx_indices =
        batch_indices_[i] * neighbor_indices_block_size_;
    size_t global_row_idx_dist = batch_indices_[i] * neighbor_dists_block_size_;

    // d_neighbor_indices_ (Idx_t, device) -> 호스트 임시 버퍼로 복사
    raft::copy(h_tmp_indices.data(),
               d_neighbor_indices_ + global_row_idx_indices,
               n_internal_neighbors_, stream);
    raft::resource::sync_stream(res_, stream);

    // Idx_t -> InternalID_t 변환 후 호스트 버퍼에 저장
    for (size_t j = 0; j < static_cast<size_t>(n_internal_neighbors_); ++j) {
      indices_host_buffer_.data_handle()[i * num_samples_ + j] =
          InternalID_t(static_cast<int>(h_tmp_indices[j]));
    }

    // 호스트 -> 디바이스 복사 (InternalID_t -> InternalID_t, 타입 일치)
    raft::copy(indices_buffer_.data_handle() + i * num_samples_,
               indices_host_buffer_.data_handle() + i * num_samples_,
               n_internal_neighbors_, stream);

    // 거리 버퍼는 타입 일치 (Data_t -> Data_t)
    raft::copy(dists_buffer_.data_handle() + i * num_samples_,
               d_neighbor_distances_ + global_row_idx_dist,
               n_internal_neighbors_, stream);
    raft::copy(dists_host_buffer_.data_handle() + i * num_samples_,
               d_neighbor_distances_ + global_row_idx_dist,
               n_internal_neighbors_, stream);
  };
};

template <typename Data_t, typename Idx_t>
void NNDUpdate<Data_t, Idx_t>::build() {
  cudaStream_t stream = raft::resource::get_cuda_stream(res_);
  update_counter_ = 0;
  graph_.h_indices = indices_host_buffer_;

  graph_.sample_graph(true);

  auto update_and_sample = [&](bool update_graph) {
    if (update_graph) {
      update_counter_ = 0;
      graph_.update_graph(indices_host_buffer_.data_handle(),
                          dists_host_buffer_.data_handle(), num_samples_,
                          update_counter_);
    }
    graph_.sample_graph(false);
    if (update_counter_ <
        static_cast<int64_t>(termination_threshold_ * n_row_ * num_samples_)) {
      update_counter_ = -1; // signal for termination
    }
    graph_.sample_graph(false);
  };

  for (size_t it = 0; it < max_iterations_; ++it) {
    raft::copy(d_list_sizes_new_.data_handle(),
               graph_.h_list_sizes_new.data_handle(), n_row_, stream);

    raft::copy(h_indices_old_.data_handle(), graph_.h_indices_old.data_handle(),
               n_row_ * num_samples_, stream);
    raft::copy(d_list_sizes_old_.data_handle(),
               graph_.h_list_sizes_old.data_handle(), n_row_, stream);
    raft::resource::sync_stream(res_, stream);

    std::thread update_and_sample_thread(update_and_sample, it);

    // add_reverse_edge()
    // add_reverse_edge()
    // local_join()

    update_and_sample_thread.join();

    if (update_counter_ == -1)
      break;

    raft::copy(indices_host_buffer_.data_handle(),
               graph_.h_indices.data_handle(), n_row_ * n_internal_neighbors_,
               stream);
    raft::copy(dists_host_buffer_.data_handle(), graph_.h_dists.data_handle(),
               n_row_ * n_internal_neighbors_, stream);
    raft::resource::sync_stream(res_, stream);

    graph_.sample_graph_new(indices_host_buffer_.data_handle(), num_samples_);
  }

  graph_.update_graph(indices_host_buffer_.data_handle(),
                      dists_host_buffer_.data_handle(), num_samples_,
                      update_counter_);

  raft::resource::sync_stream(res_, stream);

  update_neighbor_info();
}

} // namespace gpumap::nndescent
