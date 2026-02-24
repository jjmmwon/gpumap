#pragma once

#include "buffers.cuh"
#include "insertion.cuh"

#if defined(__clang__) && defined(__CUDA__)
#include <__clang_cuda_runtime_wrapper.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <limits>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <vector>

namespace gpumap::nndescent {

template <typename Idx_t>
__device__ __forceinline__ bool list_contains_id(const Idx_t *arr, int k,
                                                 Idx_t v) {
  for (int i = 0; i < k; ++i) {
    if (arr[i] == v) {
      return true;
    }
  }
  return false;
}

template <typename Data_t>
__global__ void
preprocess_l2_norms_kernel(Data_t *input_data, uint32_t batch_size,
                           uint32_t n_features, uint32_t data_block_size) {

  const size_t row = static_cast<size_t>(blockIdx.x);

  if (row >= batch_size)
    return;

  const size_t row_offset = row * data_block_size;
  float sum = 0.0f;

  for (size_t j = static_cast<size_t>(threadIdx.x); j < n_features;
       j += blockDim.x) {

    float val = static_cast<float>(input_data[row_offset + j]);
    sum += val * val;
  }

  extern __shared__ float sdata[];
  sdata[threadIdx.x] = sum;
  __syncthreads();

  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    input_data[row_offset + n_features] = static_cast<Data_t>(sdata[0]);
  }
}

template <typename Data_t, typename Idx_t, typename Index_t = int>
__global__ __forceinline__ void initialize_random_neighbors_kernel(
    InternalID_t *graph_buffer_, Data_t *distance_buffer_, int width,
    uint32_t batch_size, uint32_t batch_start_offset,
    uint32_t n_inserted_instances, size_t seed) {

  size_t row_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_local >= batch_size)
    return;

  Index_t gid = static_cast<Index_t>(batch_start_offset + row_local);

  curandStatePhilox4_32_10_t state;
  curand_init(seed, gid, 0, &state);

  InternalID_t *my_neighbors = graph_buffer_ + row_local * width;
  Data_t *my_distances = distance_buffer_ + row_local * width;

  if (n_inserted_instances <= 1) {
    for (int i = 0; i < width; ++i) {
      my_neighbors[i].id_with_flag() = std::numeric_limits<int>::max();
      my_distances[i] = std::numeric_limits<Data_t>::max();
    }
    return;
  }

  const int max_unique_fill =
      min(width, static_cast<int>(n_inserted_instances) - 1);

  int filled = 0;
  while (filled < max_unique_fill) {
    Index_t candidate = curand(&state) % n_inserted_instances;
    if (candidate == gid)
      continue;
    bool existed = false;
    for (int i = 0; i < filled; ++i) {
      if (my_neighbors[i].id() == static_cast<int>(candidate)) {
        existed = true;
        break;
      }
    }
    if (existed) {
      continue;
    }

    my_neighbors[filled].id_with_flag() = static_cast<int>(candidate);
    my_distances[filled++] = std::numeric_limits<Data_t>::max();
  }

  for (int i = filled; i < width; ++i) {
    my_neighbors[i].id_with_flag() = std::numeric_limits<int>::max();
    my_distances[i] = std::numeric_limits<Data_t>::max();
  }
}

template <typename Data_t, typename Idx_t, typename Index_t = int>
__global__ __forceinline__ void sample_forward_candidates_kernel(
    InternalID_t *graph_buffer, int *candidate_buffer, int graph_width,
    int candidate_width, uint32_t batch_size, uint32_t batch_start_offset,
    uint32_t n_neighbors, uint32_t n_inserted_instances) {
  const size_t row_local =
      static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);

  if (row_local >= batch_size) {
    return;
  }

  const Index_t gid = static_cast<Index_t>(batch_start_offset + row_local);

  InternalID_t *my_neighbors =
      graph_buffer + static_cast<size_t>(gid) * graph_width;

  int *my_candidates =
      candidate_buffer + static_cast<size_t>(gid) * candidate_width;

  for (uint32_t i = 0; i < candidate_width; ++i) {
    my_candidates[i] = -1;
  }

  Index_t my_neighbors_l[MAX_NEIGHBORS];
  for (uint32_t i = 0; i < n_neighbors; ++i) {
    my_neighbors_l[i] = static_cast<Index_t>(my_neighbors[i].id());
  }

  int candidate_count = 0;
  for (uint32_t j = 0; j < n_neighbors && candidate_count < candidate_width;
       ++j) {
    InternalID_t neighbor_id_raw = my_neighbors[j];
    if (!neighbor_id_raw.is_new()) {
      continue;
    }

    const Index_t neighbor_id = static_cast<Index_t>(neighbor_id_raw.id());
    // my_neighbors[j].mark_old();

    InternalID_t *neigh2 =
        graph_buffer + static_cast<size_t>(neighbor_id) * graph_width;

    for (uint32_t k = 0; k < n_neighbors && candidate_count < candidate_width;
         ++k) {
      InternalID_t candidate_id_raw = neigh2[k];
      if (static_cast<uint32_t>(candidate_id_raw.id()) >=
          n_inserted_instances) {
        continue;
      }

      const Index_t candidate_id = static_cast<Index_t>(candidate_id_raw.id());

      if (candidate_id == gid) {
        continue;
      }

      if (list_contains_id<Index_t>(
              my_neighbors_l, static_cast<int>(n_neighbors), candidate_id)) {
        continue;
      }
      if (list_contains_id<int>(my_candidates, candidate_count,
                                static_cast<int>(candidate_id))) {
        continue;
      }
      my_candidates[candidate_count++] = static_cast<int>(candidate_id);
    }
  }
}

template <typename Data_t, typename Idx_t, typename Index_t = int>
__global__ __forceinline__ void
forward_nndescent_kernel(const Data_t *input_data, InternalID_t *graph_buffer,
                         Data_t *distance_buffer, const int *candidate_buffer,
                         uint32_t batch_size, uint32_t batch_start_offset,
                         uint32_t n_neighbors, uint32_t n_features,
                         uint32_t data_block_size, uint32_t n_candidates,
                         uint32_t n_inserted_instances) {
  const size_t row_local =
      static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_local >= batch_size) {
    return;
  }

  const Index_t gid = static_cast<Index_t>(batch_start_offset) + row_local;

  InternalID_t *my_neighbors =
      graph_buffer + static_cast<size_t>(gid) * MAX_NEIGHBORS;
  Data_t *my_distances =
      distance_buffer + static_cast<size_t>(gid) * MAX_NEIGHBORS;

  const int *my_candidates =
      candidate_buffer + static_cast<size_t>(gid) * n_candidates;

  Index_t neighbors_local[MAX_NEIGHBORS];
  Data_t distances_local[MAX_NEIGHBORS];

  for (uint32_t i = 0; i < n_neighbors; ++i) {
    neighbors_local[i] = static_cast<Index_t>(my_neighbors[i].id());
    distances_local[i] = my_distances[i];
  }

  // 1) Compute all candidate distances first.
  Index_t candidate_ids[MAX_CANDIDATES];
  Data_t candidate_dists[MAX_CANDIDATES];
  int valid_candidate_count = 0;

  const size_t offset1 = static_cast<size_t>(gid) * data_block_size;
  for (uint32_t i = 0; i < n_candidates; ++i) {
    const int candidate = my_candidates[i];
    if (candidate < 0 ||
        static_cast<uint32_t>(candidate) >= n_inserted_instances) {
      continue;
    }
    const Index_t candidate_id = static_cast<Index_t>(candidate);
    if (candidate_id == gid) {
      continue;
    }
    if (list_contains_id<Index_t>(
            neighbors_local, static_cast<int>(n_neighbors), candidate_id)) {
      continue;
    }
    if (list_contains_id<Index_t>(candidate_ids, valid_candidate_count,
                                  candidate_id)) {
      continue;
    }

    const size_t offset2 = static_cast<size_t>(candidate_id) * data_block_size;
    float dot = 0.0f;
    for (uint32_t d = 0; d < n_features; ++d) {
      dot += static_cast<float>(input_data[offset1 + d]) *
             static_cast<float>(input_data[offset2 + d]);
    }
    float dist_sq = static_cast<float>(input_data[offset1 + n_features]) +
                    static_cast<float>(input_data[offset2 + n_features]) -
                    2.0f * dot;
    dist_sq = dist_sq < 0.0f ? 0.0f : dist_sq;

    candidate_ids[valid_candidate_count] = candidate_id;
    candidate_dists[valid_candidate_count] = static_cast<Data_t>(dist_sq);
    ++valid_candidate_count;
  }

  // 2) Sort candidates by (distance asc, id asc).
  for (int i = 1; i < valid_candidate_count; ++i) {
    const Index_t key_id = candidate_ids[i];
    const Data_t key_dist = candidate_dists[i];
    int j = i - 1;
    while (j >= 0) {
      const bool move =
          (candidate_dists[j] > key_dist) ||
          (candidate_dists[j] == key_dist && candidate_ids[j] > key_id);
      if (!move) {
        break;
      }
      candidate_ids[j + 1] = candidate_ids[j];
      candidate_dists[j + 1] = candidate_dists[j];
      --j;
    }
    candidate_ids[j + 1] = key_id;
    candidate_dists[j + 1] = key_dist;
  }

  // 3) Insert sorted candidates from the smallest distance first.
  bool updated = false;
  for (int i = 0; i < valid_candidate_count; ++i) {
    const Index_t candidate_id = candidate_ids[i];
    const Data_t dist = candidate_dists[i];

    if (list_contains_id<Index_t>(
            neighbors_local, static_cast<int>(n_neighbors), candidate_id)) {
      continue;
    }

    const int tail_pos = static_cast<int>(n_neighbors) - 1;
    const Data_t tail_dist = distances_local[tail_pos];
    const Index_t tail_id = neighbors_local[tail_pos];
    if (dist > tail_dist || (dist == tail_dist && candidate_id >= tail_id)) {
      continue;
    }

    int insert_pos = tail_pos;
    while (insert_pos > 0) {
      const Data_t prev_dist = distances_local[insert_pos - 1];
      const Index_t prev_id = neighbors_local[insert_pos - 1];
      const bool should_shift =
          (dist < prev_dist) || (dist == prev_dist && candidate_id < prev_id);
      if (!should_shift) {
        break;
      }
      neighbors_local[insert_pos] = neighbors_local[insert_pos - 1];
      distances_local[insert_pos] = distances_local[insert_pos - 1];
      --insert_pos;
    }
    neighbors_local[insert_pos] = candidate_id;
    distances_local[insert_pos] = dist;
    updated = true;
  }

  if (updated) {
    for (uint32_t i = 0; i < n_neighbors; ++i) {
      if (neighbors_local[i] == std::numeric_limits<Index_t>::max()) {
        my_neighbors[i].id_with_flag() = std::numeric_limits<int>::max();
      } else {
        my_neighbors[i].id_with_flag() = static_cast<int>(neighbors_local[i]);
      }
      my_distances[i] = distances_local[i];
    }
  }
}

template <typename Data_t>
__global__ void initialize_embeddings_kernel(
    const InternalID_t *graph_buffer, const Data_t *distance_buffer,
    Data_t *embedding, uint32_t batch_size, uint32_t batch_start_offset,
    uint32_t n_neighbors, uint32_t n_inserted_instances,
    uint32_t embedding_block_size) {
  const size_t row_local = static_cast<size_t>(blockIdx.x);
  if (row_local >= batch_size) {
    return;
  }

  const size_t row = static_cast<size_t>(batch_start_offset) + row_local;
  const InternalID_t *my_neighbors =
      graph_buffer + row * static_cast<size_t>(MAX_NEIGHBORS);
  const Data_t *my_dists =
      distance_buffer + row * static_cast<size_t>(MAX_NEIGHBORS);

  __shared__ int s_neighbor_ids[MAX_NEIGHBORS];
  __shared__ float s_weights[MAX_NEIGHBORS];
  __shared__ float s_weight_sum;

  if (threadIdx.x == 0) {
    Data_t min_dist = std::numeric_limits<Data_t>::max();
    for (uint32_t j = 0; j < n_neighbors; ++j) {
      const int nid = my_neighbors[j].id();
      if (nid < 0 || static_cast<uint32_t>(nid) >= n_inserted_instances) {
        s_neighbor_ids[j] = -1;
        s_weights[j] = 0.0f;
        continue;
      }
      if (nid >= static_cast<int>(batch_start_offset)) {
        s_neighbor_ids[j] = -1;
        s_weights[j] = 0.0f;
        continue;
      }
      if (my_dists[j] < min_dist) {
        min_dist = my_dists[j];
      }
      s_neighbor_ids[j] = nid;
    }

    if (min_dist == std::numeric_limits<Data_t>::max()) {
      s_weight_sum = 0.0f;
    } else {
      float weight_sum = 0.0f;
      for (uint32_t j = 0; j < n_neighbors; ++j) {
        const int nid = s_neighbor_ids[j];
        if (nid < 0) {
          s_weights[j] = 0.0f;
          continue;
        }
        const float w = expf(-static_cast<float>(my_dists[j] - min_dist));
        s_weights[j] = w;
        weight_sum += w;
      }
      s_weight_sum = weight_sum;
    }
  }
  __syncthreads();

  const float weight_sum = s_weight_sum;
  if (weight_sum <= 0.0f) {
    return;
  }

  for (uint32_t dim = threadIdx.x; dim < embedding_block_size;
       dim += blockDim.x) {
    float accum = 0.0f;
    for (uint32_t j = 0; j < n_neighbors; ++j) {
      const int nid = s_neighbor_ids[j];
      if (nid < 0) {
        continue;
      }
      accum +=
          s_weights[j] *
          static_cast<float>(
              embedding[static_cast<size_t>(nid) * embedding_block_size + dim]);
    }
    embedding[row * embedding_block_size + dim] =
        static_cast<Data_t>(accum / weight_sum);
  }
}

template <typename Data_t, typename Idx_t, typename Index_t>
void InsertionAlgo<Data_t, Idx_t, Index_t>::preprocess_l2_norms(
    cudaStream_t stream) {
  if (batch_indices_.empty()) {
    return;
  }
  const int threads_per_block = 256;
  const int n_blocks = static_cast<int>(batch_indices_.size());
  const int shared_mem_size = threads_per_block * sizeof(float);
  Data_t *input_data_offset =
      input_data + batch_indices_.front() * data_block_size_;

  preprocess_l2_norms_kernel<<<n_blocks, threads_per_block, shared_mem_size,
                               stream>>>(
      input_data_offset, batch_indices_.size(), n_features_, data_block_size_);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename Data_t, typename Idx_t, typename Index_t>
void InsertionAlgo<Data_t, Idx_t, Index_t>::initialize_random_neighbors(
    InternalID_t *graph_buffer_, Data_t *distance_buffer_, cudaStream_t stream,
    size_t seed) {

  size_t n_rows = batch_indices_.size();
  if (n_rows == 0) {
    return;
  }

  const int threads_per_block = 256;
  const int n_blocks = (n_rows + threads_per_block - 1) / threads_per_block;

  InternalID_t *graph_buffer_offset =
      graph_buffer_ +
      static_cast<size_t>(batch_indices_.front()) * MAX_NEIGHBORS;

  Data_t *distance_buffer_offset =
      distance_buffer_ +
      static_cast<size_t>(batch_indices_.front()) * MAX_NEIGHBORS;

  uint32_t batch_start_offset = static_cast<uint32_t>(batch_indices_.front());

  initialize_random_neighbors_kernel<Data_t, Idx_t, Index_t>
      <<<n_blocks, threads_per_block, 0, stream>>>(
          graph_buffer_offset, distance_buffer_offset, MAX_NEIGHBORS, n_rows,
          batch_start_offset, n_inserted_instances_, seed);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename Data_t, typename Idx_t, typename Index_t>
void InsertionAlgo<Data_t, Idx_t, Index_t>::sample_forward_candidates(
    InternalID_t *graph_buffer_, int *candidate_buffer_, cudaStream_t stream) {
  const uint32_t n_rows = static_cast<uint32_t>(batch_indices_.size());
  if (n_rows == 0) {
    return;
  }
  const uint32_t batch_start_offset =
      static_cast<uint32_t>(batch_indices_.front());
  const int threads_per_block = 256;
  const int blocks = (n_rows + threads_per_block - 1) / threads_per_block;

  sample_forward_candidates_kernel<Data_t, Idx_t, Index_t>
      <<<blocks, threads_per_block, 0, stream>>>(
          graph_buffer_, candidate_buffer_, MAX_NEIGHBORS, MAX_CANDIDATES,
          n_rows, batch_start_offset, MAX_NEIGHBORS, n_inserted_instances_);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename Data_t, typename Idx_t, typename Index_t>
void InsertionAlgo<Data_t, Idx_t, Index_t>::forward_nndescent(
    InternalID_t *graph_buffer_, Data_t *dists_buffer_,
    const int *candidate_buffer_, cudaStream_t stream) {
  const uint32_t n_rows = static_cast<uint32_t>(batch_indices_.size());
  if (n_rows == 0) {
    return;
  }
  const uint32_t row_offset = static_cast<uint32_t>(batch_indices_.front());
  constexpr uint32_t n_candidates = MAX_CANDIDATES;
  const int threads_per_block = 256;
  const int blocks = (n_rows + threads_per_block - 1) / threads_per_block;

  Data_t *input_data_offset = input_data;

  forward_nndescent_kernel<Data_t, Idx_t, Index_t>
      <<<blocks, threads_per_block, 0, stream>>>(
          input_data_offset, graph_buffer_, dists_buffer_, candidate_buffer_,
          n_rows, row_offset, MAX_NEIGHBORS, n_features_, data_block_size_,
          n_candidates, n_inserted_instances_);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename Data_t, typename Idx_t, typename Index_t>
void InsertionAlgo<Data_t, Idx_t, Index_t>::write_new_neighbors_info(
    cudaStream_t stream) {
  if (batch_indices_.empty()) {
    return;
  }
  size_t n_rows = batch_indices_.size();
  size_t batch_start_offset = batch_indices_.front();
  const auto *graph_host = buffers_.graph_host_buffer_.data_handle();
  const auto *dists_host = buffers_.dists_host_buffer_.data_handle();
  std::vector<Idx_t> h_neighbor_row(n_neighbors_);
  std::vector<Data_t> h_dist_row(n_neighbors_);
  const Idx_t dirty_flag_one = static_cast<Idx_t>(1);
  const Data_t sigma_init = static_cast<Data_t>(1);

  for (size_t i = 0; i < n_rows; ++i) {
    const size_t row = batch_start_offset + i;
    const auto *src_graph = graph_host + row * MAX_NEIGHBORS;
    const auto *src_dists = dists_host + row * MAX_NEIGHBORS;

    for (size_t j = 0; j < n_neighbors_; ++j) {
      const int nid = src_graph[j].id();
      const bool valid_id =
          (nid >= 0 && static_cast<size_t>(nid) < n_inserted_instances_) &&
          (static_cast<size_t>(nid) != row);
      h_neighbor_row[j] = valid_id ? static_cast<Idx_t>(nid)
                                   : std::numeric_limits<Idx_t>::max();
      h_dist_row[j] =
          valid_id ? src_dists[j] : std::numeric_limits<Data_t>::max();
    }

    raft::copy(neighbor_indices + row * neighbor_indices_block_size_,
               h_neighbor_row.data(), n_neighbors_, stream);

    raft::copy(neighbor_distances + row * neighbor_dists_block_size_,
               h_dist_row.data(), n_neighbors_, stream);

    raft::copy(neighbor_indices + row * neighbor_indices_block_size_ +
                   (neighbor_indices_block_size_ - 1),
               &dirty_flag_one, 1, stream);
    raft::copy(neighbor_distances + row * neighbor_dists_block_size_ +
                   (neighbor_dists_block_size_ - 1),
               &sigma_init, 1, stream);

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }
}

template <typename Data_t, typename Idx_t, typename Index_t>
void InsertionAlgo<Data_t, Idx_t, Index_t>::run(
    cudaStream_t stream, std::vector<Idx_t> &affected_indices,
    std::vector<std::tuple<Idx_t, Idx_t, Idx_t>> &new_edges) {
  if (batch_indices_.empty()) {
    return;
  }
  const size_t n_rows = batch_indices_.size();
  const size_t batch_start_offset = static_cast<size_t>(batch_indices_.front());

  // Buffers on device for graph structure and distances during NNDESENT
  // insertion iterations.
  InternalID_t *graph_buffer_ = buffers_.graph_buffer_.data_handle();
  Data_t *dists_buffer_ = buffers_.dists_buffer_.data_handle();
  int *candidate_buffer_ = buffers_.candidate_buffer_.data_handle();

  preprocess_l2_norms(stream);
  initialize_random_neighbors(graph_buffer_, dists_buffer_, stream, seed_);

  raft::copy(buffers_.graph_host_buffer_.data_handle() +
                 batch_start_offset * MAX_NEIGHBORS,
             graph_buffer_ + batch_start_offset * MAX_NEIGHBORS,
             n_rows * MAX_NEIGHBORS, stream);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < 1; ++i) {
    for (size_t j = 0; j < MAX_NEIGHBORS; ++j) {
      fprintf(stderr, "%d ",
              buffers_.graph_host_buffer_.data_handle()[i * MAX_NEIGHBORS + j]
                  .id());
    }
    fprintf(stderr, "\n");
    fflush(stderr);
  }

  for (size_t iter = 0; iter < max_iter_; ++iter) {
    sample_forward_candidates(graph_buffer_, candidate_buffer_, stream);
    forward_nndescent(graph_buffer_, dists_buffer_, candidate_buffer_, stream);

    // === 수렴 추적: 매 iteration마다 row 0의 최소 거리 + new 이웃 수 출력 ===
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    float h_dists[MAX_NEIGHBORS];
    cudaMemcpy(h_dists,
               dists_buffer_ + (batch_start_offset + 2) * MAX_NEIGHBORS,
               MAX_NEIGHBORS * sizeof(float), cudaMemcpyDeviceToHost);
    int h_graph_raw[MAX_NEIGHBORS];
    cudaMemcpy(h_graph_raw, graph_buffer_ + batch_start_offset * MAX_NEIGHBORS,
               MAX_NEIGHBORS * sizeof(int), cudaMemcpyDeviceToHost);

    float min_dist = h_dists[0];
    int n_new = 0;
    int n_candidates_valid = 0;
    int h_cands[MAX_CANDIDATES];
    cudaMemcpy(h_cands, candidate_buffer_ + batch_start_offset * MAX_CANDIDATES,
               MAX_CANDIDATES * sizeof(int), cudaMemcpyDeviceToHost);
    for (int j = 0; j < MAX_NEIGHBORS; j++) {
      if (h_dists[j] < min_dist)
        min_dist = h_dists[j];
      if (h_graph_raw[j] >= 0)
        n_new++;
    }
    for (int j = 0; j < MAX_CANDIDATES; j++)
      if (h_cands[j] >= 0)
        n_candidates_valid++;

    fprintf(stderr,
            "[iter %3zu] min_dist=%.4f  new_internal=%d/%d  candidates=%d/%d\n",
            iter, min_dist, n_new, MAX_NEIGHBORS, n_candidates_valid,
            MAX_CANDIDATES);
    fflush(stderr);
  }

  // Buffers on host for graph structure and distances to be used for writing
  InternalID_t *graph_host_buffer_ = buffers_.graph_host_buffer_.data_handle();
  Data_t *dists_host_buffer_ = buffers_.dists_host_buffer_.data_handle();

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  raft::copy(graph_host_buffer_ + batch_start_offset * MAX_NEIGHBORS,
             graph_buffer_ + batch_start_offset * MAX_NEIGHBORS,
             n_rows * MAX_NEIGHBORS, stream);
  raft::copy(dists_host_buffer_ + batch_start_offset * MAX_NEIGHBORS,
             dists_buffer_ + batch_start_offset * MAX_NEIGHBORS,
             n_rows * MAX_NEIGHBORS, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < n_rows; ++i) {
    const size_t row = batch_start_offset + i;
    auto *dst_graph = buffers_.h_graph.data_handle() + row * n_neighbors_;
    auto *dst_dists = buffers_.h_dists.data_handle() + row * n_neighbors_;
    const auto *src_graph = graph_host_buffer_ + row * MAX_NEIGHBORS;
    const auto *src_dists = dists_host_buffer_ + row * MAX_NEIGHBORS;
    for (size_t j = 0; j < n_neighbors_; ++j) {
      dst_graph[j] = src_graph[j];
      dst_dists[j] = src_dists[j];
    }
  }

  for (size_t i = 0; i < 1; ++i) {
    for (size_t j = 0; j < MAX_NEIGHBORS; ++j) {
      fprintf(stderr, "%d ", graph_host_buffer_[i * MAX_NEIGHBORS + j].id());
    }
    fprintf(stderr, "\n");
    fflush(stderr);
  }

  write_new_neighbors_info(stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < n_rows; ++i) {
    const size_t row = static_cast<size_t>(batch_indices_[i]);
    const InternalID_t *my_neighbors = graph_host_buffer_ + row * MAX_NEIGHBORS;

    for (uint32_t j = 0; j < n_neighbors_; ++j) {
      const int nid = my_neighbors[j].id();
      if (nid < 0 || static_cast<uint32_t>(nid) >= n_inserted_instances_) {
        continue;
      }
      affected_indices.push_back(static_cast<Idx_t>(nid));
      new_edges.emplace_back(static_cast<Idx_t>(batch_indices_[i]),
                             static_cast<Idx_t>(nid), 0);
    }
  }

  std::sort(affected_indices.begin(), affected_indices.end());
  affected_indices.erase(
      std::unique(affected_indices.begin(), affected_indices.end()),
      affected_indices.end());

  initialize_embeddings(stream);
}

template <typename Data_t, typename Idx_t, typename Index_t>
void InsertionAlgo<Data_t, Idx_t, Index_t>::initialize_embeddings(
    cudaStream_t stream) {
  if (embedding == nullptr)
    return;

  const uint32_t batch_size = static_cast<uint32_t>(batch_indices_.size());
  if (batch_size == 0) {
    return;
  }
  const uint32_t batch_start_offset =
      static_cast<uint32_t>(batch_indices_.front());
  const int threads_per_block =
      static_cast<int>(std::min<uint32_t>(embedding_block_size_, 256));

  initialize_embeddings_kernel<Data_t>
      <<<batch_size, threads_per_block, 0, stream>>>(
          buffers_.graph_buffer_.data_handle(),
          buffers_.dists_buffer_.data_handle(), embedding, batch_size,
          batch_start_offset, n_neighbors_, n_inserted_instances_,
          embedding_block_size_);
  RAFT_CUDA_TRY(cudaGetLastError());
}

} // namespace gpumap::nndescent
