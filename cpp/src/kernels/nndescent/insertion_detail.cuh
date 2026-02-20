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

template <typename Idx_t>
inline bool host_list_contains_id(const Idx_t *arr, int k, Idx_t v) {
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
    my_candidates[i] = std::numeric_limits<int>::max();
  }

  Index_t my_neighbors_l[MAX_NEIGHBORS];
  for (uint32_t i = 0; i < n_neighbors; ++i) {
    my_neighbors_l[i] = static_cast<Index_t>(my_neighbors[i].id());
  }

  int candidate_count = 0;
  for (uint32_t j = 0; j < n_neighbors && candidate_count < candidate_width;
       ++j) {
    InternalID_t neighbor_id_raw = my_neighbors[j];
    if (!neighbor_id_raw.is_new() ||
        static_cast<uint32_t>(neighbor_id_raw.id()) >= n_inserted_instances) {
      continue;
    }

    const Index_t neighbor_id = static_cast<Index_t>(neighbor_id_raw.id());

    const InternalID_t *neigh2 =
        graph_buffer + static_cast<size_t>(neighbor_id) * graph_width;

    for (uint32_t k = 0; k < n_neighbors && candidate_count < candidate_width;
         ++k) {
      InternalID_t candidate_id_raw = neigh2[k];
      if (!candidate_id_raw.is_new() ||
          static_cast<uint32_t>(candidate_id_raw.id()) >=
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
      candidate_id_raw.mark_old();
    }
  }
}

template <typename Data_t, typename Idx_t, typename Index_t = int>
__global__ __forceinline__ void
forward_nndescent_kernel(const Data_t *input_data, InternalID_t *graph_buffer,
                         Data_t *distance_buffer, const int *candidate_buffer,
                         uint32_t batch_size, uint32_t batch_start_offset,
                         uint32_t n_neighbors, uint32_t n_features,
                         uint32_t data_block_size, uint32_t n_candidates) {
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
  Index_t candidate_ids[MAX_NEIGHBORS];
  Data_t candidate_dists[MAX_NEIGHBORS];
  int valid_candidate_count = 0;

  const size_t offset1 = static_cast<size_t>(gid) * data_block_size;
  for (uint32_t i = 0; i < n_candidates; ++i) {
    const int candidate = my_candidates[i];
    if (candidate < 0) {
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
    cudaStream_t stream, size_t seed) {

  size_t n_rows = batch_indices_.size();
  if (n_rows == 0) {
    return;
  }

  const int threads_per_block = 256;
  const int n_blocks = (n_rows + threads_per_block - 1) / threads_per_block;

  InternalID_t *graph_buffer_offset = buffers_.graph_buffer_.data_handle() +
                                      batch_indices_.front() * MAX_NEIGHBORS;
  Data_t *distance_buffer_offset = buffers_.dists_buffer_.data_handle() +
                                   batch_indices_.front() * MAX_NEIGHBORS;
  uint32_t batch_start_offset = static_cast<uint32_t>(batch_indices_.front());

  initialize_random_neighbors_kernel<Data_t, Idx_t, Index_t>
      <<<n_blocks, threads_per_block, 0, stream>>>(
          graph_buffer_offset, distance_buffer_offset, MAX_NEIGHBORS, n_rows,
          batch_start_offset, n_inserted_instances_, seed);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename Data_t, typename Idx_t, typename Index_t>
void InsertionAlgo<Data_t, Idx_t, Index_t>::sample_forward_candidates(
    cudaStream_t stream) {
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
          buffers_.graph_buffer_.data_handle(),
          buffers_.candidate_buffer_.data_handle(), MAX_NEIGHBORS,
          MAX_NEIGHBORS, n_rows, batch_start_offset, n_neighbors_,
          n_inserted_instances_);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename Data_t, typename Idx_t, typename Index_t>
void InsertionAlgo<Data_t, Idx_t, Index_t>::forward_nndescent(
    cudaStream_t stream) {
  const uint32_t n_rows = static_cast<uint32_t>(batch_indices_.size());
  if (n_rows == 0) {
    return;
  }
  const uint32_t row_offset = static_cast<uint32_t>(batch_indices_.front());
  constexpr uint32_t n_candidates = MAX_NEIGHBORS;
  const int threads_per_block = 256;
  const int blocks = (n_rows + threads_per_block - 1) / threads_per_block;

  Data_t *input_data_offset = input_data;

  forward_nndescent_kernel<Data_t, Idx_t, Index_t>
      <<<blocks, threads_per_block, 0, stream>>>(
          input_data_offset, buffers_.graph_buffer_.data_handle(),
          buffers_.dists_buffer_.data_handle(),
          buffers_.candidate_buffer_.data_handle(), n_rows, row_offset,
          n_neighbors_, n_features_, data_block_size_, n_candidates);
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

  raft::copy(neighbor_indices +
                 batch_start_offset * neighbor_indices_block_size_,
             reinterpret_cast<Idx_t *>(buffers_.graph_buffer_.data_handle()) +
                 batch_start_offset * neighbor_indices_block_size_,
             n_rows * neighbor_indices_block_size_, stream);

  raft::copy(neighbor_distances +
                 batch_start_offset * neighbor_dists_block_size_,
             buffers_.dists_buffer_.data_handle() +
                 batch_start_offset * neighbor_dists_block_size_,
             n_rows * neighbor_dists_block_size_, stream);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Data_t, typename Idx_t, typename Index_t>
void InsertionAlgo<Data_t, Idx_t, Index_t>::run(
    cudaStream_t stream, std::vector<Idx_t> &affected_indices,
    std::vector<std::tuple<Idx_t, Idx_t, Idx_t>> &new_edges) {

  preprocess_l2_norms(stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  initialize_random_neighbors(stream, seed_);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  const size_t n_rows = batch_indices_.size();
  const size_t batch_start_offset = static_cast<size_t>(batch_indices_.front());

  auto *graph_host = buffers_.graph_host_buffer_.data_handle();
  auto *graph_dev = buffers_.graph_buffer_.data_handle();

  for (size_t iter = 0; iter < max_iter_; ++iter) {
    sample_forward_candidates(stream);
    forward_nndescent(stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  };

  raft::copy(graph_host + batch_start_offset * MAX_NEIGHBORS,
             graph_dev + batch_start_offset * MAX_NEIGHBORS,
             n_rows * MAX_NEIGHBORS, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  write_new_neighbors_info(stream);

  for (size_t i = 0; i < n_rows; ++i) {
    const size_t row = static_cast<size_t>(batch_indices_[i]);
    const InternalID_t *my_neighbors = graph_host + row * MAX_NEIGHBORS;

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

  initialize_embeddings();
}

template <typename Data_t, typename Idx_t, typename Index_t>
void InsertionAlgo<Data_t, Idx_t, Index_t>::initialize_embeddings() {
  if (embedding == nullptr)
    return;

  const InternalID_t *graph_host = buffers_.graph_host_buffer_.data_handle();

  // Host-side temporaries reused across nodes to avoid per-node allocation.
  std::vector<Data_t> neigh_embed_h(embedding_block_size_);
  std::vector<Data_t> result_embed_h(embedding_block_size_);
  std::vector<float> accum(embedding_block_size_);

  for (const auto raw_idx : batch_indices_) {
    const size_t row = static_cast<size_t>(raw_idx);
    const InternalID_t *my_neighbors = graph_host + row * MAX_NEIGHBORS;

    // Pass 1: min distance over old neighbors for numerical stability.
    Data_t min_dist = std::numeric_limits<Data_t>::max();
    for (uint32_t j = 0; j < n_neighbors_; ++j) {
      const int nid = my_neighbors[j].id();
      if (nid < 0 || static_cast<uint32_t>(nid) >= n_inserted_instances_)
        continue;
      if (nid >= static_cast<int>(batch_indices_.front()))
        continue;
      const Data_t d = neighbor_distances[row * neighbor_dists_block_size_ + j];
      if (d < min_dist)
        min_dist = d;
    }

    if (min_dist == std::numeric_limits<Data_t>::max())
      continue; // no valid old neighbor — skip

    // Pass 2: softmax-weighted accumulation of neighbor embeddings.
    float weight_sum = 0.0f;
    std::fill(accum.begin(), accum.end(), 0.0f);

    for (uint32_t j = 0; j < n_neighbors_; ++j) {
      const int nid = my_neighbors[j].id();
      if (nid < 0 || static_cast<uint32_t>(nid) >= n_inserted_instances_)
        continue;
      if (nid >= static_cast<int>(batch_indices_.front()))
        continue;
      const Data_t d = neighbor_distances[row * neighbor_dists_block_size_ + j];
      const float w = std::exp(-static_cast<float>(d - min_dist));
      weight_sum += w;

      // Download neighbor embedding from GPU to host.
      RAFT_CUDA_TRY(cudaMemcpy(
          neigh_embed_h.data(),
          embedding + static_cast<size_t>(nid) * embedding_block_size_,
          embedding_block_size_ * sizeof(Data_t), cudaMemcpyDeviceToHost));
      for (uint32_t dim = 0; dim < embedding_block_size_; ++dim)
        accum[dim] += w * static_cast<float>(neigh_embed_h[dim]);
    }

    for (uint32_t dim = 0; dim < embedding_block_size_; ++dim)
      result_embed_h[dim] = static_cast<Data_t>(accum[dim] / weight_sum);

    // Upload computed embedding back to GPU.
    RAFT_CUDA_TRY(cudaMemcpy(
        embedding + row * embedding_block_size_, result_embed_h.data(),
        embedding_block_size_ * sizeof(Data_t), cudaMemcpyHostToDevice));
  }
}

} // namespace gpumap::nndescent
