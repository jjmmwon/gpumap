#pragma once

#include "buffers.cuh"
#include "raft/util/cuda_rt_essentials.hpp"
#include <algorithm>
#include <cstdint>
#include <curand_kernel.h>
#include <sys/types.h>
#include <vector>

namespace gpumap {
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

// Sampling strategy: 50% from recent window [window_start, n_inserted),
// 50% from full range [0, n_inserted). If the window has fewer unique
// candidates than its quota, the shortfall is filled from the full range.
template <typename Data_t, typename Idx_t, typename Index_t = int>
__global__ __forceinline__ void initialize_random_neighbors_kernel(
    InternalID_t *graph_buffer_, Data_t *distance_buffer_, int width,
    uint32_t batch_size, uint32_t batch_start_offset,
    uint32_t n_inserted_instances, uint32_t window_start, size_t seed) {

  size_t row_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_local >= batch_size)
    return;

  Index_t gid = static_cast<Index_t>(batch_start_offset + row_local);

  curandStatePhilox4_32_10_t state;
  curand_init(seed, gid, 0, &state);

  InternalID_t *my_neighbors = graph_buffer_ + row_local * width;
  Data_t *my_distances = distance_buffer_ + row_local * width;

  const int id_sentinel = std::numeric_limits<int>::max();
  const Data_t dist_sentinel = std::numeric_limits<Data_t>::max();

  if (n_inserted_instances <= 1) {
    for (int i = 0; i < width; ++i) {
      my_neighbors[i].id_with_flag() = id_sentinel;
      my_distances[i] = dist_sentinel;
    }
    return;
  }

  const int max_unique_fill =
      min(width, static_cast<int>(n_inserted_instances) - 1);

  // window_quota: how many slots we try to fill from the recent window.
  // Capped by available unique window nodes (subtract 1 for possible self).
  const int window_size =
      static_cast<int>(n_inserted_instances) - static_cast<int>(window_start);
  const int window_quota = min(width / 2, max(0, window_size - 1));

  Index_t filled_ids[MAX_NEIGHBORS];
  int filled = 0;

  // Phase 1: sample from recent window [window_start, n_inserted_instances).
  // Limit attempts to avoid spinning when window is nearly exhausted.
  int attempts = 0;
  const int max_attempts = window_quota * 16;
  while (filled < window_quota && attempts < max_attempts) {
    ++attempts;
    const Index_t candidate = static_cast<Index_t>(
        window_start + static_cast<uint32_t>(curand(&state)) %
                           static_cast<uint32_t>(window_size));
    if (candidate == gid)
      continue;
    if (list_contains_id(filled_ids, filled, candidate))
      continue;
    filled_ids[filled++] = candidate;
  }

  // Phase 2: fill remaining slots from full range [0, n_inserted_instances).
  // Duplicate-check covers window nodes already in filled_ids.
  while (filled < max_unique_fill) {
    const Index_t candidate =
        static_cast<Index_t>(curand(&state) % n_inserted_instances);
    if (candidate == gid)
      continue;
    if (list_contains_id(filled_ids, filled, candidate))
      continue;
    filled_ids[filled++] = candidate;
  }

  // Single sequential write pass to global memory.
  for (int i = 0; i < filled; ++i) {
    my_neighbors[i].id_with_flag() = static_cast<int>(filled_ids[i]);
    my_distances[i] = dist_sentinel;
  }
  for (int i = filled; i < width; ++i) {
    my_neighbors[i].id_with_flag() = id_sentinel;
    my_distances[i] = dist_sentinel;
  }
}

template <typename Data_t, typename Idx_t, typename Index_t = int>
__global__ void sample_forward_candidates_kernel(
    InternalID_t *graph_buffer, int *candidate_buffer, int graph_width,
    int candidate_width, uint32_t batch_size, uint32_t batch_start_offset,
    uint32_t n_neighbors, uint32_t n_inserted_instances) {

  const size_t row_local =
      static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_local >= batch_size)
    return;

  const Index_t gid = static_cast<Index_t>(batch_start_offset + row_local);

  InternalID_t *my_neighbors =
      graph_buffer + static_cast<size_t>(gid) * graph_width;
  int *my_candidates =
      candidate_buffer + static_cast<size_t>(gid) * candidate_width;

  Index_t my_neighbors_l[MAX_NEIGHBORS];
  for (uint32_t i = 0; i < n_neighbors; ++i) {
    const int nid = my_neighbors[i].id();
    my_neighbors_l[i] =
        (nid >= 0 && static_cast<uint32_t>(nid) < n_inserted_instances)
            ? static_cast<Index_t>(nid)
            : std::numeric_limits<Index_t>::max();
  }
  for (uint32_t i = n_neighbors; i < static_cast<uint32_t>(MAX_NEIGHBORS);
       ++i) {
    my_neighbors_l[i] = std::numeric_limits<Index_t>::max();
  }

  int candidate_count = 0;

  for (uint32_t j = 0; j < n_neighbors && candidate_count < candidate_width;
       ++j) {
    InternalID_t neighbor_id_raw = my_neighbors[j];
    const Index_t neighbor_id = static_cast<Index_t>(neighbor_id_raw.id());

    // Cache the 2-hop neighbor IDs in registers to avoid repeated global reads
    // inside the inner k-loop. InternalID_t has no default constructor, so
    // we store the decoded int id directly.
    const InternalID_t *neigh2 =
        graph_buffer + static_cast<size_t>(neighbor_id) * graph_width;
    int neigh2_ids[MAX_NEIGHBORS];
    for (uint32_t k = 0; k < n_neighbors; ++k) {
      neigh2_ids[k] = neigh2[k].id();
    }

    for (uint32_t k = 0; k < n_neighbors && candidate_count < candidate_width;
         ++k) {
      if (static_cast<uint32_t>(neigh2_ids[k]) >= n_inserted_instances)
        continue;

      const Index_t candidate_id = static_cast<Index_t>(neigh2_ids[k]);

      if (candidate_id == gid)
        continue;
      if (list_contains_id<Index_t>(
              my_neighbors_l, static_cast<int>(n_neighbors), candidate_id))
        continue;
      if (list_contains_id<int>(my_candidates, candidate_count,
                                static_cast<int>(candidate_id)))
        continue;

      my_candidates[candidate_count++] = static_cast<int>(candidate_id);
    }
  }

  // Fill only the unused tail with -1 (avoids the upfront 256-write zero pass).
  for (int i = candidate_count; i < candidate_width; ++i) {
    my_candidates[i] = -1;
  }
}

// compute distances ────────────────────────────────────────────────
//
// For each batch node, compute the squared L2 distance to every entry in its
// candidate_buffer row and write results into candidate_dists_buffer at the
// same (node, slot) position.  Slots that are invalid (candidate == -1, out of
// range, self, or already a neighbor) receive a negative sentinel (-1) so the
// update stage can skip them cheaply without re-checking.
//
// Caller must allocate candidate_dists_buffer with shape
//   [n_inserted_instances + batch_size] × MAX_CANDIDATES.
template <typename Data_t, typename Idx_t, typename Index_t = int>
__global__ void compute_candidate_distances_kernel(
    const Data_t *input_data, const InternalID_t *graph_buffer,
    const int *candidate_buffer, Data_t *candidate_dists_buffer,
    uint32_t batch_size, uint32_t batch_start_offset, uint32_t n_neighbors,
    uint32_t n_features, uint32_t data_block_size, uint32_t n_candidates,
    uint32_t n_inserted_instances) {

  const size_t row_local =
      static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_local >= batch_size)
    return;

  const Index_t gid = static_cast<Index_t>(batch_start_offset) + row_local;

  const InternalID_t *my_neighbors =
      graph_buffer + static_cast<size_t>(gid) * MAX_NEIGHBORS;
  const int *my_candidates =
      candidate_buffer + static_cast<size_t>(gid) * n_candidates;
  Data_t *my_dists_out =
      candidate_dists_buffer + static_cast<size_t>(gid) * n_candidates;

  // Cache neighbor IDs in registers for fast membership checks.
  Index_t neighbors_local[MAX_NEIGHBORS];
  for (uint32_t i = 0; i < n_neighbors; ++i)
    neighbors_local[i] = static_cast<Index_t>(my_neighbors[i].id());

  const size_t offset1 = static_cast<size_t>(gid) * data_block_size;

  for (uint32_t i = 0; i < n_candidates; ++i) {
    const int candidate = my_candidates[i];

    if (candidate < 0 ||
        static_cast<uint32_t>(candidate) >= n_inserted_instances ||
        static_cast<Index_t>(candidate) == gid ||
        list_contains_id<Index_t>(neighbors_local,
                                  static_cast<int>(n_neighbors),
                                  static_cast<Index_t>(candidate))) {
      my_dists_out[i] = static_cast<Data_t>(-1); // sentinel: skip in update
      continue;
    }

    const size_t offset2 = static_cast<size_t>(candidate) * data_block_size;
    float dot = 0.0f;
    for (uint32_t d = 0; d < n_features; ++d) {
      dot += static_cast<float>(input_data[offset1 + d]) *
             static_cast<float>(input_data[offset2 + d]);
    }
    float dist_sq = static_cast<float>(input_data[offset1 + n_features]) +
                    static_cast<float>(input_data[offset2 + n_features]) -
                    2.0f * dot;
    my_dists_out[i] = static_cast<Data_t>(dist_sq < 0.0f ? 0.0f : dist_sq);
  }
}

template <typename Data_t, typename Index_t = int>
__device__ __forceinline__ void
sort_candidates(const Index_t *cand_ids, const Data_t *cand_dists,
                Index_t *sorted_cand_ids, Data_t *sorted_cand_dists,
                uint32_t sort_size, int count) {

  for (int i = 0; i < sort_size; ++i) {
    sorted_cand_ids[i] = -1;
    sorted_cand_dists[i] = -1;
  }

  int filled = 0;

  for (int i = 0; i < count; ++i) {
    Index_t cid = cand_ids[i];
    Data_t dist = cand_dists[i];

    if (filled >= sort_size && dist >= sorted_cand_dists[sort_size - 1])
      continue;

    int pos = filled < sort_size ? filled : sort_size - 1;

    while (pos > 0) {
      bool shift = (dist < sorted_cand_dists[pos - 1]) ||
                   (dist == sorted_cand_dists[pos - 1] &&
                    cid < sorted_cand_ids[pos - 1]);
      if (!shift)
        break;
      sorted_cand_ids[pos] = sorted_cand_ids[pos - 1];
      sorted_cand_dists[pos] = sorted_cand_dists[pos - 1];
      --pos;
    }
    sorted_cand_ids[pos] = cid;
    sorted_cand_dists[pos] = dist;
    if (filled < sort_size)
      ++filled;
  }
}

template <typename Data_t, typename Index_t>
__device__ __forceinline__ bool
insert_into_neighbors(const Index_t *sorted_cand_ids,
                      const Data_t *sorted_cand_dists, uint32_t sort_size,
                      Index_t *neighbors_local, Data_t *distances_local,
                      int neighbors_local_size) {
  bool updated = false;

  for (int i = 0; i < sort_size; ++i) {
    const Index_t cid = sorted_cand_ids[i];
    const Data_t dist = sorted_cand_dists[i];

    if (list_contains_id<Index_t>(neighbors_local, neighbors_local_size, cid))
      continue;

    const int tail = neighbors_local_size - 1;
    if (dist > distances_local[tail] || dist < static_cast<Data_t>(0) ||
        (dist == distances_local[tail] && cid >= neighbors_local[tail]))
      break;

    int pos = tail;
    while (pos > 0) {
      const bool shift =
          (dist < distances_local[pos - 1]) ||
          (dist == distances_local[pos - 1] && cid < neighbors_local[pos - 1]);
      if (!shift)
        break;
      neighbors_local[pos] = neighbors_local[pos - 1];
      distances_local[pos] = distances_local[pos - 1];
      --pos;
    }
    neighbors_local[pos] = cid;
    distances_local[pos] = dist;
    updated = true;
  }
  return updated;
}

// Reads candidate IDs and their pre-computed distances, collects the valid
// (non-sentinel) pairs, sorts them, and updates the neighbor list in-place.
template <typename Data_t, typename Idx_t, typename Index_t = int>
__global__ void update_forward_neighbors_kernel(
    InternalID_t *graph_buffer, Data_t *distance_buffer,
    const int *candidate_buffer, const Data_t *candidate_dists_buffer,
    uint32_t batch_size, uint32_t batch_start_offset) {

  const size_t row_local =
      static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_local >= batch_size)
    return;

  const Index_t gid = static_cast<Index_t>(batch_start_offset) + row_local;

  InternalID_t *my_neighbors =
      graph_buffer + static_cast<size_t>(gid) * MAX_NEIGHBORS;
  Data_t *my_distances =
      distance_buffer + static_cast<size_t>(gid) * MAX_NEIGHBORS;
  const int *my_candidates =
      candidate_buffer + static_cast<size_t>(gid) * MAX_CANDIDATES;
  const Data_t *my_dists_in =
      candidate_dists_buffer + static_cast<size_t>(gid) * MAX_CANDIDATES;

  // Load current neighbor list into registers.
  Index_t neighbors_local[MAX_NEIGHBORS];
  Data_t distances_local[MAX_NEIGHBORS];
  for (uint32_t i = 0; i < MAX_NEIGHBORS; ++i) {
    neighbors_local[i] = static_cast<Index_t>(my_neighbors[i].id());
    distances_local[i] = my_distances[i];
  }

  // Sort candidates by distance.
  Index_t sorted_cand_ids[MAX_NEIGHBORS];
  Data_t sorted_cand_dists[MAX_NEIGHBORS];
  sort_candidates(my_candidates, my_dists_in, sorted_cand_ids,
                  sorted_cand_dists, MAX_NEIGHBORS, MAX_CANDIDATES);

  // Insert sorted candidates into neighbor list.
  const bool updated =
      insert_into_neighbors(sorted_cand_ids, sorted_cand_dists, MAX_NEIGHBORS,
                            neighbors_local, distances_local, MAX_NEIGHBORS);

  // Write back updated neighbor list to global memory.
  if (updated) {
    for (uint32_t i = 0; i < MAX_NEIGHBORS; ++i) {
      my_neighbors[i] = InternalID_t(neighbors_local[i]);
      my_distances[i] = distances_local[i];
    }
  }
}

// Scatter pass: for each valid (batch_node i, candidate_slot s) pair,
// atomically claims a slot in candidate_node j's reverse list and writes (i,
// dist(i,j)). Slots beyond MAX_NEIGHBORS for any given j are silently dropped.
//
// Outputs:
//   reverse_cand_ids   [n_inserted_instances × MAX_NEIGHBORS] — batch-node IDs
//   reverse_cand_dists [n_inserted_instances × MAX_NEIGHBORS] — distances
//   reverse_cand_counts[n_inserted_instances]                 — filled slot
//   counts
//
// Caller must zero-initialize reverse_cand_counts and fill reverse_cand_ids /
// reverse_cand_dists with -1 sentinels before launch.
template <typename Data_t, typename Index_t = int>
__global__ void scatter_reverse_candidates_kernel(
    const int *candidate_buffer,          // [n_total × MAX_CANDIDATES]
    const Data_t *candidate_dists_buffer, // [n_total × MAX_CANDIDATES]
    int *reverse_cand_ids,      // [n_inserted_instances × MAX_NEIGHBORS]
    Data_t *reverse_cand_dists, // [n_inserted_instances × MAX_NEIGHBORS]
    int *reverse_cand_counts,   // [n_inserted_instances] atomic counters
    uint32_t batch_size, uint32_t batch_start_offset,
    uint32_t n_inserted_instances) {

  // One thread per (batch_node, candidate_slot).
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= static_cast<size_t>(batch_size) * MAX_CANDIDATES)
    return;

  const uint32_t row_local = static_cast<uint32_t>(tid / MAX_CANDIDATES);
  const uint32_t slot = static_cast<uint32_t>(tid % MAX_CANDIDATES);
  const Index_t gid = static_cast<Index_t>(batch_start_offset + row_local);

  const int cand_id =
      candidate_buffer[static_cast<size_t>(gid) * MAX_CANDIDATES + slot];
  const Data_t dist =
      candidate_dists_buffer[static_cast<size_t>(gid) * MAX_CANDIDATES + slot];

  // Skip sentinel/invalid entries (dist < 0 is the -1 sentinel from
  // compute_candidate_distances_kernel; cand_id < 0 means unused slot).
  if (cand_id < 0 || static_cast<uint32_t>(cand_id) >= n_inserted_instances ||
      dist < static_cast<Data_t>(0))
    return;

  // Atomically claim a slot in cand_id's reverse list.
  const int pos = atomicAdd(&reverse_cand_counts[cand_id], 1);
  if (pos >= MAX_NEIGHBORS)
    return; // overflow: too many batch nodes pointing to this candidate, drop

  const size_t out_offset = static_cast<size_t>(cand_id) * MAX_NEIGHBORS + pos;
  reverse_cand_ids[out_offset] = static_cast<int>(gid);
  reverse_cand_dists[out_offset] = dist;
}

// For each existing node j, reads the pre-scattered reverse candidate list
// (built by scatter_reverse_candidates_kernel), sorts the entries, and inserts
// the best ones into j's neighbor list using the same sort+insert logic as
// update_forward_neighbors_kernel.
template <typename Data_t, typename Idx_t, typename Index_t = int>
__global__ void update_reverse_neighbors_kernel(
    InternalID_t *graph_buffer, Data_t *distance_buffer,
    const int *reverse_cand_ids, const Data_t *reverse_cand_dists,
    const int *reverse_cand_counts, int *d_dirty_flags,
    uint32_t n_inserted_instances) {

  const size_t node_id =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (node_id >= n_inserted_instances)
    return;

  InternalID_t *my_neighbors = graph_buffer + node_id * MAX_NEIGHBORS;
  Data_t *my_distances = distance_buffer + node_id * MAX_NEIGHBORS;
  const int *my_rev_ids = reverse_cand_ids + node_id * MAX_NEIGHBORS;
  const Data_t *my_rev_dists = reverse_cand_dists + node_id * MAX_NEIGHBORS;
  const int count =
      min(reverse_cand_counts[node_id], static_cast<int>(MAX_NEIGHBORS));

  if (count == 0)
    return;

  // Load current neighbor list into registers.
  Index_t neighbors_local[MAX_NEIGHBORS];
  Data_t distances_local[MAX_NEIGHBORS];
  for (int i = 0; i < MAX_NEIGHBORS; ++i) {
    neighbors_local[i] = static_cast<Index_t>(my_neighbors[i].id());
    distances_local[i] = my_distances[i];
  }

  // Sort the scattered candidates (top-MAX_NEIGHBORS by distance).
  Index_t sorted_cand_ids[MAX_NEIGHBORS];
  Data_t sorted_cand_dists[MAX_NEIGHBORS];
  sort_candidates<Data_t, Index_t>(my_rev_ids, my_rev_dists, sorted_cand_ids,
                                   sorted_cand_dists, MAX_NEIGHBORS, count);

  // Insert sorted candidates into neighbor list.
  const bool updated = insert_into_neighbors<Data_t, Index_t>(
      sorted_cand_ids, sorted_cand_dists, MAX_NEIGHBORS, neighbors_local,
      distances_local, MAX_NEIGHBORS);

  if (updated) {
    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
      my_neighbors[i] = InternalID_t(neighbors_local[i]);
      my_distances[i] = distances_local[i];
    }
    d_dirty_flags[node_id] = 1;
  }
}

// ── Host wrappers
// ─────────────────────────────────────────────────────────────

template <typename Data_t, typename Idx_t, typename Index_t = int>
void sample_forward_candidate(std::vector<Idx_t> &batch_indices_,
                              uint32_t n_neighbors,
                              uint32_t n_inserted_instances,
                              InternalID_t *graph_buffer_,
                              int *candidate_buffer_, cudaStream_t stream) {
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
          n_rows, batch_start_offset, n_neighbors, n_inserted_instances);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename Data_t, typename Idx_t, typename Index_t = int>
void compute_candidate_distances(
    std::vector<Idx_t> &batch_indices_, uint32_t n_neighbors,
    uint32_t n_features, uint32_t data_block_size,
    uint32_t n_inserted_instances, const Data_t *input_data,
    const InternalID_t *graph_buffer_, const int *candidate_buffer_,
    Data_t *candidate_dists_buffer_, cudaStream_t stream) {
  const uint32_t n_rows = static_cast<uint32_t>(batch_indices_.size());
  if (n_rows == 0)
    return;
  const uint32_t batch_start_offset =
      static_cast<uint32_t>(batch_indices_.front());
  const int threads_per_block = 256;
  const int blocks = (n_rows + threads_per_block - 1) / threads_per_block;

  compute_candidate_distances_kernel<Data_t, Idx_t, Index_t>
      <<<blocks, threads_per_block, 0, stream>>>(
          input_data, graph_buffer_, candidate_buffer_, candidate_dists_buffer_,
          n_rows, batch_start_offset, n_neighbors, n_features, data_block_size,
          MAX_CANDIDATES, n_inserted_instances);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename Data_t, typename Idx_t, typename Index_t = int>
void update_forward_neighbors(std::vector<Idx_t> &batch_indices_,
                              uint32_t n_neighbors, InternalID_t *graph_buffer_,
                              Data_t *distance_buffer_,
                              const int *candidate_buffer_,
                              const Data_t *candidate_dists_buffer_,
                              cudaStream_t stream) {
  const uint32_t n_rows = static_cast<uint32_t>(batch_indices_.size());
  if (n_rows == 0)
    return;
  const uint32_t batch_start_offset =
      static_cast<uint32_t>(batch_indices_.front());

  const int threads_per_block = 256;
  const int blocks = (n_rows + threads_per_block - 1) / threads_per_block;

  update_forward_neighbors_kernel<Data_t, Idx_t, Index_t>
      <<<blocks, threads_per_block, 0, stream>>>(
          graph_buffer_, distance_buffer_, candidate_buffer_,
          candidate_dists_buffer_, n_rows, batch_start_offset);
  RAFT_CUDA_TRY(cudaGetLastError());
}

// Converts (batch_size × MAX_CANDIDATES) candidate_buffer /
// candidate_dists_buffer into (n_inserted_instances × MAX_NEIGHBORS) reverse
// neighbor lists, then updates graph_buffer / distance_buffer in-place for
// every existing node.
//
// Pipeline:
//   1. scatter_reverse_candidates_kernel — scatter (i→j, dist) pairs by j
//   2. update_reverse_neighbors_kernel   — sort + insert for each node j
template <typename Data_t, typename Idx_t, typename Index_t = int>
void update_reverse_neighbors(std::vector<Idx_t> &batch_indices_,
                              uint32_t n_neighbors,
                              uint32_t n_inserted_instances,
                              InternalID_t *graph_buffer_,
                              Data_t *distance_buffer_,
                              const int *candidate_buffer_,
                              const Data_t *candidate_dists_buffer_,
                              int *d_dirty_flags_, cudaStream_t stream) {
  const uint32_t n_rows = static_cast<uint32_t>(batch_indices_.size());
  if (n_rows == 0)
    return;
  const uint32_t batch_start_offset =
      static_cast<uint32_t>(batch_indices_.front());

  // Allocate reverse buffers.
  const size_t rev_ids_size =
      static_cast<size_t>(n_inserted_instances) * MAX_NEIGHBORS * sizeof(int);
  const size_t rev_dists_size = static_cast<size_t>(n_inserted_instances) *
                                MAX_NEIGHBORS * sizeof(Data_t);
  const size_t rev_counts_size =
      static_cast<size_t>(n_inserted_instances) * sizeof(int);

  int *reverse_cand_ids;
  Data_t *reverse_cand_dists;
  int *reverse_cand_counts;

  RAFT_CUDA_TRY(cudaMallocAsync(&reverse_cand_ids, rev_ids_size, stream));
  RAFT_CUDA_TRY(cudaMallocAsync(&reverse_cand_dists, rev_dists_size, stream));
  RAFT_CUDA_TRY(cudaMallocAsync(&reverse_cand_counts, rev_counts_size, stream));

  // Initialize: counts = 0, ids/dists = -1 sentinel.
  RAFT_CUDA_TRY(
      cudaMemsetAsync(reverse_cand_counts, 0, rev_counts_size, stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(reverse_cand_ids, 0xFF, rev_ids_size, stream));
  // 0xFF fill for floats gives a NaN, not a clean -1; fill with -1.0f via
  // kernel would be expensive.  Instead update_reverse_neighbors_kernel guards
  // on reverse_cand_counts so uninitialized dist slots beyond count are never
  // read.

  // Stage 1: scatter (batch_node → candidate) pairs into reverse lists.
  {
    const size_t n_work = static_cast<size_t>(n_rows) * MAX_CANDIDATES;
    const int tpb = 256;
    const int blocks = static_cast<int>((n_work + tpb - 1) / tpb);
    scatter_reverse_candidates_kernel<Data_t, Index_t>
        <<<blocks, tpb, 0, stream>>>(candidate_buffer_, candidate_dists_buffer_,
                                     reverse_cand_ids, reverse_cand_dists,
                                     reverse_cand_counts, n_rows,
                                     batch_start_offset, n_inserted_instances);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  // Stage 2: for each existing node, sort its scattered candidates and update.
  {
    const int tpb = 256;
    const int blocks = (static_cast<int>(n_inserted_instances) + tpb - 1) / tpb;
    update_reverse_neighbors_kernel<Data_t, Idx_t, Index_t>
        <<<blocks, tpb, 0, stream>>>(graph_buffer_, distance_buffer_,
                                     reverse_cand_ids, reverse_cand_dists,
                                     reverse_cand_counts, d_dirty_flags_,
                                     n_inserted_instances);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  RAFT_CUDA_TRY(cudaFreeAsync(reverse_cand_ids, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(reverse_cand_dists, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(reverse_cand_counts, stream));
}

// ── Embedding initialisation kernels ──────────────────────────────────────
//
// Step 1: assign a random Gaussian value to every dimension of every batch
//         node so that nodes with no usable existing neighbours still end up
//         at a finite position.
// Step 2: for batch nodes that have at least one *existing* (pre-batch)
//         neighbour, overwrite the random embedding with a distance-weighted
//         average of those neighbours' current embeddings.  Nodes that have
//         no existing neighbours keep their random initialisation.

template <typename Data_t, typename Idx_t, typename Index_t = int>
__global__ __forceinline__ void
initalize_random_embeddings_kernel(Data_t *embedding, int n_components,
                                   uint32_t batch_size,
                                   uint32_t batch_start_offset) {
  const size_t row_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_local >= batch_size)
    return;

  const size_t row = static_cast<size_t>(batch_start_offset) + row_local;
  curandStatePhilox4_32_10_t state;
  curand_init(42, row, 0, &state);

  for (int dim = 0; dim < n_components; ++dim) {
    embedding[row * n_components + dim] =
        static_cast<Data_t>(curand_normal(&state));
  }
}

template <typename Data_t>
__global__ void initialize_embeddings_kernel(
    const InternalID_t *graph_buffer, const Data_t *distance_buffer,
    Data_t *embedding, uint32_t batch_size, uint32_t batch_start_offset,
    uint32_t n_neighbors, uint32_t n_inserted_instances,
    uint32_t n_components) {
  const size_t row_local = static_cast<size_t>(blockIdx.x);
  if (row_local >= batch_size)
    return;

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
      if (nid < 0 || static_cast<uint32_t>(nid) >= n_inserted_instances ||
          nid >= static_cast<int>(batch_start_offset)) {
        s_neighbor_ids[j] = -1;
        s_weights[j] = 0.0f;
        continue;
      }
      if (my_dists[j] < min_dist)
        min_dist = my_dists[j];
      s_neighbor_ids[j] = nid;
    }

    if (min_dist == std::numeric_limits<Data_t>::max()) {
      s_weight_sum = 0.0f;
    } else {
      float weight_sum = 0.0f;
      for (uint32_t j = 0; j < n_neighbors; ++j) {
        if (s_neighbor_ids[j] < 0) {
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

  if (s_weight_sum <= 0.0f)
    return;

  for (uint32_t dim = threadIdx.x; dim < n_components; dim += blockDim.x) {
    float accum = 0.0f;
    for (uint32_t j = 0; j < n_neighbors; ++j) {
      const int nid = s_neighbor_ids[j];
      if (nid < 0)
        continue;
      accum += s_weights[j] *
               static_cast<float>(
                   embedding[static_cast<size_t>(nid) * n_components + dim]);
    }
    embedding[row * n_components + dim] =
        static_cast<Data_t>(accum / s_weight_sum);
  }
}

template <typename Data_t, typename Idx_t, typename Index_t = int>
void initialize_embeddings(NNDBuffers<Data_t, Index_t> &buffers,
                           std::vector<Idx_t> &batch_indices, Data_t *embedding,
                           uint32_t n_components, uint32_t n_neighbors,
                           uint32_t n_inserted_instances, cudaStream_t stream) {
  if (embedding == nullptr)
    return;
  const uint32_t batch_size = static_cast<uint32_t>(batch_indices.size());
  if (batch_size == 0)
    return;

  const uint32_t batch_start_offset =
      static_cast<uint32_t>(batch_indices.front());
  const int threads_per_block =
      static_cast<int>(std::min<uint32_t>(n_components, 256));

  initalize_random_embeddings_kernel<Data_t, Idx_t, Index_t>
      <<<batch_size, threads_per_block, 0, stream>>>(
          embedding, static_cast<int>(n_components), batch_size,
          batch_start_offset);

  initialize_embeddings_kernel<Data_t>
      <<<batch_size, threads_per_block, 0, stream>>>(
          buffers.graph_buffer_.data_handle(),
          buffers.dists_buffer_.data_handle(), embedding, batch_size,
          batch_start_offset, n_neighbors, n_inserted_instances, n_components);
  RAFT_CUDA_TRY(cudaGetLastError());
}

// ── Full insertion pipeline ────────────────────────────────────────────────
//
// n_inserted_instances: total node count INCLUDING this batch
//   (= storage.n_inserted_instances after append_data).
//   batch_start_offset = batch_indices.front()
//                      = n_inserted_instances - batch_size.
//   Existing nodes: [0, batch_start_offset).
//   Batch nodes   : [batch_start_offset, n_inserted_instances).
//
// Dirty flags are zeroed at the start and accumulated across all iterations.
// After the loop, graph + dists + flags are copied to host; a CPU diff then
// populates affected_indices and new_edges, and h_graph / h_dists are updated.
//
// new_edges: (src, dst, dist) for every newly added neighbor slot.
//   Existing nodes: slots not present in the pre-insertion h_graph.
//   Batch nodes   : all valid final neighbor slots (no prior h_graph row).
template <typename Data_t, typename Idx_t, typename Index_t = int>
void run_insertion(
    NNDBuffers<Data_t, Index_t> &buffers, std::vector<Idx_t> &batch_indices,
    Data_t *input_data, uint32_t data_block_size, uint32_t n_features,
    uint32_t n_neighbors, uint32_t n_inserted_instances, uint32_t max_iter,
    size_t seed, cudaStream_t stream, std::vector<Idx_t> &affected_indices,
    std::vector<std::tuple<Idx_t, Idx_t, float, float, float>> &new_edges,
    Data_t *embedding = nullptr, uint32_t n_components = 0,
    uint32_t window_start = 0) {
  const uint32_t batch_size = static_cast<uint32_t>(batch_indices.size());
  if (batch_size == 0)
    return;

  // New nodes change the graph topology, so previously-evaluated pairs may
  // need to be reconsidered. Clear the bloom filter so update_kernel can
  // re-evaluate old neighbors after this insertion.
  buffers.bloom_filter.clear();

  const uint32_t batch_start_offset =
      static_cast<uint32_t>(batch_indices.front());

  InternalID_t *graph_ptr = buffers.graph_buffer_.data_handle();
  Data_t *dists_ptr = buffers.dists_buffer_.data_handle();
  int *cand_ptr = buffers.candidate_buffer_.data_handle();
  int *dirty_ptr = buffers.d_dirty_flags_.data_handle();

  const int tpb = 256;

  // ── Step 1: Zero dirty flags for all current nodes ───────────────────────
  RAFT_CUDA_TRY(cudaMemsetAsync(
      dirty_ptr, 0, static_cast<size_t>(n_inserted_instances) * sizeof(int),
      stream));

  // ── Step 2: Compute L2 norms for the batch rows ──────────────────────────
  // preprocess_l2_norms_kernel uses local blockIdx.x; pass batch-slice ptr.
  {
    const size_t shmem = BLOCK_SIZE * sizeof(float);
    preprocess_l2_norms_kernel<Data_t>
        <<<static_cast<int>(batch_size), BLOCK_SIZE, shmem, stream>>>(
            input_data +
                static_cast<size_t>(batch_start_offset) * data_block_size,
            batch_size, n_features, data_block_size);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  // ── Step 3: Random-initialize KNN lists for batch nodes ──────────────────
  // initialize_random_neighbors_kernel uses local row index; pass batch-slice
  // pointers.  n_inserted_instances includes the batch, so random sampling
  // may also draw other batch nodes as initial neighbors.
  {
    const int blocks = (static_cast<int>(batch_size) + tpb - 1) / tpb;
    initialize_random_neighbors_kernel<Data_t, Idx_t, Index_t>
        <<<blocks, tpb, 0, stream>>>(
            graph_ptr + static_cast<size_t>(batch_start_offset) * MAX_NEIGHBORS,
            dists_ptr + static_cast<size_t>(batch_start_offset) * MAX_NEIGHBORS,
            MAX_NEIGHBORS, batch_size, batch_start_offset, n_inserted_instances,
            window_start, seed);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  // Candidate-distance scratch buffer (batch_size rows).  Pointer shifted back
  // by batch_start_offset rows so global gid-based indexing maps into it.
  Data_t *cand_dists_storage;
  RAFT_CUDA_TRY(cudaMallocAsync(&cand_dists_storage,
                                static_cast<size_t>(batch_size) *
                                    MAX_CANDIDATES * sizeof(Data_t),
                                stream));
  Data_t *cand_dists_ptr =
      cand_dists_storage -
      static_cast<ptrdiff_t>(batch_start_offset) * MAX_CANDIDATES;

  // ── Step 4: NND iteration loop ───────────────────────────────────────────
  for (uint32_t iter = 0; iter < max_iter; ++iter) {
    sample_forward_candidate<Data_t, Idx_t, Index_t>(
        batch_indices, n_neighbors, n_inserted_instances, graph_ptr, cand_ptr,
        stream);

    compute_candidate_distances<Data_t, Idx_t, Index_t>(
        batch_indices, n_neighbors, n_features, data_block_size,
        n_inserted_instances, input_data, graph_ptr, cand_ptr, cand_dists_ptr,
        stream);

    // Forward: update batch nodes' KNN lists.
    update_forward_neighbors<Data_t, Idx_t, Index_t>(
        batch_indices, n_neighbors, graph_ptr, dists_ptr, cand_ptr,
        cand_dists_ptr, stream);

    // Reverse: update existing nodes' KNN lists (skip if none exist yet).
    if (batch_start_offset > 0) {
      update_reverse_neighbors<Data_t, Idx_t, Index_t>(
          batch_indices, n_neighbors, batch_start_offset, graph_ptr, dists_ptr,
          cand_ptr, cand_dists_ptr, dirty_ptr, stream);
    }
  }

  RAFT_CUDA_TRY(cudaFreeAsync(cand_dists_storage, stream));

  // ── Step 5: Copy final graph + dirty flags to host ───────────────────────
  const size_t n_graph_elems =
      static_cast<size_t>(n_inserted_instances) * MAX_NEIGHBORS;
  raft::copy(buffers.graph_host_buffer_.data_handle(), graph_ptr, n_graph_elems,
             stream);
  raft::copy(buffers.dists_host_buffer_.data_handle(), dists_ptr, n_graph_elems,
             stream);
  raft::copy(buffers.h_dirty_flags_.data_handle(), dirty_ptr,
             static_cast<size_t>(n_inserted_instances), stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  // ── Step 6: CPU diff ─────────────────────────────────────────────────────
  const InternalID_t *new_graph = buffers.graph_host_buffer_.data_handle();
  const Data_t *new_dists_buf = buffers.dists_host_buffer_.data_handle();
  const int *dirty = buffers.h_dirty_flags_.data_handle();

  for (uint32_t i = 0; i < batch_size; ++i) {
    Idx_t node_id = batch_start_offset + i;
    affected_indices.push_back(node_id);
    const InternalID_t *row =
        new_graph + static_cast<size_t>(node_id) * MAX_NEIGHBORS;

    for (uint32_t k = 0; k < n_neighbors; ++k) {
      const int nid = row[k].id();
      if (nid < 0 || static_cast<uint32_t>(nid) >= n_inserted_instances)
        continue;
      new_edges.emplace_back(node_id, static_cast<Idx_t>(nid), 0.f, -1.f, -1.f);
    }

    auto *dst_graph = buffers.h_graph.data_handle() +
                      static_cast<size_t>(node_id) * n_neighbors;
    auto *dst_dists = buffers.h_dists.data_handle() +
                      static_cast<size_t>(node_id) * n_neighbors;
    const auto *src_graph =
        new_graph + static_cast<size_t>(node_id) * MAX_NEIGHBORS;
    const auto *src_dists =
        new_dists_buf + static_cast<size_t>(node_id) * MAX_NEIGHBORS;
    for (size_t k = 0; k < n_neighbors; ++k) {
      dst_graph[k] = src_graph[k];
      dst_dists[k] = src_dists[k];
    }
  }

  for (uint32_t node = 0; node < n_inserted_instances - batch_start_offset;
       ++node) {
    if (!dirty[node])
      continue;

    affected_indices.push_back(static_cast<Idx_t>(node));

    const InternalID_t *new_row = new_graph + node * MAX_NEIGHBORS;
    const Data_t *new_dist_row = new_dists_buf + node * MAX_NEIGHBORS;
    auto *dst_graph =
        buffers.h_graph.data_handle() + static_cast<size_t>(node) * n_neighbors;
    auto *dst_dists =
        buffers.h_dists.data_handle() + static_cast<size_t>(node) * n_neighbors;
    for (size_t k = 0; k < n_neighbors; ++k) {
      dst_graph[k] = new_row[k];
      dst_dists[k] = new_dist_row[k];

      auto new_nid = new_row[k].id();
      if (new_nid >= batch_start_offset &&
          static_cast<uint32_t>(new_nid) < n_inserted_instances) {
        new_edges.emplace_back(static_cast<Idx_t>(node),
                               static_cast<Idx_t>(new_nid), 0.f, -1.f, -1.f);
      }
    }
  }

  // ── Step 7: Initialise embeddings for batch nodes ────────────────────────
  initialize_embeddings<Data_t, Idx_t, Index_t>(
      buffers, batch_indices, embedding, n_components, n_neighbors,
      n_inserted_instances, stream);
}

} // namespace gpumap
