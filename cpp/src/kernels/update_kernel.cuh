#pragma once

#include "../utils/matrix_utils.cuh"
#include "buffers.cuh"

#if defined(__clang__) && defined(__CUDA__)
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <limits>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace gpumap {

constexpr int DIRTY_NONE = 0;
constexpr int DIRTY_LIST_CHANGED = 1;
constexpr int DIRTY_NEIGHBOR_CHANGED = 2;

template <typename Index_t> struct ResultItem;

template <> class ResultItem<int> {
private:
  using Index_t = int;
  using Dist_t = float;
  Index_t id_;
  Dist_t dist_;

public:
  __host__ __device__ ResultItem()
      : id_(std::numeric_limits<Index_t>::max()),
        dist_(std::numeric_limits<Dist_t>::max()) {};
  __host__ __device__ ResultItem(const Index_t id_with_flag, const Dist_t dist)
      : id_(id_with_flag), dist_(dist) {};
  __host__ __device__ bool is_new() const { return id_ >= 0; }
  __host__ __device__ Index_t &id_with_flag() { return id_; }
  __host__ __device__ Index_t id() const {
    if (is_new())
      return id_;
    return -id_ - 1;
  }
  __host__ __device__ Dist_t &dist() { return dist_; }

  __host__ __device__ void mark_old() {
    if (id_ >= 0)
      id_ = -id_ - 1;
  }

  __host__ __device__ bool operator<(const ResultItem<Index_t> &other) const {
    if (dist_ == other.dist_)
      return id() < other.id();
    return dist_ < other.dist_;
  }
  __host__ __device__ bool operator==(const ResultItem<Index_t> &other) const {
    return id() == other.id();
  }
  __host__ __device__ bool operator>=(const ResultItem<Index_t> &other) const {
    return !(*this < other);
  }
  __host__ __device__ bool operator<=(const ResultItem<Index_t> &other) const {
    return (*this == other) || (*this < other);
  }
  __host__ __device__ bool operator>(const ResultItem<Index_t> &other) const {
    return !(*this <= other);
  }
  __host__ __device__ bool operator!=(const ResultItem<Index_t> &other) const {
    return !(*this == other);
  }
};

template <typename Data_t>
uint32_t insert_to_ordered_list(InternalID_t *list, Data_t *dist_list,
                                const uint32_t width,
                                const InternalID_t neighb_id,
                                const Data_t dist) {
  if (dist > dist_list[width - 1]) {
    return width;
  }

  uint32_t idx_insert = width;
  bool position_found = false;

  for (int i = 0; i < width; i++) {
    if (list[i].id() == neighb_id.id()) {
      return width;
    }
    if (!position_found && dist_list[i] > dist) {
      idx_insert = i;
      position_found = true;
    }
  }
  if (idx_insert == width) {
    return idx_insert;
  }

  memmove(list + idx_insert + 1, list + idx_insert,
          sizeof(*list) * (width - idx_insert - 1));
  memmove(dist_list + idx_insert + 1, dist_list + idx_insert,
          sizeof(*dist_list) * (width - idx_insert - 1));

  list[idx_insert] = neighb_id;
  dist_list[idx_insert] = dist;
  return idx_insert;
}

template <typename Index_t>
__global__ void add_rev_edges_kernel(const Index_t *graph, Index_t *rev_graph,
                                     int num_samples, int2 *list_sizes) {
  size_t list_id = blockIdx.x;
  int2 list_size = list_sizes[list_id];

  for (int idx = threadIdx.x; idx < list_size.x; idx += blockDim.x) {
    Index_t rev_list_id = graph[list_id * num_samples + idx];

    const bool invalid_neighbor =
        rev_list_id == std::numeric_limits<Index_t>::max() ||
        static_cast<size_t>(rev_list_id) >= static_cast<size_t>(gridDim.x);

    if (invalid_neighbor)
      continue;

    int idx_in_rev_list = atomicAdd(&list_sizes[rev_list_id].y, 1);

    if (idx_in_rev_list >= num_samples) {
      atomicExch(&list_sizes[rev_list_id].y, num_samples);
    } else {
      rev_graph[rev_list_id * num_samples + idx_in_rev_list] = list_id;
    }
  }
}

template <typename Data_t>
__device__ __forceinline__ void
load_vec(Data_t *vec_buffer, const Data_t *d_vec, const int load_dims,
         const int padding_dims, const int lane_id) {
  constexpr int num_load_elems_per_warp = raft::warp_size();
  for (int step = 0;
       step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
    int idx = step * num_load_elems_per_warp + lane_id;
    if (idx < load_dims) {
      vec_buffer[idx] = d_vec[idx];
    } else if (idx < padding_dims) {
      vec_buffer[idx] = 0.0f;
    }
  }
}

template <typename Index_t>
__device__ __forceinline__ bool is_valid_list_id(Index_t id, int n_lists) {
  return id >= 0 && static_cast<size_t>(id) < static_cast<size_t>(n_lists);
}

__device__ __forceinline__ uint bfe(uint lane_id, uint pos) {
  uint res;
  asm("bfe.u32 %0,%1,%2,%3;" : "=r"(res) : "r"(lane_id), "r"(pos), "r"(1));
  return res;
}

template <typename T>
__device__ __forceinline__ T xor_swap(T x, int mask, int dir) {
  T y = __shfl_xor_sync(raft::warp_full_mask(), x, mask, raft::warp_size());
  return x < y == dir ? y : x;
}

template <typename T>
__device__ __forceinline__ void warp_bitonic_sort(T *element_ptr,
                                                  const int lane_id) {
  static_assert(raft::warp_size() == 32);
  auto &element = *element_ptr;
  element = xor_swap(element, 0x01, bfe(lane_id, 1) ^ bfe(lane_id, 0));
  element = xor_swap(element, 0x02, bfe(lane_id, 2) ^ bfe(lane_id, 1));
  element = xor_swap(element, 0x01, bfe(lane_id, 2) ^ bfe(lane_id, 0));
  element = xor_swap(element, 0x04, bfe(lane_id, 3) ^ bfe(lane_id, 2));
  element = xor_swap(element, 0x02, bfe(lane_id, 3) ^ bfe(lane_id, 1));
  element = xor_swap(element, 0x01, bfe(lane_id, 3) ^ bfe(lane_id, 0));
  element = xor_swap(element, 0x08, bfe(lane_id, 4) ^ bfe(lane_id, 3));
  element = xor_swap(element, 0x04, bfe(lane_id, 4) ^ bfe(lane_id, 2));
  element = xor_swap(element, 0x02, bfe(lane_id, 4) ^ bfe(lane_id, 1));
  element = xor_swap(element, 0x01, bfe(lane_id, 4) ^ bfe(lane_id, 0));
  element = xor_swap(element, 0x10, bfe(lane_id, 4));
  element = xor_swap(element, 0x08, bfe(lane_id, 3));
  element = xor_swap(element, 0x04, bfe(lane_id, 2));
  element = xor_swap(element, 0x02, bfe(lane_id, 1));
  element = xor_swap(element, 0x01, bfe(lane_id, 0));
}

template <typename Index_t>
__device__ __forceinline__ void
remove_duplicates(Index_t *list_a, int size_a, Index_t *list_b, int size_b,
                  int &unique_counter, int execute_warp_id) {
  static_assert(raft::warp_size() == 32);
  if (!(threadIdx.x >= execute_warp_id * raft::warp_size() &&
        threadIdx.x <
            execute_warp_id * raft::warp_size() + raft::warp_size())) {
    return;
  }
  int lane_id = threadIdx.x % raft::warp_size();
  Index_t elem = std::numeric_limits<Index_t>::max();
  if (lane_id < size_a) {
    elem = list_a[lane_id];
  }
  warp_bitonic_sort(&elem, lane_id);

  if (elem != std::numeric_limits<Index_t>::max()) {
    list_a[lane_id] = elem;
  }

  Index_t elem_b = std::numeric_limits<Index_t>::max();
  if (lane_id < size_b) {
    elem_b = list_b[lane_id];
  }
  __syncwarp();

  int idx_l = 0;
  int idx_r = size_a;
  bool existed = false;
  while (idx_l < idx_r) {
    int idx = (idx_l + idx_r) / 2;
    Index_t val_a = list_a[idx];
    if (val_a == elem_b) {
      existed = true;
      break;
    }
    if (elem_b > val_a) {
      idx_l = idx + 1;
    } else {
      idx_r = idx;
    }
  }
  if (!existed && elem_b != std::numeric_limits<Index_t>::max()) {
    int idx = atomicAdd(&unique_counter, 1);
    list_a[size_a + idx] = elem_b;
  }
}

template <typename Index_t>
__device__ ResultItem<Index_t>
get_min_item(const Index_t id, const int idx_in_list, const Index_t *neighbs,
             const int neighb_count, const float *distances,
             const bool find_in_row = true) {
  int lane_id = threadIdx.x % raft::warp_size();

  static_assert(MAX_NUM_BI_SAMPLES == 64);

  int idx[MAX_NUM_BI_SAMPLES / raft::warp_size()];
  float dist[MAX_NUM_BI_SAMPLES / raft::warp_size()] = {
      std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};

  idx[0] = lane_id;
  idx[1] = raft::warp_size() + lane_id;

  if (idx[0] < neighb_count && neighbs[idx[0]] != id) {
    dist[0] =
        find_in_row
            ? distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + lane_id]
            : distances[idx_in_list + lane_id * SKEWED_MAX_NUM_BI_SAMPLES];
  }

  if (idx[1] < neighb_count && neighbs[idx[1]] != id) {
    dist[1] = find_in_row
                  ? distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES +
                              raft::warp_size() + lane_id]
                  : distances[idx_in_list + (raft::warp_size() + lane_id) *
                                                SKEWED_MAX_NUM_BI_SAMPLES];
  }

  if (dist[1] < dist[0]) {
    dist[0] = dist[1];
    idx[0] = idx[1];
  }

  __syncwarp();

  for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
    int other_idx = __shfl_down_sync(raft::warp_full_mask(), idx[0], offset);
    float other_dist =
        __shfl_down_sync(raft::warp_full_mask(), dist[0], offset);
    if (other_dist < dist[0]) {
      dist[0] = other_dist;
      idx[0] = other_idx;
    }
  }

  ResultItem<Index_t> result;
  result.dist() = __shfl_sync(raft::warp_full_mask(), dist[0], 0);
  const int winner_idx = __shfl_sync(raft::warp_full_mask(), idx[0], 0);
  result.id_with_flag() = (winner_idx >= 0 && winner_idx < neighb_count)
                              ? neighbs[winner_idx]
                              : std::numeric_limits<Index_t>::max();

  return result;
}

template <typename Index_t, typename Dist_t, typename ID_t = InternalID_t>
__device__ void insert_to_global_graph(ResultItem<Index_t> elem, size_t list_id,
                                       ID_t *graph, Dist_t *dists,
                                       int n_internal_neighbors, int *locks,
                                       int *dirty_flags, int n_neighbors) {
  const int lane_id = threadIdx.x % raft::warp_size();
  const size_t global_idx_base =
      list_id * static_cast<size_t>(n_internal_neighbors);

  if (elem.id() == static_cast<Index_t>(list_id)) {
    return;
  }

  int lock_acquired = 0;
  do {
    if (lane_id == 0) {
      lock_acquired = (atomicCAS(&locks[list_id], 0, 1) == 0);
    }
    lock_acquired = __shfl_sync(raft::warp_full_mask(), lock_acquired, 0);
  } while (!lock_acquired);

  int local_insert_pos = n_internal_neighbors;
  bool local_duplicate = false;

  for (int local_idx = lane_id; local_idx < n_internal_neighbors;
       local_idx += raft::warp_size()) {
    ResultItem<Index_t> graph_elem;
    graph_elem.id_with_flag() =
        graph[global_idx_base + local_idx].id_with_flag();
    graph_elem.dist() = static_cast<float>(dists[global_idx_base + local_idx]);

    if (elem == graph_elem) {
      local_duplicate = true;
    }
    if (elem < graph_elem && local_insert_pos == n_internal_neighbors) {
      local_insert_pos = local_idx;
    }
  }

  int insert_pos = local_insert_pos;
  for (int offset = raft::warp_size() / 2; offset > 0; offset >>= 1) {
    int other_pos =
        __shfl_down_sync(raft::warp_full_mask(), insert_pos, offset);
    insert_pos = min(insert_pos, other_pos);
  }
  insert_pos = __shfl_sync(raft::warp_full_mask(), insert_pos, 0);

  bool duplicated = __any_sync(raft::warp_full_mask(), local_duplicate);
  if (!duplicated && insert_pos < n_internal_neighbors && lane_id == 0) {
    for (int i = n_internal_neighbors - 1; i > insert_pos; --i) {
      graph[global_idx_base + i].id_with_flag() =
          graph[global_idx_base + (i - 1)].id_with_flag();
      dists[global_idx_base + i] = dists[global_idx_base + (i - 1)];
    }
    graph[global_idx_base + insert_pos].id_with_flag() = elem.id_with_flag();
    dists[global_idx_base + insert_pos] = static_cast<Dist_t>(elem.dist());
    const int dirty_state = (insert_pos < n_neighbors) ? DIRTY_NEIGHBOR_CHANGED
                                                       : DIRTY_LIST_CHANGED;
    atomicMax(&dirty_flags[list_id], dirty_state);
  }

  __threadfence();
  if (lane_id == 0) {
    atomicExch(&locks[list_id], 0);
  }
}

template <typename Data_t, typename Index_t>
__global__ void
local_join_kernel(const Index_t *graph_new, Index_t *rev_graph_new,
                  const int2 *sizes_new, const Index_t *graph_old,
                  const Index_t *rev_graph_old, const int2 *sizes_old,
                  const int width, const Data_t *data, const int n_features,
                  const int data_block_size, InternalID_t *graph, Data_t *dists,
                  int graph_width, int *locks, int *dirty_flags,
                  int n_output_neighbors) {
  __shared__ int s_list[MAX_NUM_BI_SAMPLES * 2];

  constexpr int APAD = 4;
  constexpr int BPAD = 4;
  constexpr int TILE_COL_WIDTH = 32;

  __shared__ float s_nv[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + APAD];
  __shared__ float s_ov[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + BPAD];
  __shared__ float s_distances[MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES];

  int *s_unique_counter = (int *)&s_ov[0][0];

  if (threadIdx.x == 0) {
    s_unique_counter[0] = 0;
    s_unique_counter[1] = 0;
  }

  Index_t *new_neighbors = s_list;
  Index_t *old_neighbors = s_list + MAX_NUM_BI_SAMPLES;

  size_t list_id = blockIdx.x;
  int2 list_new_size2 = sizes_new[list_id];
  int list_new_size = list_new_size2.x + list_new_size2.y;
  int2 list_old_size2 = sizes_old[list_id];
  int list_old_size = list_old_size2.x + list_old_size2.y;

  if (!list_new_size)
    return;

  int tx = threadIdx.x;

  if (tx < list_new_size2.x) {
    const Index_t nid = graph_new[list_id * width + tx];
    new_neighbors[tx] = is_valid_list_id(nid, gridDim.x)
                            ? nid
                            : std::numeric_limits<Index_t>::max();
  } else if (tx < list_new_size) {
    const Index_t nid =
        rev_graph_new[list_id * width + (tx - list_new_size2.x)];
    new_neighbors[tx] = is_valid_list_id(nid, gridDim.x)
                            ? nid
                            : std::numeric_limits<Index_t>::max();
  }

  if (tx < list_old_size2.x) {
    const Index_t nid = graph_old[list_id * width + tx];
    old_neighbors[tx] = is_valid_list_id(nid, gridDim.x)
                            ? nid
                            : std::numeric_limits<Index_t>::max();
  } else if (tx < list_old_size) {
    const Index_t nid =
        rev_graph_old[list_id * width + (tx - list_old_size2.x)];
    old_neighbors[tx] = is_valid_list_id(nid, gridDim.x)
                            ? nid
                            : std::numeric_limits<Index_t>::max();
  }

  __syncthreads();

  remove_duplicates(new_neighbors, list_new_size2.x,
                    new_neighbors + list_new_size2.x, list_new_size2.y,
                    s_unique_counter[0], 0);

  remove_duplicates(old_neighbors, list_old_size2.x,
                    old_neighbors + list_old_size2.x, list_old_size2.y,
                    s_unique_counter[1], 1);

  __syncthreads();

  list_new_size = list_new_size2.x + s_unique_counter[0];
  list_old_size = list_old_size2.x + s_unique_counter[1];

  int warp_id = threadIdx.x / raft::warp_size();
  int lane_id = threadIdx.x % raft::warp_size();
  constexpr int num_warps = BLOCK_SIZE / raft::warp_size();

  const int n_steps = raft::ceildiv(n_features, TILE_COL_WIDTH);
  int tid = threadIdx.x;

  // Phase 1: new x new
  for (int i = tid; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES;
       i += blockDim.x) {
    s_distances[i] = 0.0f;
  }
  __syncthreads();

  for (int step = 0; step < n_steps; ++step) {
    const int base = step * TILE_COL_WIDTH;
    const int num_load_elems =
        (step == n_steps - 1) ? (n_features - base) : TILE_COL_WIDTH;

#pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES; i++) {
      int idx = i * num_warps + warp_id;
      if (idx < list_new_size) {
        const Index_t nid = new_neighbors[idx];
        if (is_valid_list_id(nid, gridDim.x)) {
          size_t offset =
              static_cast<size_t>(nid) * (size_t)data_block_size + (size_t)base;
          load_vec(s_nv[idx], data + offset, num_load_elems, TILE_COL_WIDTH,
                   lane_id);
        }
      }
    }
    __syncthreads();
    for (int i = threadIdx.x;
         i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
      int r = i / SKEWED_MAX_NUM_BI_SAMPLES;
      int c = i % SKEWED_MAX_NUM_BI_SAMPLES;

      if (r < list_new_size && c < list_new_size &&
          is_valid_list_id(new_neighbors[r], gridDim.x) &&
          is_valid_list_id(new_neighbors[c], gridDim.x)) {
        float acc = 0.0f;
#pragma unroll
        for (int d = 0; d < TILE_COL_WIDTH; ++d) {
          if (d < num_load_elems)
            acc += s_nv[r][d] * s_nv[c][d];
        }
        s_distances[i] += acc;
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES;
       i += blockDim.x) {
    int r = i / SKEWED_MAX_NUM_BI_SAMPLES;
    int c = i % SKEWED_MAX_NUM_BI_SAMPLES;

    if (r < list_new_size && c < list_new_size &&
        is_valid_list_id(new_neighbors[r], gridDim.x) &&
        is_valid_list_id(new_neighbors[c], gridDim.x)) {
      Index_t a = new_neighbors[r];
      Index_t b = new_neighbors[c];

      float dot = s_distances[i];
      float dist2 = data[a * (size_t)data_block_size + n_features] +
                    data[b * (size_t)data_block_size + n_features] - 2.0f * dot;
      s_distances[i] = dist2 < 0.0f ? 0.0f : dist2;
    }
  }
  __syncthreads();

  // Phase 1 insertion: for each new-neighbor (src), try inserting every other
  // new-neighbor (cand) into src's KNN list. All pairwise distances are already
  // computed in s_distances[r * SKEWED_MAX_NUM_BI_SAMPLES + c].
  for (int step = 0; step < raft::ceildiv(list_new_size, num_warps); step++) {
    int idx_in_list = step * num_warps + warp_id;

    if (idx_in_list >= list_new_size)
      continue;

    const Index_t src_id = new_neighbors[idx_in_list];
    if (!is_valid_list_id(src_id, gridDim.x))
      continue;

    for (int c = 0; c < list_new_size; c++) {
      if (c == idx_in_list)
        continue;
      const Index_t cand_id = new_neighbors[c];
      if (!is_valid_list_id(cand_id, gridDim.x))
        continue;
      const float dist =
          s_distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + c];
      ResultItem<Index_t> elem(cand_id, dist);
      insert_to_global_graph(elem, src_id, graph, dists, graph_width, locks,
                             dirty_flags, n_output_neighbors);
    }
  }

  if (!list_old_size)
    return;

  __syncthreads();

  for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES;
       i += blockDim.x) {
    s_distances[i] = 0.0f;
  }
  __syncthreads();

  // Phase 2: new x old
  for (int step = 0; step < n_steps; step++) {
    const int base = step * TILE_COL_WIDTH;
    const int num_load_elems =
        (step == n_steps - 1) ? (n_features - base) : TILE_COL_WIDTH;

#pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
      int idx = i * num_warps + warp_id;
      if (idx < list_new_size) {
        const Index_t nid = new_neighbors[idx];
        if (is_valid_list_id(nid, gridDim.x)) {
          size_t offset =
              static_cast<size_t>(nid) * (size_t)data_block_size + (size_t)base;
          load_vec(s_nv[idx], data + offset, num_load_elems, TILE_COL_WIDTH,
                   lane_id);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
      int idx = i * num_warps + warp_id;
      if (idx < list_old_size) {
        const Index_t nid = old_neighbors[idx];
        if (is_valid_list_id(nid, gridDim.x)) {
          size_t offset =
              static_cast<size_t>(nid) * (size_t)data_block_size + (size_t)base;
          load_vec(s_ov[idx], data + offset, num_load_elems, TILE_COL_WIDTH,
                   lane_id);
        }
      }
    }

    __syncthreads();

    for (int i = threadIdx.x;
         i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
      int r = i / SKEWED_MAX_NUM_BI_SAMPLES;
      int c = i % SKEWED_MAX_NUM_BI_SAMPLES;
      if (r < list_new_size && c < list_old_size &&
          is_valid_list_id(new_neighbors[r], gridDim.x) &&
          is_valid_list_id(old_neighbors[c], gridDim.x)) {
        float acc = 0.0f;
#pragma unroll
        for (int d = 0; d < TILE_COL_WIDTH; ++d) {
          if (d < num_load_elems)
            acc += s_nv[r][d] * s_ov[c][d];
        }
        s_distances[i] += acc;
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES;
       i += blockDim.x) {
    int r = i / SKEWED_MAX_NUM_BI_SAMPLES;
    int c = i % SKEWED_MAX_NUM_BI_SAMPLES;

    if (r < list_new_size && c < list_old_size &&
        is_valid_list_id(new_neighbors[r], gridDim.x) &&
        is_valid_list_id(old_neighbors[c], gridDim.x)) {
      Index_t a = new_neighbors[r];
      Index_t b = old_neighbors[c];

      float dot = s_distances[i];
      float dist2 = data[a * (size_t)data_block_size + n_features] +
                    data[b * (size_t)data_block_size + n_features] - 2.0f * dot;
      s_distances[i] = dist2 < 0.0f ? 0.0f : dist2;
    }
  }
  __syncthreads();

  // Phase 2 new-side: each new-neighbor (src=new[r]) considers every
  // old-neighbor (cand=old[c]) as a candidate.
  // s_distances[r * SKEWED_MAX_NUM_BI_SAMPLES + c] = dist(new[r], old[c]).
  for (int step = 0; step < raft::ceildiv(list_new_size, num_warps); step++) {
    int r = step * num_warps + warp_id;
    if (r >= list_new_size)
      continue;

    const Index_t src_id = new_neighbors[r];
    if (!is_valid_list_id(src_id, gridDim.x))
      continue;

    for (int c = 0; c < list_old_size; c++) {
      const Index_t cand_id = old_neighbors[c];
      if (!is_valid_list_id(cand_id, gridDim.x))
        continue;
      const float dist = s_distances[r * SKEWED_MAX_NUM_BI_SAMPLES + c];
      ResultItem<Index_t> elem(cand_id, dist);
      insert_to_global_graph(elem, src_id, graph, dists, graph_width, locks,
                             dirty_flags, n_output_neighbors);
    }
  }

  // Phase 2 old-side: each old-neighbor (src=old[c]) considers every
  // new-neighbor (cand=new[r]) as a candidate.
  // Distance is the same matrix, accessed column-wise.
  for (int step = 0; step < raft::ceildiv(list_old_size, num_warps); step++) {
    int c = step * num_warps + warp_id;
    if (c >= list_old_size)
      continue;

    const Index_t src_id = old_neighbors[c];
    if (!is_valid_list_id(src_id, gridDim.x))
      continue;

    for (int r = 0; r < list_new_size; r++) {
      const Index_t cand_id = new_neighbors[r];
      if (!is_valid_list_id(cand_id, gridDim.x))
        continue;
      const float dist = s_distances[r * SKEWED_MAX_NUM_BI_SAMPLES + c];
      ResultItem<Index_t> elem(cand_id, dist);
      insert_to_global_graph(elem, src_id, graph, dists, graph_width, locks,
                             dirty_flags, n_output_neighbors);
    }
  }
}

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
             uint32_t neighbor_dists_block_size,

             uint32_t n_features, uint32_t n_neighbors,
             uint32_t n_inserted_instances)
      : buffers_(buffers), batch_indices_(std::move(batch_indices)),
        input_data(input_data), neighbor_indices(neighbor_indices),
        neighbor_distances(neighbor_distances),
        data_block_size_(data_block_size),
        neighbor_indices_block_size_(neighbor_indices_block_size),
        neighbor_dists_block_size_(neighbor_dists_block_size),
        n_features_(n_features), n_neighbors_(n_neighbors),
        n_inserted_instances_(n_inserted_instances) {
    std::sort(batch_indices_.begin(), batch_indices_.end());
    batch_indices_.erase(
        std::unique(batch_indices_.begin(), batch_indices_.end()),
        batch_indices_.end());
  }

  size_t
  run(cudaStream_t const stream, std::vector<Idx_t> &affected_indices,
      std::vector<std::tuple<Idx_t, Idx_t, float, float, float>> &new_edges);
  void local_join(cudaStream_t const stream);
  void add_reverse_edges(Index_t *graph_ptr, Index_t *h_rev_graph_ptr,
                         Index_t *d_rev_graph_ptr, int2 *list_sizes,
                         cudaStream_t stream);
  void sample_graph(bool sample_new);
  void update_graph(
      std::vector<std::tuple<Idx_t, Idx_t, float, float, float>> &new_edges);
  void sort_lists();

private:
  NNDBuffers<Data_t, Index_t> &buffers_;
  std::vector<Idx_t> batch_indices_;

  Data_t *input_data;
  Idx_t *neighbor_indices;
  Data_t *neighbor_distances;

  uint32_t data_block_size_;
  uint32_t neighbor_indices_block_size_;
  uint32_t neighbor_dists_block_size_;

  uint32_t n_features_;
  uint32_t n_neighbors_;
  uint32_t n_inserted_instances_;
};

template <typename Data_t, typename Idx_t, typename Index_t>
void UpdateAlgo<Data_t, Idx_t, Index_t>::sample_graph(bool sample_new) {
  const size_t n_rows = batch_indices_.size();

  std::fill_n(buffers_.h_graph_old.data_handle(),
              n_inserted_instances_ * NUM_SAMPLES,
              std::numeric_limits<Index_t>::max());
  if (sample_new) {
    std::fill_n(buffers_.h_graph_new.data_handle(),
                n_inserted_instances_ * NUM_SAMPLES,
                std::numeric_limits<Index_t>::max());
  }

#pragma omp parallel for
  for (size_t i = 0; i < n_inserted_instances_; ++i) {
    buffers_.h_list_sizes_old.data_handle()[i].x = 0;
    buffers_.h_list_sizes_old.data_handle()[i].y = 0;
    buffers_.h_list_sizes_new.data_handle()[i].x = 0;
    buffers_.h_list_sizes_new.data_handle()[i].y = 0;
  }

#pragma omp parallel for
  for (size_t i = 0; i < n_rows; ++i) {
    const size_t row = static_cast<size_t>(batch_indices_[i]);

    auto list = buffers_.h_graph.data_handle() + row * n_neighbors_;
    auto list_old = buffers_.h_graph_old.data_handle() + row * NUM_SAMPLES;
    auto list_new = buffers_.h_graph_new.data_handle() + row * NUM_SAMPLES;

    for (size_t j = 0; j < n_neighbors_; ++j) {
      auto neighbor = list[j];
      if (static_cast<size_t>(neighbor.id()) >= n_inserted_instances_)
        continue;

      const Index_t nid = static_cast<Index_t>(neighbor.id());

      if (!neighbor.is_new()) {
        // Skip pairs already evaluated in a previous iteration.
        if (buffers_.bloom_filter.check(row, nid))
          continue;
        if (buffers_.h_list_sizes_old.data_handle()[row].x < NUM_SAMPLES) {
          list_old[buffers_.h_list_sizes_old.data_handle()[row].x++] = nid;
        }
      } else if (sample_new) {
        if (buffers_.h_list_sizes_new.data_handle()[row].x < NUM_SAMPLES) {
          list[j].mark_old();
          list_new[buffers_.h_list_sizes_new.data_handle()[row].x++] = nid;
          // Mark (row, nid) as evaluated so future iterations skip it.
          buffers_.bloom_filter.add(row, nid);
        }
      }

      if (buffers_.h_list_sizes_old.data_handle()[row].x == NUM_SAMPLES &&
          buffers_.h_list_sizes_new.data_handle()[row].x == NUM_SAMPLES) {
        break;
      }
    }
  }
}

template <typename Data_t, typename Idx_t, typename Index_t>
void UpdateAlgo<Data_t, Idx_t, Index_t>::update_graph(
    std::vector<std::tuple<Idx_t, Idx_t, float, float, float>> &new_edges) {
  const InternalID_t *new_neighbors = buffers_.graph_host_buffer_.data_handle();
  const Data_t *new_dists = buffers_.dists_host_buffer_.data_handle();

#pragma omp parallel
  {
    std::vector<std::tuple<Idx_t, Idx_t, float, float, float>> local_edges;

#pragma omp for
    for (size_t i = 0; i < n_inserted_instances_; i++) {
      for (size_t j = 0; j < MAX_NEIGHBORS; j++) {
        auto neighbor = new_neighbors[i * MAX_NEIGHBORS + j];
        auto new_dist = new_dists[i * MAX_NEIGHBORS + j];

        if (new_dist == std::numeric_limits<Data_t>::max())
          break;
        if (neighbor.id() >= n_inserted_instances_ ||
            (size_t)neighbor.id() == i)
          continue;

        auto list = buffers_.h_graph.data_handle() + i * n_neighbors_;
        auto dist_list = buffers_.h_dists.data_handle() + i * n_neighbors_;
        uint32_t insert_pos = insert_to_ordered_list(
            list, dist_list, n_neighbors_, neighbor, new_dist);

        if (insert_pos != n_neighbors_) {
          local_edges.emplace_back(static_cast<Idx_t>(i),
                                   static_cast<Idx_t>(neighbor.id()), 0.f, -1.f,
                                   -1.f);
          buffers_.h_dirty_flags_.data_handle()[i] = DIRTY_NEIGHBOR_CHANGED;
        }
      }
    }

#pragma omp critical
    new_edges.insert(new_edges.end(), local_edges.begin(), local_edges.end());
  }
}

template <typename Data_t, typename Idx_t, typename Index_t>
void UpdateAlgo<Data_t, Idx_t, Index_t>::sort_lists() {
#pragma omp parallel for
  for (size_t i = 0; i < n_inserted_instances_; i++) {
    if (!buffers_.h_dirty_flags_.data_handle()[i]) {
      continue;
    }
    std::vector<std::pair<Data_t, Index_t>> sorted_list;
    sorted_list.reserve(n_neighbors_);

    for (size_t j = 0; j < n_neighbors_; j++) {
      sorted_list.emplace_back(
          buffers_.h_dists.data_handle()[i * n_neighbors_ + j],
          static_cast<Index_t>(
              buffers_.h_graph.data_handle()[i * n_neighbors_ + j]
                  .id_with_flag()));
    }

    std::sort(sorted_list.begin(), sorted_list.end());

    for (size_t j = 0; j < n_neighbors_; j++) {
      buffers_.h_graph.data_handle()[i * n_neighbors_ + j].id_with_flag() =
          static_cast<int>(sorted_list[j].second);
      buffers_.h_dists.data_handle()[i * n_neighbors_ + j] =
          sorted_list[j].first;
    }
    buffers_.h_dirty_flags_.data_handle()[i] = DIRTY_NONE;
  }
}

template <typename Data_t, typename Idx_t, typename Index_t>
void UpdateAlgo<Data_t, Idx_t, Index_t>::local_join(cudaStream_t const stream) {
  local_join_kernel<<<n_inserted_instances_, BLOCK_SIZE, 0, stream>>>(
      buffers_.h_graph_new.data_handle(),
      buffers_.h_rev_graph_new_.data_handle(),
      buffers_.d_list_sizes_new_.data_handle(),
      buffers_.h_graph_old_.data_handle(),
      buffers_.h_rev_graph_old_.data_handle(),
      buffers_.d_list_sizes_old_.data_handle(), NUM_SAMPLES, input_data,
      n_features_, data_block_size_, buffers_.graph_buffer_.data_handle(),
      buffers_.dists_buffer_.data_handle(), MAX_NEIGHBORS,
      buffers_.d_locks_.data_handle(), buffers_.d_dirty_flags_.data_handle(),
      n_neighbors_);
}

template <typename Data_t, typename Idx_t, typename Index_t>
void UpdateAlgo<Data_t, Idx_t, Index_t>::add_reverse_edges(
    Index_t *graph_ptr, Index_t *h_rev_graph_ptr, Index_t *d_rev_graph_ptr,
    int2 *list_sizes, cudaStream_t const stream) {

  cudaMemsetAsync(d_rev_graph_ptr, 0xFF,
                  static_cast<size_t>(n_inserted_instances_) * NUM_SAMPLES *
                      sizeof(Index_t),
                  stream);

  add_rev_edges_kernel<<<n_inserted_instances_, 256, 0, stream>>>(
      graph_ptr, d_rev_graph_ptr, NUM_SAMPLES, list_sizes);

  raft::copy(h_rev_graph_ptr, d_rev_graph_ptr,
             n_inserted_instances_ * NUM_SAMPLES, stream);
}

template <typename Data_t, typename Idx_t, typename Index_t>
size_t UpdateAlgo<Data_t, Idx_t, Index_t>::run(
    cudaStream_t stream, std::vector<Idx_t> &affected_indices,
    std::vector<std::tuple<Idx_t, Idx_t, float, float, float>> &new_edges) {
  static const bool kProfile = std::getenv("GPUMAP_PROFILE_UPDATE") != nullptr;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto lap = [&](const char *label) {
    if (!kProfile)
      return;
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[UPD] %-16s %8.3f ms\n", label,
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    t0 = t1;
  };

  cudaMemsetAsync(buffers_.d_locks_.data_handle(), 0,
                  n_inserted_instances_ * sizeof(int), stream);

  std::vector<int> d_dirty_flags_host(n_inserted_instances_, 0);
  std::vector<Idx_t> neighbor_changed_indices;

  std::fill_n(buffers_.h_dirty_flags_.data_handle(), n_inserted_instances_,
              DIRTY_NONE);

  raft::copy(buffers_.d_dirty_flags_.data_handle(),
             buffers_.h_dirty_flags_.data_handle(), n_inserted_instances_,
             stream);
  lap("init");

  sample_graph(true);
  lap("sample_graph");
  raft::copy(buffers_.d_list_sizes_new_.data_handle(),
             buffers_.h_list_sizes_new.data_handle(), n_inserted_instances_,
             stream);
  raft::copy(buffers_.h_graph_old_.data_handle(),
             buffers_.h_graph_old.data_handle(),
             n_inserted_instances_ * NUM_SAMPLES, stream);
  raft::copy(buffers_.d_list_sizes_old_.data_handle(),
             buffers_.h_list_sizes_old.data_handle(), n_inserted_instances_,
             stream);

  cudaStreamSynchronize(stream);
  lap("copy_sizes");

  add_reverse_edges(buffers_.h_graph_new.data_handle(),
                    buffers_.h_rev_graph_new_.data_handle(),
                    buffers_.d_rev_graph_buffer_.data_handle(),
                    buffers_.d_list_sizes_new_.data_handle(), stream);
  add_reverse_edges(buffers_.h_graph_old_.data_handle(),
                    buffers_.h_rev_graph_old_.data_handle(),
                    buffers_.d_rev_graph_buffer_.data_handle(),
                    buffers_.d_list_sizes_old_.data_handle(), stream);
  lap("reverse_edges");

  local_join(stream);
  lap("local_join");

  raft::copy(buffers_.graph_host_buffer_.data_handle(),
             buffers_.graph_buffer_.data_handle(),
             n_inserted_instances_ * MAX_NEIGHBORS, stream);
  raft::copy(buffers_.dists_host_buffer_.data_handle(),
             buffers_.dists_buffer_.data_handle(),
             n_inserted_instances_ * MAX_NEIGHBORS, stream);
  cudaStreamSynchronize(stream);
  lap("copy_graph_d2h");

  update_graph(new_edges);
  lap("update_graph");
  raft::copy(d_dirty_flags_host.data(), buffers_.d_dirty_flags_.data_handle(),
             n_inserted_instances_, stream);
  cudaStreamSynchronize(stream);
  lap("copy_dirty");

  for (size_t i = 0; i < n_inserted_instances_; ++i) {
    if (d_dirty_flags_host[i] > buffers_.h_dirty_flags_.data_handle()[i]) {
      buffers_.h_dirty_flags_.data_handle()[i] = d_dirty_flags_host[i];
    }
    if (buffers_.h_dirty_flags_.data_handle()[i] == DIRTY_NEIGHBOR_CHANGED) {
      neighbor_changed_indices.push_back(static_cast<Idx_t>(i));
    }
  }
  lap("scan_dirty");

  sort_lists();
  lap("sort_lists");
  if (kProfile) {
    fprintf(stderr, "[UPD] batch=%zu changed=%zu n=%u\n",
            batch_indices_.size(), neighbor_changed_indices.size(),
            n_inserted_instances_);
  }

  if (neighbor_indices != nullptr && neighbor_distances != nullptr) {
    // scatter_rows_host()/the sync_*_scratch_ buffers below are fixed at
    // uint32_t; NNDBuffers has no Idx_t template parameter of its own, and
    // every instantiation in this codebase uses Idx_t = uint32_t.
    static_assert(std::is_same<Idx_t, uint32_t>::value,
                 "sync_knn_rows scatter path assumes Idx_t == uint32_t");
    const size_t n_changed = neighbor_changed_indices.size();
    // Stage every changed row's (indices, dists) contiguously on the host,
    // then scatter them to their (arbitrary, non-contiguous) device rows in
    // one kernel launch each, instead of issuing 2-3 tiny H2D copies per row.
    std::vector<uint32_t> stg_row_ids(n_changed);
    std::vector<uint32_t> stg_idx(n_changed * n_neighbors_);
    std::vector<Data_t> stg_dist(n_changed * n_neighbors_);

#pragma omp parallel for
    for (size_t k = 0; k < n_changed; ++k) {
      const Idx_t row_id = neighbor_changed_indices[k];
      const size_t row = static_cast<size_t>(row_id);
      auto row_graph = buffers_.h_graph.data_handle() + row * n_neighbors_;
      auto row_dists = buffers_.h_dists.data_handle() + row * n_neighbors_;

      stg_row_ids[k] = static_cast<uint32_t>(row_id);
      uint32_t *dst_idx = stg_idx.data() + k * n_neighbors_;
      Data_t *dst_dist = stg_dist.data() + k * n_neighbors_;
      for (size_t j = 0; j < n_neighbors_; ++j) {
        const int id = row_graph[j].id();
        const bool valid_id =
            (id >= 0 && static_cast<size_t>(id) < n_inserted_instances_) &&
            (static_cast<size_t>(id) != row);

        dst_idx[j] = valid_id ? static_cast<uint32_t>(id)
                             : std::numeric_limits<uint32_t>::max();
        dst_dist[j] =
            valid_id ? row_dists[j] : std::numeric_limits<Data_t>::max();
      }
    }

    utils::scatter_rows_host<uint32_t, uint32_t>(
        reinterpret_cast<uint32_t *>(neighbor_indices),
        neighbor_indices_block_size_, stg_row_ids.data(), stg_idx.data(),
        n_neighbors_, static_cast<uint32_t>(n_changed),
        buffers_.sync_row_ids_scratch_.data_handle(),
        buffers_.sync_idx_scratch_.data_handle(),
        rmm::cuda_stream_view(stream));
    utils::scatter_rows_host<Data_t, uint32_t>(
        neighbor_distances, neighbor_dists_block_size_, stg_row_ids.data(),
        stg_dist.data(), n_neighbors_, static_cast<uint32_t>(n_changed),
        buffers_.sync_row_ids_scratch_.data_handle(),
        buffers_.sync_dist_scratch_.data_handle(),
        rmm::cuda_stream_view(stream));

    // valid_row (any real, non-self neighbor survived filtering above) is
    // exactly "the sentinel wasn't written to every slot of the row".
    affected_indices.reserve(affected_indices.size() + n_changed);
    std::vector<uint32_t> stg_dirty_row_ids;
    stg_dirty_row_ids.reserve(n_changed);
    for (size_t k = 0; k < n_changed; ++k) {
      const uint32_t *row_idx = stg_idx.data() + k * n_neighbors_;
      bool valid_row = false;
      for (size_t j = 0; j < n_neighbors_; ++j) {
        if (row_idx[j] != std::numeric_limits<uint32_t>::max()) {
          valid_row = true;
          break;
        }
      }
      if (valid_row) {
        stg_dirty_row_ids.push_back(stg_row_ids[k]);
        affected_indices.push_back(neighbor_changed_indices[k]);
      }
    }

    const std::vector<uint32_t> stg_dirty_ones(stg_dirty_row_ids.size(), 1u);
    utils::scatter_rows_host<uint32_t, uint32_t>(
        reinterpret_cast<uint32_t *>(neighbor_indices) + n_neighbors_,
        neighbor_indices_block_size_, stg_dirty_row_ids.data(),
        stg_dirty_ones.data(), 1, static_cast<uint32_t>(stg_dirty_row_ids.size()),
        buffers_.sync_row_ids_scratch_.data_handle(),
        buffers_.sync_idx_scratch_.data_handle(),
        rmm::cuda_stream_view(stream));

    cudaStreamSynchronize(stream);
  }
  lap("sync_knn_rows");

  return static_cast<size_t>(new_edges.size());
}

template <typename Data_t, typename Idx_t, typename Index_t = int>
size_t update_kernel(
    NNDBuffers<Data_t, Index_t> &buffers, std::vector<Idx_t> batch_indices,
    std::vector<Idx_t> &affected_indices,
    std::vector<std::tuple<Idx_t, Idx_t, float, float, float>> &new_edges,
    cudaStream_t update_stream,

    Data_t *input_data, Idx_t *neighbor_indices, Data_t *neighbor_distances,
    uint32_t data_block_size, uint32_t neighbor_block_size,
    uint32_t neighbor_dists_block_size,

    uint32_t n_features, uint32_t n_neighbors, size_t n_inserted_instances) {
  if (batch_indices.empty()) {
    return 0;
  }

  UpdateAlgo<Data_t, Idx_t, Index_t> updater(
      buffers, std::move(batch_indices), input_data, neighbor_indices,
      neighbor_distances, data_block_size, neighbor_block_size,
      neighbor_dists_block_size, n_features, n_neighbors, n_inserted_instances);

  return updater.run(update_stream, affected_indices, new_edges);
}

} // namespace gpumap
