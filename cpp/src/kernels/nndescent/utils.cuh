#pragma once

#include <cstddef>
#include <limits>

namespace gpumap::nndescent::utils {

template <typename Idx_t>
__device__ bool list_contains_id(const Idx_t *arr, int k, Idx_t v) {
  for (int i = 0; i < k; ++i) {
    if (arr[i] == v) {
      return true;
    }
  }
  return false;
}

template <typename Idx_t> __device__ bool is_invalid_id(Idx_t id) {
  return id == std::numeric_limits<Idx_t>::max() || id < 0;
}

// Squared L2 distance using ||a-b||^2 = ||a||^2 + ||b||^2 - 2 * dot(a,b).
// Input layout is (N, D+1), where the last value is precomputed norm^2.
template <typename Data_t, typename Idx_t>
__device__ Data_t compute_l2_distance_sq(const Data_t *__restrict__ data,
                                         Idx_t idx1, Idx_t idx2,
                                         size_t n_features) {
  const size_t offset1 = static_cast<size_t>(idx1) * (n_features + 1);
  const size_t offset2 = static_cast<size_t>(idx2) * (n_features + 1);

  const float norm1 = static_cast<float>(data[offset1 + n_features]);
  const float norm2 = static_cast<float>(data[offset2 + n_features]);

  float dot = 0.0f;
  for (size_t i = 0; i < n_features; ++i) {
    dot += static_cast<float>(data[offset1 + i]) *
           static_cast<float>(data[offset2 + i]);
  }

  float dist_sq = norm1 + norm2 - 2.0f * dot;
  dist_sq = dist_sq < 0.0f ? 0.0f : dist_sq;
  return static_cast<Data_t>(dist_sq);
}

template <typename Data_t, typename Idx_t>
__device__ bool should_insert_neighbor(const Idx_t *neighbors,
                                       const Data_t *distances, int n_neighbors,
                                       Idx_t new_id, Data_t new_dist) {
  if (n_neighbors <= 0 || is_invalid_id(new_id)) {
    return false;
  }

  if (list_contains_id<Idx_t>(neighbors, n_neighbors, new_id)) {
    return false;
  }

  return new_dist < distances[n_neighbors - 1];
}

// Insert into ascending-sorted (distance, id) arrays.
template <typename Data_t, typename Idx_t>
__device__ __host__ bool
insert_neighbor_sorted(Idx_t *neighbors, Data_t *distances, size_t n_neighbors,
                       Idx_t new_id, Data_t new_dist) {
  if (!should_insert_neighbor<Data_t, Idx_t>(neighbors, distances, n_neighbors,
                                             new_id, new_dist)) {
    return false;
  }

  int insert_pos = n_neighbors - 1;
  while (insert_pos > 0 && new_dist < distances[insert_pos - 1]) {
    neighbors[insert_pos] = neighbors[insert_pos - 1];
    distances[insert_pos] = distances[insert_pos - 1];
    --insert_pos;
  }

  if (new_id < 0)
    new_id = -new_id - 1;

  neighbors[insert_pos] = new_id;
  distances[insert_pos] = new_dist;
  return true;
}

// list1: (N, D+1), list2: (M, D+1), dist_mtx: (N, M)
// Last column of list1/list2 is precomputed norm^2.
template <typename Data_t>
__global__ void exact_l2_distance_matrix_from_normhack(
    const Data_t *__restrict__ list1, const Data_t *__restrict__ list2,
    Data_t *__restrict__ dist_mtx, int n_rows1, int n_rows2, int n_features) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n_rows1 || j >= n_rows2)
    return;

  const Data_t *a = list1 + static_cast<size_t>(i) * (n_features + 1);
  const Data_t *b = list2 + static_cast<size_t>(j) * (n_features + 1);

  float dot = 0.0f;
  for (int d = 0; d < n_features; ++d) {
    dot += static_cast<float>(a[d]) * static_cast<float>(b[d]);
  }

  float dist_sq = static_cast<float>(a[n_features]) +
                  static_cast<float>(b[n_features]) - 2.0f * dot;
  dist_sq = dist_sq < 0.0f ? 0.0f : dist_sq;
  dist_mtx[static_cast<size_t>(i) * n_rows2 + j] = static_cast<Data_t>(dist_sq);
}
} // namespace gpumap::nndescent::utils
