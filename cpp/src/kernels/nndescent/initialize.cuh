#pragma once

#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

namespace gpumap::nndescent {

template <typename Idx_t>
__device__ __forceinline__ bool contains_id(const Idx_t *arr, int k, Idx_t v) {
  for (int i = 0; i < k; ++i) {
    if (arr[i] == v) {
      return true;
    }
  }
  return false;
}

template <typename Idx_t>
__global__ void
initialize_random_neighbors_kernel(Idx_t *neighbor_indices, size_t n_instances,
                                   size_t n_neighbors, size_t total_instances,
                                   unsigned long long seed) {

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_instances)
    return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  Idx_t *my_neighbors = neighbor_indices + idx * n_neighbors;

  int filled = 0;
  while (filled < n_neighbors) {
    Idx_t candidate = curand(&state) % total_instances;
    if (candidate == idx)
      continue;
    if (contains_id<Idx_t>(my_neighbors, filled, candidate))
      continue;

    my_neighbors[filled++] = candidate;
  }
}
template <typename Data_t, typename Idx_t>
__global__ void preprocess_l2_norms_kernel(
    Data_t *input_data, // last column is reserved for norms
    size_t n_instances, size_t n_features) {

  const size_t row = static_cast<size_t>(blockIdx.x);

  if (row >= n_instances)
    return;

  const size_t row_offset = row * (n_features + 1);
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

template <typename Data_t, typename Idx_t, size_t MAX_K, size_t MAX_C>
__global__ void
sample_forward_candidates_kernel(const Idx_t *__restrict__ neighbor_indices,
                                 int *__restrict__ neighbor_candidate_indices,
                                 Idx_t n_start, Idx_t n_end, int n_neighbors,
                                 int n_candidates) {

  const Idx_t gid =
      static_cast<Idx_t>(blockIdx.x * blockDim.x + threadIdx.x) + n_start;

  if (gid >= n_end)
    return;

  const Idx_t *__restrict__ my_neighbors = neighbor_indices + gid * n_neighbors;

  int *__restrict__ my_candidates =
      neighbor_candidate_indices + gid * n_candidates;

  Idx_t my_neighbors_l[MAX_K];
#pragma unroll
  for (int i = 0; i < MAX_K; ++i) {
    if (i < n_neighbors) {
      my_neighbors_l[i] = my_neighbors[i];
    } else {
      my_neighbors_l[i] = std::numeric_limits<Idx_t>::max();
    }
  }

  int my_candidates_l[MAX_C];
#pragma unroll
  for (int i = 0; i < MAX_C; ++i)
    my_candidates_l[i] = -1;

  int candidate_count = 0;

  // 각 이웃의 이웃을 후보로
#pragma unroll
  for (size_t j = 0; j < MAX_K; ++j) {
    if (j >= n_neighbors)
      break;
    if (candidate_count >= n_candidates)
      break;

    const Idx_t neighbor_id = my_neighbors_l[j];
    if (is_invalid_id(neighbor_id))
      continue;

    const Idx_t *__restrict__ neigh2 =
        neighbor_indices + static_cast<size_t>(neighbor_id) * n_neighbors;

#pragma unroll
    for (size_t k = 0; k < MAX_K; ++k) {
      if (k >= n_neighbors)
        break;
      if (candidate_count >= n_candidates)
        break;

      const Idx_t candidate_id = neigh2[k];
      if (is_invalid_id(candidate_id))
        continue;
      if (candidate_id == gid)
        continue;
      if (contains_id<Idx_t>(my_neighbors_l, n_neighbors, candidate_id))
        continue;
      if (contains_id<int>(my_candidates_l, candidate_count, candidate_id))
        continue;

      my_candidates_l[candidate_count++] = static_cast<int>(candidate_id);
    }
  }
#pragma unroll
  for (int i = 0; i < MAX_C; ++i) {
    my_candidates[i] = my_candidates_l[i];
  }
}

void iter_forward_nndescent() {}

} // namespace gpumap::nndescent