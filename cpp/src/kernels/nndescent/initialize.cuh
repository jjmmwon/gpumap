#pragma once

#include "utils.cuh"
#include <cstddef>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

namespace gpumap::nndescent {

template <typename Data_t, typename Idx_t, int MAX_K = 32>
__global__ void initialize_random_neighbors_kernel(
    Idx_t *neighbor_indices, Data_t *neighbor_distances, Idx_t n_start,
    Idx_t n_end, Idx_t n_neighbors, Idx_t neighbor_indices_block_size,
    Idx_t neighbor_dists_block_size,
    Idx_t n_instances,
    Idx_t n_inserted_instances, size_t seed) {

  Idx_t idx = blockIdx.x * blockDim.x + threadIdx.x + n_start;
  if (idx >= n_end)
    return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  Idx_t *my_neighbors = neighbor_indices + idx * neighbor_indices_block_size;
  Data_t *my_distances = neighbor_distances + idx * neighbor_dists_block_size;

  int filled = 0;
  while (filled < n_neighbors) {
    Idx_t candidate = curand(&state) % n_inserted_instances;
    if (candidate == idx)
      continue;
    if (contains_id<Idx_t, MAX_K>(my_neighbors, filled, candidate))
      continue;

    my_neighbors[filled] = candidate;
    my_distances[filled++] = std::numeric_limits<Data_t>::max();
  }
}

template <typename Data_t, typename Idx_t>
__global__ void preprocess_l2_norms_kernel(
    Data_t *input_data, // last column is reserved for norms
    size_t n_start, size_t n_end, size_t n_features) {

  const size_t row = static_cast<size_t>(blockIdx.x) + n_start;

  if (row >= n_end)
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
__global__ void sample_forward_candidates_kernel(
    const Idx_t *__restrict__ neighbor_indices,
    int *__restrict__ neighbor_candidate_indices, Idx_t n_start, Idx_t n_end,
    size_t neighbor_indices_block_size, size_t n_neighbors, int n_candidates) {

  const Idx_t gid =
      static_cast<Idx_t>(blockIdx.x * blockDim.x + threadIdx.x) + n_start;

  if (gid >= n_end)
    return;

  const Idx_t *__restrict__ my_neighbors =
      neighbor_indices + gid * neighbor_indices_block_size;

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
        neighbor_indices +
        static_cast<size_t>(neighbor_id) * neighbor_indices_block_size;

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
      if (contains_id<Idx_t, MAX_K>(my_neighbors_l, n_neighbors, candidate_id))
        continue;
      if (contains_id<int, MAX_C>(my_candidates_l, candidate_count,
                                  candidate_id))
        continue;

      my_candidates_l[candidate_count++] = static_cast<int>(candidate_id);
    }
  }
#pragma unroll
  for (int i = 0; i < MAX_C; ++i) {
    my_candidates[i] = my_candidates_l[i];
  }
}

// Forward NNDescent kernel: evaluate candidates and update neighbors
template <typename Data_t, typename Idx_t, size_t MAX_K, size_t MAX_C>
__global__ void forward_nndescent_kernel(
    const Data_t *__restrict__ input_data, Idx_t *__restrict__ neighbor_indices,
    Data_t *__restrict__ neighbor_distances,
    const int *__restrict__ neighbor_candidate_indices, Idx_t n_start,
    Idx_t n_end, size_t neighbor_indices_block_size, size_t n_neighbors,
    int n_candidates, size_t n_features) {

  const Idx_t gid =
      static_cast<Idx_t>(blockIdx.x * blockDim.x + threadIdx.x) + n_start;

  if (gid >= n_end)
    return;

  // Load neighbors to local memory
  Idx_t *my_neighbors = neighbor_indices + gid * neighbor_indices_block_size;
  Data_t *my_distances = neighbor_distances + gid * neighbor_indices_block_size;

  Idx_t neighbors_local[MAX_K];
  Data_t distances_local[MAX_K];

#pragma unroll
  for (int i = 0; i < MAX_K; ++i) {
    if (i < n_neighbors) {
      neighbors_local[i] = my_neighbors[i];
      distances_local[i] = my_distances[i];
    } else {
      neighbors_local[i] = std::numeric_limits<Idx_t>::max();
      distances_local[i] = std::numeric_limits<Data_t>::max();
    }
  }

  // Load candidates
  const int *my_candidates = neighbor_candidate_indices + gid * n_candidates;

  bool updated = false;

  // Evaluate each candidate
#pragma unroll
  for (int i = 0; i < MAX_C; ++i) {
    if (i >= n_candidates)
      break;

    int candidate = my_candidates[i];
    if (candidate < 0)
      continue;

    Idx_t candidate_id = static_cast<Idx_t>(candidate);

    // Compute distance
    Data_t dist = compute_l2_distance<Data_t, Idx_t>(input_data, gid,
                                                     candidate_id, n_features);

    // Try to insert
    if (insert_neighbor_sorted<Data_t, Idx_t>(neighbors_local, distances_local,
                                              n_neighbors, candidate_id,
                                              dist)) {
      updated = true;
    }
  }

  // Write back to global memory if updated
  if (updated) {
#pragma unroll
    for (int i = 0; i < MAX_K; ++i) {
      if (i < n_neighbors) {
        my_neighbors[i] = neighbors_local[i];
        my_distances[i] = distances_local[i];
      }
    }
  }
}

// Host function to run forward NNDescent for n_iter iterations
template <typename Data_t, typename Idx_t, size_t MAX_K = 64,
          size_t MAX_C = 128>
void iter_forward_nndescent(
    const Data_t *__restrict__ input_data, Idx_t *__restrict__ neighbor_indices,
    Data_t *__restrict__ neighbor_distances,
    int *__restrict__ neighbor_candidate_indices, Idx_t n_start, Idx_t n_end,
    size_t neighbor_indices_block_size, size_t n_neighbors, int n_candidates,
    size_t n_features, size_t n_iter, cudaStream_t stream) {

  const int threads_per_block = 256;
  const Idx_t n_instances = n_end - n_start;
  const int blocks = (n_instances + threads_per_block - 1) / threads_per_block;

  for (size_t iter = 0; iter < n_iter; ++iter) {
    // Sample forward candidates
    sample_forward_candidates_kernel<Data_t, Idx_t, MAX_K, MAX_C>
        <<<blocks, threads_per_block, 0, stream>>>(
            neighbor_indices, neighbor_candidate_indices, n_start, n_end,
            neighbor_indices_block_size, n_neighbors, n_candidates);

    // Evaluate candidates and update neighbors
    forward_nndescent_kernel<Data_t, Idx_t, MAX_K, MAX_C>
        <<<blocks, threads_per_block, 0, stream>>>(
            input_data, neighbor_indices, neighbor_distances,
            neighbor_candidate_indices, n_start, n_end,
            neighbor_indices_block_size, n_neighbors, n_candidates, n_features);
  }
}

} // namespace gpumap::nndescent