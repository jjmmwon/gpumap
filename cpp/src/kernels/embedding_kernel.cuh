#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>

namespace gpumap {

template <typename Idx_t>
__device__ __forceinline__ bool is_neighbor(const Idx_t *neighbors, size_t n_neighbors, Idx_t dst) {
  for (int i = 0; i < n_neighbors; ++i) {
    if (neighbors[i] == dst) {
      return true;
    }
  }
  return false;
}

template <typename Data_t>
__device__ __forceinline__ Data_t smooth_knn_sigma(
    const Data_t *dists, size_t n_neighbors, Data_t rho, float local_connectivity = 1.0f) {

  const float target = log2f(static_cast<float>(n_neighbors));
  Data_t lo = 0.0f;
  Data_t hi = INFINITY;
  Data_t mid = 1.0f;

  for (int it = 0; it < 64; ++it) {
    float sum = 0.0f;
    for (size_t i = 0; i < n_neighbors; ++i) {
      const Data_t d = dists[i] - rho;
      if (d > 0) {
        sum += expf(-static_cast<float>(d) / static_cast<float>(mid));
      } else {
        sum += 1.0f;
      }
    }

    if (sum > target * local_connectivity) {
      hi = mid;
    } else {
      lo = mid;
    }

    if (isfinite(static_cast<float>(hi))) {
      mid = (lo + hi) * 0.5f;
    } else {
      mid *= 2.0f;
    }
  }

  if (mid < 1e-6f) {
    mid = 1e-6f;
  }
  return mid;
}

template <typename Idx_t>
__global__ void filter_edges_by_neighbors_kernel(
    const Idx_t *__restrict__ srcs, const Idx_t *__restrict__ dsts,
    Idx_t *__restrict__ out_srcs, Idx_t *__restrict__ out_dsts,
    size_t n_edges, const Idx_t *__restrict__ neighbor_indices,
    size_t neighbor_indices_block_size, 
        size_t n_neighbors,
        unsigned long long *__restrict__ out_count) {

  const size_t tid = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n_edges) {
    return;
  }

  const Idx_t src = srcs[tid];
  const Idx_t dst = dsts[tid];
  const Idx_t *my_neighbors = neighbor_indices + src * neighbor_indices_block_size;

  if (!is_neighbor(my_neighbors, n_neighbors, dst)) {
    return;
  }

  const unsigned long long out_idx = atomicAdd(out_count, 1ULL);
  out_srcs[out_idx] = src;
  out_dsts[out_idx] = dst;
}

template <typename Idx_t>
void filter_edges_by_neighbors(const Idx_t *srcs, const Idx_t *dsts,
                               Idx_t *out_srcs, Idx_t *out_dsts,
                               size_t n_edges,
                               const Idx_t *neighbor_indices,
                               size_t neighbor_indices_block_size, 
                                                             size_t n_neighbors,
                                                             unsigned long long *out_count, 
                               cudaStream_t stream) {

  const int threads_per_block = 256;
  const int blocks = static_cast<int>((n_edges + threads_per_block - 1) /
                                      threads_per_block);

  cudaMemsetAsync(out_count, 0, sizeof(unsigned long long), stream);

  filter_edges_by_neighbors_kernel<Idx_t>
      <<<blocks, threads_per_block, 0, stream>>>(
          srcs, dsts, out_srcs, out_dsts, n_edges, neighbor_indices,
          neighbor_indices_block_size, n_neighbors, out_count);
}

template <typename Data_t, typename Idx_t>
__global__ void update_sigma_for_dirty_kernel(
    const Idx_t *__restrict__ indices, size_t n_indices,
    Idx_t *__restrict__ neighbor_indices, size_t neighbor_indices_block_size,
    Data_t *__restrict__ neighbor_dists, size_t neighbor_dists_block_size,
    size_t n_neighbors) {

  const size_t tid = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n_indices) {
    return;
  }

  const Idx_t row = indices[tid];
  Idx_t *my_neighbors = neighbor_indices + row * neighbor_indices_block_size;
  Data_t *my_dists = neighbor_dists + row * neighbor_dists_block_size;

  const size_t dirty_idx = neighbor_indices_block_size - 1;
  if (!my_neighbors[dirty_idx]) {
    return;
  }

  const Data_t rho = my_dists[0];
  const Data_t sigma = smooth_knn_sigma(my_dists, n_neighbors, rho);
  my_dists[neighbor_dists_block_size - 1] = sigma;
  my_neighbors[dirty_idx] = 0;
}

template <typename Data_t, typename Idx_t>
void update_sigma_for_dirty(const Idx_t *indices, size_t n_indices,
                            Idx_t *neighbor_indices,
                            size_t neighbor_indices_block_size,
                            Data_t *neighbor_dists,
                            size_t neighbor_dists_block_size,
                            size_t n_neighbors,
                            cudaStream_t stream) {

  const int threads_per_block = 256;
  const int blocks = static_cast<int>((n_indices + threads_per_block - 1) /
                                      threads_per_block);

  update_sigma_for_dirty_kernel<Data_t, Idx_t>
      <<<blocks, threads_per_block, 0, stream>>>(
          indices, n_indices, neighbor_indices, neighbor_indices_block_size,
          neighbor_dists, neighbor_dists_block_size, n_neighbors);
}

} // namespace gpumap
