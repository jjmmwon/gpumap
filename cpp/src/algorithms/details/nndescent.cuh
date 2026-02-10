
#pragma once
#include <cstddef>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

namespace gpumap::detail::nndescent {

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
__global__ void calculate_l2_norms_sqr_kernel(
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
// TODO:
//  새로 들어온 점은 이웃의 이웃에 대해서만 반복 탐색
//  새로 들어온 점의 이웃은 새로운 점이 이웃인지 확인 및 update queue에 추가
//  ???: 새로운 점들 간의 이웃 탐색은 어떻게 하지?

} // namespace gpumap::detail::nndescent
