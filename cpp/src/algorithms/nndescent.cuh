
#pragma once
#include "storage/matrix2D.cuh"
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

namespace gpumap {

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

__global__ void init_newflags_kernel(uint8_t *is_new, int n, int k) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = n * k;
  if (tid >= total)
    return;
  is_new[tid] = 1;
}

// L2 distance 최적화를 위한 데이터 전처리 커널
template <typename Data_t, typename Idx_t>
__global__ void preprocess_data_kernel(
    const DeviceMatrix2D<Data_t, Idx_t> *input_data, // 입력 데이터 매트릭스
    float *l2_norms,                                 // 출력: 각 벡터의 L2 norm
    size_t n_instances,                              // 인스턴스 개수
    size_t n_features                                // 특성 개수
) {
  extern __shared__ float shared_norms[]; // shared memory for L2 norms

  const size_t tid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t block_size = blockDim.x;

  // 전역 인덱스 계산
  const size_t global_idx = bid * block_size + tid;

  // 각 스레드가 하나의 인스턴스(벡터)를 담당
  if (global_idx < n_instances) {
    float norm_squared = 0.0f;

    // L2 norm 제곱 계산: ||x||^2 = sum(x_i^2)
    for (size_t feature_idx = 0; feature_idx < n_features; ++feature_idx) {
      float value = static_cast<float>((*input_data)(global_idx, feature_idx));
      norm_squared += value * value;
    }

    // L2 norm 계산 (제곱근)
    float norm = sqrtf(norm_squared);

    // 전역 메모리에 저장
    l2_norms[global_idx] = norm;

    // shared memory에도 저장 (블록 내 스레드들이 공유)
    shared_norms[tid] = norm;
  } else {
    // 범위를 벗어난 스레드는 0으로 초기화
    if (tid < block_size) {
      shared_norms[tid] = 0.0f;
    }
  }

  // shared memory 동기화
  __syncthreads();

  // 추가 처리가 필요한 경우 여기에 작성
  // 예: 블록 내의 norm 값들을 사용한 추가 계산
}

// L2 distance 계산을 위한 최적화된 거리 계산 커널
template <typename Data_t, typename Idx_t>
__device__ __forceinline__ float
compute_l2_distance_optimized(const DeviceMatrix2D<Data_t, Idx_t> *input_data,
                              const float *l2_norms, size_t idx_a, size_t idx_b,
                              size_t n_features) {
  // L2 distance = sqrt(||a||^2 + ||b||^2 - 2*dot(a,b))
  float norm_a_sq = l2_norms[idx_a] * l2_norms[idx_a];
  float norm_b_sq = l2_norms[idx_b] * l2_norms[idx_b];

  // dot product 계산
  float dot_product = 0.0f;
  for (size_t i = 0; i < n_features; ++i) {
    float a_val = static_cast<float>((*input_data)(idx_a, i));
    float b_val = static_cast<float>((*input_data)(idx_b, i));
    dot_product += a_val * b_val;
  }

  // L2 distance 계산
  float distance_sq = norm_a_sq + norm_b_sq - 2.0f * dot_product;

  // 부동소수점 오차로 인한 음수 방지
  distance_sq = fmaxf(distance_sq, 0.0f);

  return sqrtf(distance_sq);
}

} // namespace gpumap