#pragma once

// #include <__clang_cuda_runtime_wrapper.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace gpumap {

__device__ __forceinline__ float atomicMaxFloat(float *addr, float val) {
  int *addr_as_int = reinterpret_cast<int *>(addr);
  int old = *addr_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_int, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

template <typename Idx_t>
__device__ __forceinline__ bool is_neighbor(const Idx_t *neighbors,
                                            size_t n_neighbors, Idx_t dst) {
  for (int i = 0; i < (int)n_neighbors; ++i) {
    if (neighbors[i] == dst) {
      return true;
    }
  }
  return false;
}

template <typename Data_t>
__device__ __forceinline__ Data_t clip(Data_t value) {
  if (value > static_cast<Data_t>(4.0)) {
    return static_cast<Data_t>(4.0);
  } else if (value < static_cast<Data_t>(-4.0)) {
    return static_cast<Data_t>(-4.0);
  } else {
    return value;
  }
}

template <typename Idx_t, typename Data_t>
__device__ __forceinline__ Data_t find_neighbor_dist(const Idx_t *neighbors,
                                                     const Data_t *dists,
                                                     size_t n_neighbors,
                                                     Idx_t dst, bool *found) {
  for (size_t i = 0; i < n_neighbors; ++i) {
    if (neighbors[i] == dst) {
      *found = true;
      return dists[i];
    }
  }
  *found = false;
  return Data_t{0};
}

template <typename Data_t>
__device__ __forceinline__ Data_t
smooth_knn_sigma(const Data_t *dists, size_t n_neighbors, Data_t rho,
                 float local_connectivity = 1.0f) {

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
    Idx_t *__restrict__ out_srcs, Idx_t *__restrict__ out_dsts, size_t n_edges,
    const Idx_t *__restrict__ neighbor_indices,
    size_t neighbor_indices_block_size, size_t n_neighbors,
    unsigned long long *__restrict__ out_count) {

  const size_t tid = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n_edges) {
    return;
  }

  const Idx_t src = srcs[tid];
  const Idx_t dst = dsts[tid];
  const Idx_t *my_neighbors =
      neighbor_indices + src * neighbor_indices_block_size;

  if (!is_neighbor(my_neighbors, n_neighbors, dst)) {
    return;
  }

  const unsigned long long out_idx = atomicAdd(out_count, 1ULL);
  out_srcs[out_idx] = src;
  out_dsts[out_idx] = dst;
}

template <typename Idx_t>
void filter_edges_by_neighbors(const Idx_t *srcs, const Idx_t *dsts,
                               Idx_t *out_srcs, Idx_t *out_dsts, size_t n_edges,
                               const Idx_t *neighbor_indices,
                               size_t neighbor_indices_block_size,
                               size_t n_neighbors,
                               unsigned long long *out_count,
                               cudaStream_t stream) {

  const int threads_per_block = 256;
  const int blocks =
      static_cast<int>((n_edges + threads_per_block - 1) / threads_per_block);

  cudaMemsetAsync(out_count, 0, sizeof(unsigned long long), stream);

  filter_edges_by_neighbors_kernel<Idx_t>
      <<<blocks, threads_per_block, 0, stream>>>(
          srcs, dsts, out_srcs, out_dsts, n_edges, neighbor_indices,
          neighbor_indices_block_size, n_neighbors, out_count);
}

template <typename Idx_t, typename Payload_t>
__global__ void filter_edges_by_neighbors_kernel(
    const Idx_t *__restrict__ srcs, const Idx_t *__restrict__ dsts,
    const Payload_t *__restrict__ payloads, Idx_t *__restrict__ out_srcs,
    Idx_t *__restrict__ out_dsts, Payload_t *__restrict__ out_payloads,
    size_t n_edges, const Idx_t *__restrict__ neighbor_indices,
    size_t neighbor_indices_block_size, size_t n_neighbors,
    unsigned long long *__restrict__ out_count) {

  const size_t tid = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n_edges) {
    return;
  }

  const Idx_t src = srcs[tid];
  const Idx_t dst = dsts[tid];
  const Idx_t *my_neighbors =
      neighbor_indices + src * neighbor_indices_block_size;

  if (!is_neighbor(my_neighbors, n_neighbors, dst)) {
    return;
  }

  const unsigned long long out_idx = atomicAdd(out_count, 1ULL);
  out_srcs[out_idx] = src;
  out_dsts[out_idx] = dst;
  out_payloads[out_idx] = payloads[tid];
}

template <typename Idx_t, typename Payload_t>
void filter_edges_by_neighbors(
    const Idx_t *srcs, const Idx_t *dsts, const Payload_t *payloads,
    Idx_t *out_srcs, Idx_t *out_dsts, Payload_t *out_payloads, size_t n_edges,
    const Idx_t *neighbor_indices, size_t neighbor_indices_block_size,
    size_t n_neighbors, unsigned long long *out_count, cudaStream_t stream) {

  const int threads_per_block = 256;
  const int blocks =
      static_cast<int>((n_edges + threads_per_block - 1) / threads_per_block);

  cudaMemsetAsync(out_count, 0, sizeof(unsigned long long), stream);

  filter_edges_by_neighbors_kernel<Idx_t, Payload_t>
      <<<blocks, threads_per_block, 0, stream>>>(
          srcs, dsts, payloads, out_srcs, out_dsts, out_payloads, n_edges,
          neighbor_indices, neighbor_indices_block_size, n_neighbors,
          out_count);
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
                            size_t n_neighbors, cudaStream_t stream) {

  const int threads_per_block = 256;
  const int blocks =
      static_cast<int>((n_indices + threads_per_block - 1) / threads_per_block);

  update_sigma_for_dirty_kernel<Data_t, Idx_t>
      <<<blocks, threads_per_block, 0, stream>>>(
          indices, n_indices, neighbor_indices, neighbor_indices_block_size,
          neighbor_dists, neighbor_dists_block_size, n_neighbors);
}

template <typename Data_t, typename Idx_t>
__global__ void update_embeddings_kernel(
    const Idx_t *__restrict__ srcs, const Idx_t *__restrict__ dsts,
    float *__restrict__ current_epochs, float *__restrict__ next_sample_epochs,
    float *__restrict__ next_neg_epochs, size_t n_edges,
    Data_t *__restrict__ embedding, int n_components,
    size_t n_inserted_instances, const Idx_t *__restrict__ neighbor_indices,
    size_t neighbor_indices_block_size,
    const Data_t *__restrict__ neighbor_dists, size_t neighbor_dists_block_size,
    size_t n_neighbors, float *max_w, float a, float b, float alpha,
    size_t max_epoch, size_t negative_sample_rate, unsigned long long seed) {

  const size_t tid = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n_edges) {
    return;
  }

  const Idx_t src = srcs[tid];
  const Idx_t dst = dsts[tid];

  // TODO: skip if epoch < next_sample_epochs[tid] (UMAP-style sampling)
  if (src >= n_inserted_instances || dst >= n_inserted_instances) {
    return;
  }
  const float current_epoch = current_epochs[tid];
  const float next_sample_epoch = next_sample_epochs[tid];
  const float next_neg_epoch = next_neg_epochs[tid];

  current_epochs[tid] = current_epoch + 2.0f;

  if (current_epoch < next_sample_epoch) {
    return;
  }

  const Idx_t *src_neighbors =
      neighbor_indices + src * neighbor_indices_block_size;
  const Data_t *src_dists = neighbor_dists + src * neighbor_dists_block_size;
  const Idx_t *dst_neighbors =
      neighbor_indices + dst * neighbor_indices_block_size;
  const Data_t *dst_dists = neighbor_dists + dst * neighbor_dists_block_size;

  bool found_ij = false;
  bool found_ji = false;
  const Data_t d_ij =
      find_neighbor_dist(src_neighbors, src_dists, n_neighbors, dst, &found_ij);
  const Data_t d_ji =
      find_neighbor_dist(dst_neighbors, dst_dists, n_neighbors, src, &found_ji);
  if (!found_ij) {
    current_epochs[tid] = max_epoch;
    return;
  }

  // TODO: weight 계산을 update sigma할때 한번에 하고 재사용하는 방식을 할지
  // 고려
  const Data_t rho_i = src_dists[0];
  const Data_t rho_j = dst_dists[0];
  const Data_t sigma_i = src_dists[neighbor_dists_block_size - 1];
  const Data_t sigma_j = dst_dists[neighbor_dists_block_size - 1];
  const Data_t denom_i = sigma_i > Data_t{1e-6f} ? sigma_i : Data_t{1e-6f};
  const Data_t denom_j = sigma_j > Data_t{1e-6f} ? sigma_j : Data_t{1e-6f};
  const Data_t p_ij = expf(-static_cast<float>(
      fmaxf(0.0f, static_cast<float>(d_ij - rho_i)) / denom_i));
  const Data_t p_ji =
      found_ji ? expf(-static_cast<float>(
                     fmaxf(0.0f, static_cast<float>(d_ji - rho_j)) / denom_j))
               : Data_t{0};
  const Data_t w_ij = Data_t{1} - (Data_t{1} - p_ij) * (Data_t{1} - p_ji);

  atomicMaxFloat(max_w, w_ij);

  const float epoch_per_sample =
      *max_w / w_ij + 1e-6f; // add small value to avoid division by zero
  const float epoch_per_negative_sample =
      epoch_per_sample / static_cast<float>(negative_sample_rate);

  if (current_epoch < 1.0f || next_sample_epoch < 0.0f ||
      next_neg_epoch < 0.0f) {
    next_sample_epochs[tid] = epoch_per_sample;
    next_neg_epochs[tid] = epoch_per_negative_sample;
    return;
  }

  const size_t src_offset = static_cast<size_t>(src) * n_components;
  const size_t dst_offset = static_cast<size_t>(dst) * n_components;
  const Data_t dx = embedding[src_offset] - embedding[dst_offset];
  const Data_t dy = embedding[src_offset + 1] - embedding[dst_offset + 1];
  const Data_t r2 = dx * dx + dy * dy + Data_t{1e-6f};
  const Data_t r2b = powf(r2, b);
  const Data_t grad_coeff =
      Data_t{2} * a * b * powf(r2, b - Data_t{1}) / (Data_t{1} + a * r2b);
  const Data_t decayed_alpha =
      (1.0f - current_epoch / static_cast<float>(max_epoch)) * alpha;

  const Data_t grad_dx = clip(grad_coeff * dx) * decayed_alpha;
  const Data_t grad_dy = clip(grad_coeff * dy) * decayed_alpha;

  atomicAdd(&embedding[src_offset], -grad_dx);
  atomicAdd(&embedding[src_offset + 1], -grad_dy);
  atomicAdd(&embedding[dst_offset], grad_dx);
  atomicAdd(&embedding[dst_offset + 1], grad_dy);

  const int n_neg_samples = static_cast<int>((current_epoch - next_neg_epoch) /
                                             epoch_per_negative_sample);

  curandStatePhilox4_32_10_t rng_state;
  curand_init(seed ^ static_cast<unsigned long long>(tid), current_epoch, 0,
              &rng_state);

  for (size_t n = 0; n < n_neg_samples; ++n) {
    Idx_t neg = static_cast<Idx_t>(curand(&rng_state) % n_inserted_instances);
    if (neg == src || neg == dst) {
      neg = (neg + 1) % n_inserted_instances;
    }

    const size_t neg_offset = static_cast<size_t>(neg) * n_components;
    const Data_t ndx = embedding[src_offset] - embedding[neg_offset];
    const Data_t ndy = embedding[src_offset + 1] - embedding[neg_offset + 1];
    const Data_t nr2 = ndx * ndx + ndy * ndy + Data_t{1e-6f};
    const Data_t nr2b = powf(nr2, b);
    const Data_t gneg =
        (Data_t{2} * b) / ((Data_t{1} + a * nr2b) * (nr2 + Data_t{1e-3f}));
    // Linearly increase from 0 to 1 in the first 50% of epochs
    float bypass_factor =
        static_cast<float>(current_epoch) / static_cast<float>(max_epoch) > 0.1f
            ? 1.0f
            : 10.0f * static_cast<float>(current_epoch) /
                  static_cast<float>(max_epoch);

    const Data_t grad_ndx = clip(gneg * ndx) * decayed_alpha * bypass_factor;
    const Data_t grad_ndy = clip(gneg * ndy) * decayed_alpha * bypass_factor;

    atomicAdd(&embedding[src_offset], grad_ndx);
    atomicAdd(&embedding[src_offset + 1], grad_ndy);
    // atomicAdd(&embedding[neg_offset], -grad_ndx);
    // atomicAdd(&embedding[neg_offset + 1], -grad_ndy);
  }

  next_sample_epochs[tid] += epoch_per_sample;
  next_neg_epochs[tid] += epoch_per_negative_sample * n_neg_samples;
}

template <typename Data_t, typename Idx_t>
void update_embeddings(
    const Idx_t *srcs, const Idx_t *dsts, float *current_epochs,
    float *next_sample_epochs, float *next_neg_epochs, size_t n_edges,
    Data_t *embedding, int n_components, size_t n_inserted_instances,
    const Idx_t *neighbor_indices, size_t neighbor_indices_block_size,
    const Data_t *neighbor_dists, size_t neighbor_dists_block_size,
    size_t n_neighbors, float *max_w, float a, float b, float alpha,
    size_t max_epoch, size_t negative_sample_rate, cudaStream_t stream) {

  if (n_edges == 0) {
    return;
  }
  const int threads_per_block = 256;
  const int blocks =
      static_cast<int>((n_edges + threads_per_block - 1) / threads_per_block);
  const unsigned long long seed = 42ULL;
  update_embeddings_kernel<Data_t, Idx_t>
      <<<blocks, threads_per_block, 0, stream>>>(
          srcs, dsts, current_epochs, next_sample_epochs, next_neg_epochs,
          n_edges, embedding, n_components, n_inserted_instances,
          neighbor_indices, neighbor_indices_block_size, neighbor_dists,
          neighbor_dists_block_size, n_neighbors, max_w, a, b, alpha, max_epoch,
          negative_sample_rate, seed);
}

} // namespace gpumap
