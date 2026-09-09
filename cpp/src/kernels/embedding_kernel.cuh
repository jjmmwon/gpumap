#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda/std/tuple>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/shuffle.h>

namespace gpumap {

// ─────────────────────────────────────────────────────────────────────────────
// Constants (matching cuML FuzzySimplSet::Naive)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr float SMOOTH_K_TOLERANCE = 1e-5f;
static constexpr float MIN_K_DIST_SCALE = 1e-3f;

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

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

// Squared Euclidean distance between two 2-vectors in registers
__device__ __forceinline__ float rdist2(const float a[2], const float b[2]) {
  float d0 = a[0] - b[0], d1 = a[1] - b[1];
  return d0 * d0 + d1 * d1;
}

__device__ __forceinline__ float fclip(float v) {
  if (!isfinite(v))
    return 0.0f;
  return fmaxf(fminf(v, 4.0f), -4.0f);
}

// Attractive gradient coefficient: -2ab * d2^(b-1) / (a*d2^b + 1)
__device__ __forceinline__ float attr_coeff(float d2, float a, float b) {
  if (d2 <= 0.0f)
    return 0.0f;
  return (-2.0f * a * b * powf(d2, b - 1.0f)) / (a * powf(d2, b) + 1.0f);
}

// Gradient truncation for reproducible float summation.
// Matches cuML truncate_gradient: rounds x to the nearest ULP of rounding.
__device__ __forceinline__ float truncate_gradient(float rounding, float x) {
  return (rounding + x) - rounding;
}

// Repulsive gradient coefficient: 2*gamma*b / ((0.001+d2) * (a*d2^b+1))
__device__ __forceinline__ float rep_coeff(float d2, float gamma, float a,
                                           float b) {
  if (d2 <= 0.0f)
    return 0.0f;
  return (2.0f * gamma * b) / ((0.001f + d2) * (a * powf(d2, b) + 1.0f));
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: global mean of all KNN distances (shared-memory block reduce)
// ─────────────────────────────────────────────────────────────────────────────
// Shared memory layout: [0, tpb) = float sum, [tpb, 2*tpb) = int count
// Total dynamic smem = 2 * tpb * sizeof(float) bytes.
__global__ void compute_mean_dist_kernel(const float *__restrict__ knn_dists,
                                         int dist_stride, int n, int k,
                                         float *__restrict__ sum_out,
                                         int *__restrict__ count_out) {
  extern __shared__ float smem[]; // size = 2 * blockDim.x floats
  float *smem_sum = smem;
  float *smem_cnt =
      smem + blockDim.x; // reinterpret second half as float for cnt

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int total = n * k;

  float v = 0.0f;
  float cnt = 0.0f;
  if (gid < total) {
    int row = gid / k, ki = gid % k;
    float raw = knn_dists[(size_t)row * dist_stride + ki];
    // Skip sentinel (FLT_MAX) and invalid distances
    if (raw >= 0.0f && raw < FLT_MAX / 2.0f) {
      v = raw;
      cnt = 1.0f;
    }
  }

  // Reduce sum
  smem_sum[tid] = v;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s)
      smem_sum[tid] += smem_sum[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(sum_out, smem_sum[0]);

  // Reduce count (stored as float to avoid mixed smem types)
  smem_cnt[tid] = cnt;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s)
      smem_cnt[tid] += smem_cnt[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(count_out, (int)smem_cnt[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: smooth_knn_dist — sigma and rho per point
// Matches cuML FuzzySimplSet::Naive::smooth_knn_dist_kernel (local_conn=1.0)
// Input: strided knn_dists[row * dist_stride + ki]
// ─────────────────────────────────────────────────────────────────────────────
__global__ void smooth_knn_dist_kernel(const float *__restrict__ knn_dists,
                                       int dist_stride, int n, int k,
                                       float mean_dist,
                                       float *__restrict__ sigmas,
                                       float *__restrict__ rhos) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n)
    return;

  const float *d = knn_dists + (size_t)row * dist_stride;
  // d[0] = self (dist 0), d[1] = rho (nearest real neighbor), d[2..k-1] = rest.
  // There are k-1 actual neighbors, so target = log2(k-1).
  // Binary search sums over j=2..k-1 (skipping self at 0 and rho at 1).
  float target = __log2f((float)(k - 1)); // k-1 actual neighbors

  // Pass 1: filter sentinel/invalid distances, find start_nonzero, row mean
  // Sentinel: FLT_MAX (unfilled slot); also guard NaN/Inf and negative.
  int start_nz = -1;
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    float di = d[i];
    if (!isfinite(di) || di < 0.0f || di >= FLT_MAX / 2.0f)
      di = 0.0f;
    sum += di;
    if (di > 0.0f && start_nz == -1)
      start_nz = i;
  }

  if (start_nz == -1) { // all zeros or all sentinel (degenerate)
    rhos[row] = 0.0f;
    sigmas[row] = MIN_K_DIST_SCALE * mean_dist;
    return;
  }

  float row_mean = sum / (float)k;

  // rho = first nonzero distance = d[1] when self is at col 0
  rhos[row] = d[start_nz];

  // Binary search for sigma (j starts from 2: skip self at 0 and rho at 1)
  // Sentinel slots (filtered to 0) contribute 1.0f to psum, treated as
  // imaginary neighbors at distance=rho.
  float lo = 0.0f, hi = FLT_MAX, mid = 1.0f;
  for (int it = 0; it < 64; ++it) {
    float psum = 0.0f;
    for (int j = 2; j < k; ++j) {
      float dj_raw = d[j];
      if (!isfinite(dj_raw) || dj_raw < 0.0f || dj_raw >= FLT_MAX / 2.0f)
        dj_raw = 0.0f;
      float dj = dj_raw - rhos[row];
      psum += (dj > 0.0f) ? __expf(-dj / mid) : 1.0f;
    }
    if (fabsf(psum - target) < SMOOTH_K_TOLERANCE)
      break;
    if (psum > target) {
      hi = mid;
      mid = (lo + hi) * 0.5f;
    } else {
      lo = mid;
      mid = (hi == FLT_MAX) ? mid * 2.0f : (lo + hi) * 0.5f;
    }
  }
  // Clamp sigma: reject non-convergence (FLT_MAX) and degenerate values
  if (!isfinite(mid) || mid <= 0.0f)
    mid = 1.0f;
  sigmas[row] = mid;

  // Minimum scale: 1e-3 * row_mean (or global mean if rho==0)
  float min_scale =
      MIN_K_DIST_SCALE * ((rhos[row] > 0.0f) ? row_mean : mean_dist);
  if (sigmas[row] < min_scale)
    sigmas[row] = min_scale;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: directed membership strengths p_ij
// Matches cuML FuzzySimplSet::Naive::compute_membership_strength_kernel
// Output: COO with n*k entries (rows, cols, vals)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Idx_t>
__global__ void compute_membership_strength_kernel(
    const Idx_t *__restrict__ knn_indices, int idx_stride,
    const float *__restrict__ knn_dists, int dist_stride,
    const float *__restrict__ sigmas, const float *__restrict__ rhos,
    const Idx_t *__restrict__ batch_indices, int n, int n_batch, int k,
    int *__restrict__ coo_rows, int *__restrict__ coo_cols,
    float *__restrict__ coo_vals) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_batch * k)
    return;

  int bi = tid / k, ki = tid % k;
  int row = (int)batch_indices[bi]; // indirection through batch
  Idx_t raw_col = knn_indices[(size_t)row * idx_stride + ki];
  bool valid_col = static_cast<size_t>(raw_col) < static_cast<size_t>(n);
  int col = valid_col ? static_cast<int>(raw_col) : row;
  float dist = knn_dists[(size_t)row * dist_stride + ki];

  float val = 0.0f;
  // Treat sentinel distance (FLT_MAX unfilled slot) as absent edge.
  bool valid_dist = isfinite(dist) && dist >= 0.0f && dist < FLT_MAX / 2.0f;
  if (valid_col && col != row && valid_dist) {
    float rho = rhos[row], sigma = sigmas[row];
    if (dist - rho <= 0.0f || sigma == 0.0f)
      val = 1.0f;
    else {
      val = __expf(-((dist - rho) / sigma));
      if (val < FLT_MIN)
        val = FLT_MIN;
    }
  }

  coo_rows[tid] = row;
  coo_cols[tid] = col;
  coo_vals[tid] = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: symmetrize COO + expand with reverse entries for asymmetric edges
//
// For each directed edge (i, j, p_ij):
//   1. Look up p_ji by scanning row j's k-NN.
//   2. Compute symmetrized weight w = p_ij + p_ji - p_ij * p_ji.
//   3. Update coo_vals[tid] = w  (in-place).
//   4. If p_ji == 0 (asymmetric: j did not list i as a neighbor), atomically
//      append the reverse entry (j, i, w) to the extra buffer so that the
//      full set-union graph matches cuML's symmetrized output.
//
// The extra buffer (extra_rows/cols/vals) must have capacity n_batch*k.
// d_extra_count is an atomic counter initialised to 0 before this kernel.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Idx_t>
__global__ void symmetrize_and_expand_kernel(
    const Idx_t *__restrict__ knn_indices, int idx_stride,
    const float *__restrict__ knn_dists, int dist_stride,
    const float *__restrict__ sigmas, const float *__restrict__ rhos,
    int *__restrict__ coo_rows, int *__restrict__ coo_cols,
    float *__restrict__ coo_vals, int *__restrict__ extra_rows,
    int *__restrict__ extra_cols, float *__restrict__ extra_vals,
    int *__restrict__ d_extra_count, int n, int n_batch, int k) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_batch * k)
    return;

  int i = coo_rows[tid];
  int j = coo_cols[tid];
  float p_ij = coo_vals[tid];

  if (i < 0 || i >= n || j < 0 || j >= n) {
    coo_vals[tid] = 0.0f;
    return;
  }

  // Find p_ji: scan row j's neighbors for i
  float p_ji = 0.0f;
  {
    float rho_j = rhos[j], sigma_j = sigmas[j];
    for (int kj = 0; kj < k; ++kj) {
      if ((int)knn_indices[(size_t)j * idx_stride + kj] == i) {
        float d_ji = knn_dists[(size_t)j * dist_stride + kj];
        // Skip sentinel distance (unfilled slot with valid-looking index)
        if (!isfinite(d_ji) || d_ji < 0.0f || d_ji >= FLT_MAX / 2.0f)
          break;
        if (d_ji - rho_j <= 0.0f || sigma_j == 0.0f)
          p_ji = 1.0f;
        else {
          p_ji = __expf(-((d_ji - rho_j) / sigma_j));
          if (p_ji < FLT_MIN)
            p_ji = FLT_MIN;
        }
        break;
      }
    }
  }

  float w = p_ij + p_ji - p_ij * p_ji;
  coo_vals[tid] = w;

  // Asymmetric edge: j did not include i → add reverse (j, i, w)
  if (p_ji == 0.0f) {
    int idx = atomicAdd(d_extra_count, 1);
    extra_rows[idx] = j;
    extra_cols[idx] = i;
    extra_vals[idx] = w;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: block reduce-max (for make_epochs_per_sample)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reduce_max_kernel(const float *__restrict__ vals, int n,
                                  float *__restrict__ max_out) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  float v = 0.0f;
  if (gid < n)
    v = vals[gid];
  smem[tid] = v;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s && smem[tid + s] > smem[tid])
      smem[tid] = smem[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    atomicMaxFloat(max_out, smem[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 6: compact surviving edges into tmp buffers (atomic scatter)
// Used by trim_graph. Keeps only edges with val >= threshold.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void compact_trim_kernel(
    const int *__restrict__ in_rows, const int *__restrict__ in_cols,
    const float *__restrict__ in_vals, int in_nnz, int *__restrict__ out_rows,
    int *__restrict__ out_cols, float *__restrict__ out_vals,
    int *__restrict__ counter, float threshold) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= in_nnz)
    return;
  if (in_vals[tid] >= threshold) {
    int pos = atomicAdd(counter, 1);
    out_rows[pos] = in_rows[tid];
    out_cols[pos] = in_cols[tid];
    out_vals[pos] = in_vals[tid];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher: trim_graph
// Removes edges with weight < max_w / n_epochs (would never be sampled).
// Matches cuML FuzzySimplSet::trim_graph.
// Modifies coo_rows/cols/vals in-place and returns new nnz.
// ─────────────────────────────────────────────────────────────────────────────
int trim_graph(int *coo_rows, int *coo_cols, float *coo_vals, int nnz,
               float max_w, int n_epochs, cudaStream_t stream) {
  if (nnz == 0 || max_w <= 0.0f)
    return nnz;

  const float threshold = max_w / (float)n_epochs;
  const int tpb = 256;
  const int blocks = (nnz + tpb - 1) / tpb;

  rmm::device_uvector<int> tmp_rows(nnz, stream);
  rmm::device_uvector<int> tmp_cols(nnz, stream);
  rmm::device_uvector<float> tmp_vals(nnz, stream);
  rmm::device_scalar<int> d_counter(0, stream);

  compact_trim_kernel<<<blocks, tpb, 0, stream>>>(
      coo_rows, coo_cols, coo_vals, nnz, tmp_rows.data(), tmp_cols.data(),
      tmp_vals.data(), d_counter.data(), threshold);

  int new_nnz = 0;
  cudaMemcpyAsync(&new_nnz, d_counter.data(), sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaMemcpyAsync(coo_rows, tmp_rows.data(), new_nnz * sizeof(int),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(coo_cols, tmp_cols.data(), new_nnz * sizeof(int),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(coo_vals, tmp_vals.data(), new_nnz * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  return new_nnz;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel helper: count edges per head vertex (for rounding factor)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void compute_degrees_kernel(const int *__restrict__ rows, int nnz,
                                       int *__restrict__ degrees) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < nnz)
    atomicAdd(&degrees[rows[tid]], 1);
}

__global__ void reduce_max_int_kernel(const int *__restrict__ vals, int n,
                                      int *__restrict__ max_out) {
  extern __shared__ int smemi[];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int v = 0;
  if (gid < n)
    v = vals[gid];
  smemi[tid] = v;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s && smemi[tid + s] > smemi[tid])
      smemi[tid] = smemi[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    atomicMax(max_out, smemi[0]);
}

// Computes the gradient rounding factor from the COO head array.
// Matches cuML create_gradient_rounding_factor: 2^ceil(log2(max_deg*4*alpha)).
float compute_rounding_factor(const int *coo_rows, int nnz, int n, float alpha,
                              cudaStream_t stream) {
  if (nnz == 0)
    return 1.0f;
  const int tpb = 256;

  rmm::device_uvector<int> degrees(n, stream);
  cudaMemsetAsync(degrees.data(), 0, n * sizeof(int), stream);
  compute_degrees_kernel<<<(nnz + tpb - 1) / tpb, tpb, 0, stream>>>(
      coo_rows, nnz, degrees.data());

  rmm::device_scalar<int> d_max(0, stream);
  reduce_max_int_kernel<<<(n + tpb - 1) / tpb, tpb, tpb * sizeof(int),
                          stream>>>(degrees.data(), n, d_max.data());

  int max_deg = 0;
  cudaMemcpyAsync(&max_deg, d_max.data(), sizeof(int), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  if (max_deg == 0)
    return 1.0f;
  float max_abs = (float)max_deg * 4.0f * std::abs(alpha);
  float delta = max_abs / (1.0f - 2.0f * (float)max_deg * FLT_EPSILON);
  int exp;
  std::frexp(delta, &exp);
  return std::ldexp(1.0f, exp);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 7: make_epochs_per_sample
// Converts weights → epochs_per_sample in-place.
//   eps[e] = n_epochs / (n_epochs * w[e] / max_w)  if w[e] > 0
//          = -1                                      otherwise
// Matches cuML SimplSetEmbed::Algo::make_epochs_per_sample
// ─────────────────────────────────────────────────────────────────────────────
__global__ void make_epochs_per_sample_kernel(float *__restrict__ vals, int nnz,
                                              float max_w, int n_epochs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz)
    return;
  float w = vals[tid];
  float n_samples = (float)n_epochs * (w / max_w);
  vals[tid] = (n_samples > 0.0f) ? ((float)n_epochs / n_samples) : -1.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 7: init epoch state arrays
//   epoch_of_next_sample[e] = eps[e]
//   epoch_of_next_neg[e]    = eps[e] * nsr_inv
// Matches cuML SimplSetEmbed::Algo::optimize_layout initialization
// ─────────────────────────────────────────────────────────────────────────────
__global__ void init_epoch_state_kernel(const float *__restrict__ eps, int nnz,
                                        float nsr_inv,
                                        float *__restrict__ next_sample,
                                        float *__restrict__ next_neg) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz)
    return;
  float e = eps[tid];
  next_sample[tid] = e;
  next_neg[tid] = e * nsr_inv;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 8: SGD epoch — non-deterministic, n_components=2
// Matches cuML optimize_batch_kernel_reg (non-deterministic path,
// move_other=true).
//
// Parameters that come from UMAP class members are passed as args:
//   a, b            — from UMAPParams.a / .b
//   gamma           — UMAPParams.repulsion_strength (default 1.0)
//   nsr_inv         — 1.0 / UMAPParams.negative_sample_rate
//   alpha           — current learning rate (decayed externally)
//   seed            — incremented externally each epoch
// ─────────────────────────────────────────────────────────────────────────────
__global__ void
optimize_batch_kernel(float *__restrict__ embedding, // (n, 2) row-major
                      const int *__restrict__ head,
                      const int *__restrict__ tail, int nnz, int n,
                      const float *__restrict__ epochs_per_sample,
                      float *__restrict__ epoch_of_next_sample,
                      float *__restrict__ epoch_of_next_neg, float alpha,
                      int epoch, float gamma, float a, float b, float nsr_inv,
                      uint64_t seed, float rounding, int n_epochs) {
  size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t skip = (size_t)gridDim.x * blockDim.x;

  for (; row < (size_t)nnz; row += skip) {
    float next_sample = epoch_of_next_sample[row];
    if (next_sample > (float)epoch)
      continue;

    float eps = epochs_per_sample[row];
    float eps_neg = eps * nsr_inv;

    int j = head[row];
    int k = tail[row];

    float *cur_ptr = embedding + j * 2;
    float *oth_ptr = embedding + k * 2;

    float cur[2] = {cur_ptr[0], cur_ptr[1]};
    float other[2] = {oth_ptr[0], oth_ptr[1]};

    float d2 = rdist2(cur, other);
    float attr = attr_coeff(d2, a, b);

    // Attractive force: update cur register, apply -grad to tail immediately
    float grads[2];
    for (int d = 0; d < 2; ++d) {
      float g = fclip(attr * (cur[d] - other[d])) * alpha;
      cur[d] += g;
      grads[d] = g;
    }
    atomicAdd(oth_ptr, truncate_gradient(rounding, -grads[0]));
    atomicAdd(oth_ptr + 1, truncate_gradient(rounding, -grads[1]));

    epoch_of_next_sample[row] = next_sample + eps;

    // Negative sampling
    float next_neg = epoch_of_next_neg[row];
    int n_neg = (int)(((float)epoch - next_neg) / eps_neg);
    // a bypass factor that increases from 0 to 1 over the first 20% of emax
    // float bypass_factor = (epoch < 0.2f * (float)n_epochs)
    //                           ? (epoch / (0.2f * (float)n_epochs))
    //                           : 1.0f;
    float bypass_factor =
        1.0f; // no bypass (matches cuML non-deterministic path)

    if (n_neg > 0) {
      curandStatePhilox4_32_10_t rng;
      curand_init(seed, row, 0, &rng);
      for (int p = 0; p < n_neg; ++p) {
        int r = (int)(curand(&rng) % (unsigned int)n);
        if (r == j)
          continue;
        float *neg_ptr = embedding + r * 2;
        float neg[2] = {neg_ptr[0], neg_ptr[1]};
        float d2n = rdist2(cur, neg);
        float rep = rep_coeff(d2n, gamma, a, b);
        for (int d = 0; d < 2; ++d) {
          float g = (rep > 0.0f) ? fclip(rep * (cur[d] - neg[d])) * alpha
                                 : 4.0f * alpha;
          g *= bypass_factor; // apply bypass factor to repulsive gradient
          cur[d] += g;
          grads[d] += g;
        }
      }
      epoch_of_next_neg[row] = next_neg + (float)n_neg * eps_neg;
    }

    // Write accumulated gradient to head
    atomicAdd(cur_ptr, truncate_gradient(rounding, grads[0]));
    atomicAdd(cur_ptr + 1, truncate_gradient(rounding, grads[1]));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: prepend_self_knn_kernel
// Builds embed-time KNN arrays (size n*k) by prepending a self-edge
// (index=row, dist=0) at column 0 and shifting original neighbors to 1..k-1,
// dropping the k-th (last) neighbor so total column count stays k.
// This matches traditional UMAP where knn_indices[i][0]==i, knn_dists[i][0]==0.
// Output stride is k; input stride is src_idx_stride / src_dist_stride.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Idx_t>
__global__ void
prepend_self_knn_kernel(const Idx_t *__restrict__ src_indices,
                        int src_idx_stride, const float *__restrict__ src_dists,
                        int src_dist_stride, Idx_t *__restrict__ dst_indices,
                        float *__restrict__ dst_dists, int n, int k,
                        bool src_dists_are_squared_euclidean) {
  // Each thread handles one output element; output size = n * k
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * k;
  if (tid >= total)
    return;

  int row = tid / k;
  int col = tid % k;

  if (col == 0) {
    dst_indices[(size_t)row * k] = static_cast<Idx_t>(row);
    dst_dists[(size_t)row * k] = 0.0f;
  } else {
    // src col = col - 1; last src neighbor (k-1) is dropped
    int src_col = col - 1;
    dst_indices[(size_t)row * k + col] =
        src_indices[(size_t)row * src_idx_stride + src_col];
    // NND stores squared L2 (dot-product trick: l2_a + l2_b - 2*dot) for
    // Metric::Euclidean; smooth_knn_dist_kernel expects true distances, so
    // convert here at the single point where the internal ranking
    // representation becomes the embedding-facing one.
    float d = src_dists[(size_t)row * src_dist_stride + src_col];
    dst_dists[(size_t)row * k + col] =
        src_dists_are_squared_euclidean ? sqrtf(fmaxf(d, 0.0f)) : d;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher: compute_fuzzy_graph
// Equivalent to ML::UMAP::get_graph with precomputed KNN (self at col 0).
//
// coo_rows / coo_cols / coo_vals must be pre-allocated with 2*n_batch*k
// entries.
//   [0,        n_batch*k)  — directed edges (filled by membership_strength)
//   [n_batch*k, 2*n_batch*k) — reverse entries for asymmetric edges (filled
//   here)
//
// Returns actual nnz = n_batch*k + number of asymmetric reverse entries added.
//
// out_sigmas / out_rhos: optional (default nullptr). If non-null, the
// per-row sigma/rho computed in Step 2 are also copied there (n floats
// each) for a caller to reuse.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Idx_t>
int compute_fuzzy_graph(const Idx_t *knn_indices, int knn_idx_stride,
                        const float *knn_dists, int knn_dist_stride, int n,
                        int k, const Idx_t *batch_indices, int n_batch,
                        int *coo_rows, int *coo_cols, float *coo_vals,
                        cudaStream_t stream, float *out_sigmas = nullptr,
                        float *out_rhos = nullptr) {
  const int tpb = 256;
  const int total = n * k;
  const int blk_total = (total + tpb - 1) / tpb;
  const int blk_n = (n + tpb - 1) / tpb;
  const int blk_batch = (n_batch * k + tpb - 1) / tpb;

  // Step 1: global mean distance (over all n rows, excluding sentinel FLT_MAX)
  rmm::device_scalar<float> d_sum(0.0f, stream);
  rmm::device_scalar<int> d_count(0, stream);
  compute_mean_dist_kernel<<<blk_total, tpb, 2 * tpb * sizeof(float), stream>>>(
      knn_dists, knn_dist_stride, n, k, d_sum.data(), d_count.data());
  float mean_dist = 0.0f;
  int valid_count = 0;
  cudaMemcpyAsync(&mean_dist, d_sum.data(), sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&valid_count, d_count.data(), sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  mean_dist = (valid_count > 0 && isfinite(mean_dist) && mean_dist > 0.0f)
                  ? mean_dist / (float)valid_count
                  : 1.0f;

  // Step 2: sigma and rho for all n rows
  rmm::device_uvector<float> sigmas(n, stream), rhos(n, stream);
  cudaMemsetAsync(sigmas.data(), 0, n * sizeof(float), stream);
  cudaMemsetAsync(rhos.data(), 0, n * sizeof(float), stream);
  smooth_knn_dist_kernel<<<blk_n, tpb, 0, stream>>>(
      knn_dists, knn_dist_stride, n, k, mean_dist, sigmas.data(), rhos.data());

  if (out_sigmas) {
    cudaMemcpyAsync(out_sigmas, sigmas.data(), (size_t)n * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
  }
  if (out_rhos) {
    cudaMemcpyAsync(out_rhos, rhos.data(), (size_t)n * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
  }

  // Step 3: directed membership strengths p_ij → first half of COO
  compute_membership_strength_kernel<Idx_t><<<blk_batch, tpb, 0, stream>>>(
      knn_indices, knn_idx_stride, knn_dists, knn_dist_stride, sigmas.data(),
      rhos.data(), batch_indices, n, n_batch, k, coo_rows, coo_cols, coo_vals);

  // Step 4: symmetrize + expand
  // Updates vals in-place; appends (j,i,w) for asymmetric edges into second
  // half.
  const int n_directed = n_batch * k;
  rmm::device_scalar<int> d_extra_count(0, stream);
  symmetrize_and_expand_kernel<Idx_t><<<blk_batch, tpb, 0, stream>>>(
      knn_indices, knn_idx_stride, knn_dists, knn_dist_stride, sigmas.data(),
      rhos.data(), coo_rows, coo_cols, coo_vals, coo_rows + n_directed,
      coo_cols + n_directed, coo_vals + n_directed, d_extra_count.data(), n,
      n_batch, k);

  int extra_count = 0;
  cudaMemcpyAsync(&extra_count, d_extra_count.data(), sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  return n_directed + extra_count;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher: run_optimize_layout
// Equivalent to ML::UMAP::optimize_layout (non-deterministic, move_other=true).
// coo_vals is consumed (converted to eps_per_sample in-place).
//
// Parameters from UMAP class members passed as args:
//   a, b, gamma, negative_sample_rate, n_epochs, initial_alpha, seed
// ─────────────────────────────────────────────────────────────────────────────
void run_optimize_layout(float *embedding, int n, int *coo_rows, int *coo_cols,
                         float *coo_vals, int nnz, float initial_alpha,
                         float gamma, float a, float b, int n_epochs,
                         int negative_sample_rate, uint64_t seed,
                         cudaStream_t stream) {
  if (nnz <= 0 || n <= 0 || n_epochs <= 0)
    return;

  const int tpb = 256;
  float nsr_inv = 1.0f / (float)negative_sample_rate;

  // Step 1: find max edge weight
  {
    int blk = (nnz + tpb - 1) / tpb;
    rmm::device_scalar<float> d_max(0.0f, stream);
    reduce_max_kernel<<<blk, tpb, tpb * sizeof(float), stream>>>(coo_vals, nnz,
                                                                 d_max.data());
    float max_w = 0.0f;
    cudaMemcpyAsync(&max_w, d_max.data(), sizeof(float), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    if (!(max_w > 0.0f)) {
      return;
    }

    // Step 2: trim low-weight edges (matches cuML trim_graph before optimize)
    // Removes edges where n_epochs * w / max_w < 1 (never sampled).
    nnz =
        trim_graph(coo_rows, coo_cols, coo_vals, nnz, max_w, n_epochs, stream);

    if (nnz <= 0) {
      return;
    }

    // Step 3: coo_vals → epochs_per_sample (in-place)
    int blk2 = (nnz + tpb - 1) / tpb;
    make_epochs_per_sample_kernel<<<blk2, tpb, 0, stream>>>(coo_vals, nnz,
                                                            max_w, n_epochs);
  }
  float *eps = coo_vals; // semantic alias

  // Step 5: compute gradient rounding factor (matches cuML
  // create_gradient_rounding_factor)
  float rounding =
      compute_rounding_factor(coo_rows, nnz, n, initial_alpha, stream);

  // Step 6: init epoch state (matches cuML optimize_layout initialization)
  int blk_nnz = (nnz + tpb - 1) / tpb;
  rmm::device_uvector<float> next_sample(nnz, stream);
  rmm::device_uvector<float> next_neg(nnz, stream);
  init_epoch_state_kernel<<<blk_nnz, tpb, 0, stream>>>(
      eps, nnz, nsr_inv, next_sample.data(), next_neg.data());

  // Step 7: SGD loop — alpha decay + seed increment match cuML
  //   optimization_iteration_finalization: alpha = initial_alpha*(1 -
  //   n/n_epochs), seed++
  float alpha = initial_alpha;
  uint64_t cur_seed = seed;
  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    optimize_batch_kernel<<<blk_nnz, tpb, 0, stream>>>(
        embedding, coo_rows, coo_cols, nnz, n, eps, next_sample.data(),
        next_neg.data(), alpha, epoch, gamma, a, b, nsr_inv, cur_seed, rounding,
        n_epochs);
    // finalize: update alpha and seed for next epoch (matches cuML)
    alpha = initial_alpha * (1.0f - (float)epoch / (float)n_epochs);
    cur_seed += 1;
  }

  // Sync before returning so that next_sample/next_neg (local uvectors) are
  // not freed while the last optimize_batch_kernel is still running.
  cudaStreamSynchronize(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: heuristic_neighbor_avg_kernel
//
// Repositions rows NOT selected this round to the fuzzy-membership-weighted
// average of their k-NN neighbors' current embedding position, using the
// same v_ij weighting as compute_membership_strength_kernel. Reads from
// `embedding_in` (a pre-launch snapshot) and writes to `embedding_out` so
// concurrently-updated rows never observe each other's new value. Rows with
// no positively-weighted neighbor are left untouched. n_components is
// hardcoded to 2, matching optimize_batch_kernel's layout assumption.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Idx_t>
__global__ void heuristic_neighbor_avg_kernel(
    const Idx_t *__restrict__ knn_indices, int idx_stride,
    const float *__restrict__ knn_dists, int dist_stride,
    const float *__restrict__ sigmas, const float *__restrict__ rhos,
    const Idx_t *__restrict__ update_indices, int n_update, int k,
    const float *__restrict__ embedding_in, float *__restrict__ embedding_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_update)
    return;

  int row = static_cast<int>(update_indices[idx]);
  const float sigma = sigmas[row];
  const float rho = rhos[row];

  float wsum = 0.0f;
  float acc0 = 0.0f, acc1 = 0.0f;

  for (int ki = 0; ki < k; ++ki) {
    Idx_t raw_col = knn_indices[(size_t)row * idx_stride + ki];
    int col = static_cast<int>(raw_col);
    if (col == row)
      continue; // self at knn col 0 (embed-time KNN layout)

    float dist = knn_dists[(size_t)row * dist_stride + ki];
    bool valid_dist = isfinite(dist) && dist >= 0.0f && dist < FLT_MAX / 2.0f;
    if (!valid_dist)
      continue;

    float w;
    if (dist - rho <= 0.0f || sigma == 0.0f) {
      w = 1.0f;
    } else {
      w = __expf(-((dist - rho) / sigma));
      if (w < FLT_MIN)
        w = FLT_MIN;
    }

    wsum += w;
    acc0 += w * embedding_in[(size_t)col * 2 + 0];
    acc1 += w * embedding_in[(size_t)col * 2 + 1];
  }

  if (wsum > 0.0f) {
    embedding_out[(size_t)row * 2 + 0] = acc0 / wsum;
    embedding_out[(size_t)row * 2 + 1] = acc1 / wsum;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher: run_heuristic_neighbor_avg_update
//
// Takes sigma/rho as caller-supplied buffers (e.g. compute_fuzzy_graph's
// out_sigmas/out_rhos) to avoid a duplicate sigma/rho pass. Safe to call
// with n_update == 0 (no-op).
// ─────────────────────────────────────────────────────────────────────────────
template <typename Idx_t>
void run_heuristic_neighbor_avg_update(const Idx_t *knn_indices,
                                       int knn_idx_stride,
                                       const float *knn_dists,
                                       int knn_dist_stride, int n, int k,
                                       const float *sigmas, const float *rhos,
                                       const Idx_t *update_indices,
                                       int n_update, float *embedding,
                                       cudaStream_t stream) {
  if (n_update <= 0 || n <= 0 || k <= 0)
    return;

  const int tpb = 256;
  const int blk_update = (n_update + tpb - 1) / tpb;

  // Snapshot current embedding so rows updated in this launch never read a
  // neighbor's partially-written new value.
  rmm::device_uvector<float> embedding_snapshot((size_t)n * 2, stream);
  cudaMemcpyAsync(embedding_snapshot.data(), embedding,
                  (size_t)n * 2 * sizeof(float), cudaMemcpyDeviceToDevice,
                  stream);

  heuristic_neighbor_avg_kernel<Idx_t><<<blk_update, tpb, 0, stream>>>(
      knn_indices, knn_idx_stride, knn_dists, knn_dist_stride, sigmas, rhos,
      update_indices, n_update, k, embedding_snapshot.data(), embedding);

  cudaStreamSynchronize(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: destripe_data
// Copies strided data (stride = n_features+1) → contiguous (stride =
// n_features) Used to pass storage.data as X to cuML or our own kernels.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Data_t>
__global__ void destripe_data_kernel(const Data_t *__restrict__ src,
                                     size_t src_stride, size_t n_rows,
                                     size_t n_cols, float *__restrict__ dst) {
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_rows * n_cols)
    return;
  size_t row = tid / n_cols, col = tid % n_cols;
  dst[tid] = static_cast<float>(src[row * src_stride + col]);
}

template <typename Data_t>
void destripe_data(const Data_t *src, size_t src_stride, size_t n_rows,
                   size_t n_cols, float *dst, cudaStream_t stream) {
  size_t total = n_rows * n_cols;
  if (total == 0)
    return;
  const int tpb = 256;
  const int blocks = (int)((total + tpb - 1) / tpb);
  destripe_data_kernel<Data_t>
      <<<blocks, tpb, 0, stream>>>(src, src_stride, n_rows, n_cols, dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: convert_indices_strided
// Converts strided uint32_t neighbor_indices → contiguous int64_t knn_indices.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Idx_t>
__global__ void
convert_indices_strided_kernel(const Idx_t *__restrict__ src, size_t block_size,
                               size_t n_nodes, size_t n_neighbors,
                               int64_t *__restrict__ dst) {
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_nodes * n_neighbors)
    return;
  size_t node = tid / n_neighbors, k = tid % n_neighbors;
  dst[tid] = static_cast<int64_t>(src[node * block_size + k]);
}

template <typename Idx_t>
void convert_indices_strided(const Idx_t *src, size_t block_size,
                             size_t n_nodes, size_t n_neighbors, int64_t *dst,
                             cudaStream_t stream) {
  size_t total = n_nodes * n_neighbors;
  if (total == 0)
    return;
  const int tpb = 256;
  const int blocks = (int)((total + tpb - 1) / tpb);
  convert_indices_strided_kernel<Idx_t>
      <<<blocks, tpb, 0, stream>>>(src, block_size, n_nodes, n_neighbors, dst);
}

template <typename Data_t, typename Idx_t>
__global__ void add_embedding_noise_kernel(Data_t *__restrict__ embedding,
                                           const Idx_t *__restrict__ indices,
                                           size_t n_indices, int n_components,
                                           float effective_scale,
                                           unsigned long long seed) {
  const size_t tid = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n_indices * static_cast<size_t>(n_components))
    return;

  const size_t node_local = tid / static_cast<size_t>(n_components);
  const int comp = static_cast<int>(tid % static_cast<size_t>(n_components));
  const size_t node = static_cast<size_t>(indices[node_local]);

  curandStatePhilox4_32_10_t rng;
  curand_init(seed ^ tid, 0, 0, &rng);

  embedding[node * static_cast<size_t>(n_components) + comp] +=
      static_cast<Data_t>(curand_normal(&rng) * effective_scale);
}

// n_total: total rows in embedding (for x-range computation)
template <typename Data_t, typename Idx_t>
void add_embedding_noise(Data_t *embedding, const Idx_t *indices,
                         size_t n_indices, int n_components, int n_total,
                         float noise_scale, unsigned long long seed,
                         cudaStream_t stream) {
  if (n_indices == 0 || noise_scale == 0.0f || n_total == 0)
    return;

  // Compute x-axis range over all n_total rows
  auto emb_ptr = thrust::device_pointer_cast(embedding);
  auto x_iter = thrust::make_permutation_iterator(
      emb_ptr, thrust::make_transform_iterator(
                   thrust::make_counting_iterator<int>(0),
                   [n_components] __host__ __device__(int i) -> int {
                     return i * n_components;
                   }));
  auto mm = thrust::minmax_element(thrust::cuda::par.on(stream), x_iter,
                                   x_iter + n_total);
  cudaStreamSynchronize(stream);
  float x_range = static_cast<float>(*mm.second - *mm.first);

  const float effective_scale = noise_scale * x_range;

  const size_t n_work = n_indices * static_cast<size_t>(n_components);
  const int tpb = 256;
  const int blocks = static_cast<int>((n_work + tpb - 1) / tpb);

  add_embedding_noise_kernel<Data_t, Idx_t><<<blocks, tpb, 0, stream>>>(
      embedding, indices, n_indices, n_components, effective_scale, seed);

  // Sync so that the caller can safely free `indices` after this returns.
  cudaStreamSynchronize(stream);
}

} // namespace gpumap
