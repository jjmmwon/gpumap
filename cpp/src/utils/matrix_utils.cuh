#pragma once

#include "raft/util/cuda_rt_essentials.hpp"
#include "raft/util/cudart_utils.hpp"
#include "rmm/cuda_stream_view.hpp"
#include <cstddef>

namespace gpumap {
namespace utils {

template <typename Data_t, typename Idx_t>
__global__ void memcpy_device_row_wise_kernel(Data_t *dst, Data_t const *src,
                                              Idx_t n_rows, size_t n_features,
                                              size_t data_block_size) {

  const size_t row = static_cast<size_t>(blockIdx.x);

  if (row >= n_rows)
    return;

  const size_t dst_row_offset = row * data_block_size;
  const size_t src_row_offset = row * n_features;

  for (size_t j = static_cast<size_t>(threadIdx.x); j < n_features;
       j += blockDim.x) {

    dst[dst_row_offset + j] = src[src_row_offset + j];
  }
}

template <typename T, typename Idx_t>
void append_device(T *dst_device, T const *src_device, Idx_t n_new_instances,
                   Idx_t n_features, Idx_t data_block_size,
                   rmm::cuda_stream_view stream) {

  int threads = 256;

  memcpy_device_row_wise_kernel<<<n_new_instances, threads, 0,
                                  stream.value()>>>(
      dst_device, src_device, n_new_instances, n_features, data_block_size);

  RAFT_CUDA_TRY(cudaGetLastError());
}

// Append host data to matrix (general version)
template <typename T, typename Idx_t>
void append_host(T *dst_device, T const *src_host, Idx_t n_new_instances,
                 Idx_t n_features, Idx_t data_block_size,
                 rmm::cuda_stream_view stream) {

  const size_t dst_pitch = static_cast<size_t>(data_block_size) * sizeof(T);
  const size_t src_pitch = static_cast<size_t>(n_features) * sizeof(T);
  const size_t width = static_cast<size_t>(n_features) * sizeof(T);
  const size_t height = static_cast<size_t>(n_new_instances);

  cudaError_t err =
      cudaMemcpy2DAsync(dst_device, dst_pitch, src_host, src_pitch, width,
                        height, cudaMemcpyHostToDevice, stream.value());
  RAFT_CUDA_TRY(err);
}

// Scatter `n_rows` contiguous rows of `width` elements each (packed in
// `src_device`, row i at src_device[i * width]) into arbitrary rows of
// `dst_device`, selected by `row_ids_device[i]` with row stride
// `dst_row_stride`. One block per row; used to replace many small per-row
// host->device copies at scattered offsets with a single kernel launch.
template <typename T, typename Idx_t>
__global__ void scatter_rows_kernel(T *dst_device, Idx_t dst_row_stride,
                                    Idx_t const *row_ids_device,
                                    T const *src_device, Idx_t width,
                                    Idx_t n_rows) {
  const Idx_t i = static_cast<Idx_t>(blockIdx.x);
  if (i >= n_rows)
    return;

  const size_t dst_offset =
      static_cast<size_t>(row_ids_device[i]) * dst_row_stride;
  const size_t src_offset = static_cast<size_t>(i) * width;

  for (Idx_t j = static_cast<Idx_t>(threadIdx.x); j < width; j += blockDim.x) {
    dst_device[dst_offset + j] = src_device[src_offset + j];
  }
}

// Host-side driver for scatter_rows_kernel: copies `row_ids_host` and
// `src_host` (n_rows * width contiguous elements) to device scratch buffers
// supplied by the caller, then scatters into dst_device. Caller owns the
// scratch buffers so repeated calls can reuse the same allocation.
template <typename T, typename Idx_t>
void scatter_rows_host(T *dst_device, Idx_t dst_row_stride,
                       Idx_t const *row_ids_host, T const *src_host,
                       Idx_t width, Idx_t n_rows, Idx_t *row_ids_scratch,
                       T *src_scratch, rmm::cuda_stream_view stream) {
  if (n_rows == 0)
    return;

  raft::copy(row_ids_scratch, row_ids_host, static_cast<size_t>(n_rows),
             stream.value());
  raft::copy(src_scratch, src_host,
             static_cast<size_t>(n_rows) * static_cast<size_t>(width),
             stream.value());

  const int threads = 32;
  scatter_rows_kernel<T, Idx_t><<<n_rows, threads, 0, stream.value()>>>(
      dst_device, dst_row_stride, row_ids_scratch, src_scratch, width,
      n_rows);
  RAFT_CUDA_TRY(cudaGetLastError());
}
} // namespace utils

} // namespace gpumap