#pragma once

#include "raft/util/cuda_rt_essentials.hpp"
#include "rmm/cuda_stream_view.hpp"
#include <cstddef>

namespace gpumap {
namespace utils {

template <typename Data_t, typename Idx_t>
__global__ void memcpy_device_row_wise_kernel(Data_t *dst, Data_t const *src,
                                              Idx_t n_rows, size_t n_features, size_t data_block_size) {

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
                   Idx_t n_features, Idx_t data_block_size, rmm::cuda_stream_view stream) {

  int threads = 256;

  memcpy_device_row_wise_kernel<<<n_new_instances, threads, 0, stream.value()>>>(
      dst_device, src_device, n_new_instances, n_features, data_block_size);

  RAFT_CUDA_TRY(cudaGetLastError());
}

// Append host data to matrix (일반 버전)
template <typename T, typename Idx_t>
void append_host(T *dst_device, T const *src_host, Idx_t n_new_instances,
  Idx_t n_features, Idx_t data_block_size, rmm::cuda_stream_view stream) {

  const size_t dst_pitch = static_cast<size_t>(data_block_size) * sizeof(T);
  const size_t src_pitch = static_cast<size_t>(n_features) * sizeof(T);
  const size_t width = static_cast<size_t>(n_features) * sizeof(T);
  const size_t height = static_cast<size_t>(n_new_instances);

  cudaError_t err =
      cudaMemcpy2DAsync(dst_device, dst_pitch, src_host, src_pitch, width,
                        height, cudaMemcpyHostToDevice, stream.value());
  RAFT_CUDA_TRY(err);
}

// Overwrite device data in matrix (일반 버전)
// template <typename T, typename Idx_t>
// void overwrite_device(T *dst_device, T const *src_device,
//                       Idx_t n_overwrite_rows, Idx_t n_cols,
//                       rmm::cuda_stream_view stream) {

//   size_t n_elem =
//       static_cast<size_t>(n_overwrite_rows) * static_cast<size_t>(n_cols);

//   int threads = 256;

//   memcpy_device_row_wise_kernel<<<n_overwrite_rows, threads, 0,
//                                   stream.value()>>>(dst_device, src_device,
//                                                     n_overwrite_rows, n_cols, n_cols);
//   RAFT_CUDA_TRY(cudaGetLastError());
// }
} // namespace utils

} // namespace gpumap