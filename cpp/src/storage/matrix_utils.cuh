#pragma once

#include "matrix2D.cuh"
#include "raft/core/resources.hpp"
#include <stdexcept>

namespace proumap {
namespace storage {
namespace utils {

// Append device data to matrix
template <typename T, typename Idx_t = uint32_t>
void append_device(raft::resources const &res, DeviceMatrix2D<T, Idx_t> &matrix,
                   T const *src_device, Idx_t n_new_rows) {
  if (matrix.n_rows + n_new_rows > matrix.max_rows)
    throw std::runtime_error("Exceeding maximum number of rows");

  auto stream = rmm::cuda_stream_view(raft::resource::get_cuda_stream(res));

  size_t dst_off =
      static_cast<size_t>(matrix.n_rows) * static_cast<size_t>(matrix.max_cols);

  size_t n_elem =
      static_cast<size_t>(n_new_rows) * static_cast<size_t>(matrix.max_cols);

  RAFT_CUDA_TRY(cudaMemcpyAsync(matrix.data() + dst_off, src_device,
                                n_elem * sizeof(T), cudaMemcpyDeviceToDevice,
                                stream.value()));

  matrix.n_rows += n_new_rows;
}

// Append host data to matrix
template <typename T, typename Idx_t = uint32_t>
void append_host(raft::resources const &res, DeviceMatrix2D<T, Idx_t> &matrix,
                 T const *src_host, Idx_t n_new_rows) {
  if (matrix.n_rows + n_new_rows > matrix.max_rows)
    throw std::runtime_error("Exceeding maximum number of rows");

  auto stream = rmm::cuda_stream_view(raft::resource::get_cuda_stream(res));

  size_t dst_off =
      static_cast<size_t>(matrix.n_rows) * static_cast<size_t>(matrix.max_cols);

  size_t n_elem =
      static_cast<size_t>(n_new_rows) * static_cast<size_t>(matrix.max_cols);

  RAFT_CUDA_TRY(cudaMemcpyAsync(matrix.data() + dst_off, src_host,
                                n_elem * sizeof(T), cudaMemcpyHostToDevice,
                                stream.value()));

  matrix.n_rows += n_new_rows;
}

// Overwrite device data in matrix
template <typename T, typename Idx_t = uint32_t>
void overwrite_device(raft::resources const &res,
                      DeviceMatrix2D<T, Idx_t> &matrix, T const *src_device,
                      Idx_t row_start, Idx_t n_overwrite_rows) {
  if (row_start + n_overwrite_rows > matrix.n_rows)
    throw std::runtime_error("Overwrite exceeds number of rows");

  auto stream = rmm::cuda_stream_view(raft::resource::get_cuda_stream(res));

  size_t dst_off =
      static_cast<size_t>(row_start) * static_cast<size_t>(matrix.max_cols);

  size_t n_elem = static_cast<size_t>(n_overwrite_rows) *
                  static_cast<size_t>(matrix.max_cols);

  RAFT_CUDA_TRY(cudaMemcpyAsync(matrix.data() + dst_off, src_device,
                                n_elem * sizeof(T), cudaMemcpyDeviceToDevice,
                                stream.value()));
}

} // namespace utils
} // namespace storage
} // namespace proumap