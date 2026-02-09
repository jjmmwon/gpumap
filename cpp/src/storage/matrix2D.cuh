#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>
#include <stdexcept>
#include <sys/types.h>
#include <vector>

namespace gpumap {
template <typename Data_t, typename Idx_t = uint32_t> struct DeviceMatrix2D {
  Idx_t max_rows;
  Idx_t max_cols;

  Idx_t n_rows = 0;

  rmm::device_uvector<Data_t> buf;

  DeviceMatrix2D(Idx_t max_rows_, Idx_t max_cols_,
                 rmm::cuda_stream_view stream_)
      : max_rows(max_rows_), max_cols(max_cols_),
        buf(max_rows_ * max_cols_, stream_) {}

  // GPU 접근을 위한 __device__ 메서드들
  __device__ __host__ Data_t *data() { return buf.data(); }
  __device__ __host__ const Data_t *data() const { return buf.data(); }

  __device__ __host__ Data_t &operator()(Idx_t row, Idx_t col) {
    return buf.data()[row * max_cols + col];
  }

  __device__ __host__ const Data_t &operator()(Idx_t row, Idx_t col) const {
    return buf.data()[row * max_cols + col];
  }

  __device__ __host__ Data_t *row_ptr(Idx_t row) {
    return buf.data() + row * max_cols;
  }

  __device__ __host__ const Data_t *row_ptr(Idx_t row) const {
    return buf.data() + row * max_cols;
  }

  __device__ __host__ Idx_t get_n_rows() const { return n_rows; }
  __device__ __host__ Idx_t get_max_cols() const { return max_cols; }
  __device__ __host__ Idx_t get_max_rows() const { return max_rows; }

  auto view() const {
    return raft::make_device_matrix_view<const Data_t, Idx_t, raft::row_major>(
        data(), n_rows, max_cols);
  }

  auto submatrix_view(Idx_t row_start, Idx_t n_sub_rows) const
      -> raft::device_matrix_view<const Data_t, Idx_t, raft::row_major> {
    if (row_start + n_sub_rows > n_rows)
      throw std::runtime_error("Submatrix exceeds number of rows");

    // Safe casting to avoid overflow in pointer arithmetic
    auto offset =
        static_cast<size_t>(row_start) * static_cast<size_t>(max_cols);
    auto *ptr = data() + offset;

    return raft::make_device_matrix_view<const Data_t, Idx_t, raft::row_major>(
        ptr, n_sub_rows, max_cols);
  }

  __device__ __host__ void
  append_device(const Data_t *src_device, Idx_t n_new_rows,
                rmm::cuda_stream_view stream = rmm::cuda_stream_default);
  __device__ __host__ void
  append_host(const Data_t *src_host, Idx_t n_new_rows,
              rmm::cuda_stream_view stream = rmm::cuda_stream_default);
  __device__ __host__ void
  overwrite_device(const Data_t *src_device, Idx_t row_start,
                   Idx_t n_overwrite_rows,
                   rmm::cuda_stream_view stream = rmm::cuda_stream_default);
};
} // namespace gpumap
