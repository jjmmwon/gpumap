#pragma once

#include <cstdint>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>
#include <stdexcept>
#include <sys/types.h>

namespace proumap {
namespace storage {
template <typename Data_t, typename Idx_t = uint32_t> struct DeviceMatrix2D {
  Idx_t max_rows;
  Idx_t max_cols;

  Idx_t n_rows = 0;

  rmm::device_uvector<Data_t> buf;

  DeviceMatrix2D() : buf(0, rmm::cuda_stream_default) {}
  DeviceMatrix2D(Idx_t max_rows_, Idx_t max_cols_,
                 rmm::cuda_stream_view stream_)
      : max_rows(max_rows_), max_cols(max_cols_),
        buf(max_rows_ * max_cols_, stream_) {}

  Idx_t capacity_rows() const { return max_rows; }
  Idx_t rows() const { return n_rows; }

  Data_t *data() { return buf.data(); }
  const Data_t *data() const { return buf.data(); }

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
};
} // namespace storage
} // namespace proumap
