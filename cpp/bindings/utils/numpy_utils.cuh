#pragma once

#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace binding_utils {

struct NumpyArray2DViewF32 {
  const float *ptr = nullptr;
  int64_t n_rows = 0;
  int64_t n_cols = 0;
};

inline NumpyArray2DViewF32 require_numpy_2d_f32(pybind11::array arr) {
  pybind11::buffer_info info = arr.request();
  if (info.ndim != 2) {
    throw std::runtime_error("Expected 2D numpy array.");
  }

  const std::string expected_format =
      pybind11::format_descriptor<float>::format();
  if (info.format != expected_format) {
    throw std::runtime_error("Only float32 numpy arrays are supported.");
  }

  if (info.strides.size() != 2) {
    throw std::runtime_error("Invalid numpy strides.");
  }

  const ssize_t stride_row = info.strides[0];
  const ssize_t stride_col = info.strides[1];
  const ssize_t expected_col_stride = static_cast<ssize_t>(sizeof(float));
  const ssize_t expected_row_stride =
      static_cast<ssize_t>(sizeof(float)) * info.shape[1];

  if (stride_col != expected_col_stride || stride_row != expected_row_stride) {
    throw std::runtime_error(
        "Require C-contiguous numpy array (row-major float32).");
  }

  NumpyArray2DViewF32 v;
  v.ptr = static_cast<const float *>(info.ptr);
  v.n_rows = static_cast<int64_t>(info.shape[0]);
  v.n_cols = static_cast<int64_t>(info.shape[1]);
  return v;
}

} // namespace binding_utils
