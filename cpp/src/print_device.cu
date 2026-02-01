#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check_cuda(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
  }
}

extern "C" void print_device_matrix_f32_impl(const float *d_ptr, int64_t n_rows,
                                             int64_t n_cols, int64_t ld,
                                             int max_rows, int max_cols) {
  const int64_t r_lim = std::min<int64_t>(n_rows, std::max(0, max_rows));
  const int64_t c_lim = std::min<int64_t>(n_cols, std::max(0, max_cols));

  std::cout << "[print_device_matrix_f32_impl] shape=(" << n_rows << ", "
            << n_cols << "), ld=" << ld << ", showing (" << r_lim << " x "
            << c_lim << ")\n";

  if (r_lim == 0 || c_lim == 0)
    return;

  std::vector<float> h_buf(static_cast<size_t>(c_lim));

  for (int64_t r = 0; r < r_lim; ++r) {
    const float *d_row = d_ptr + r * ld;

    check_cuda(cudaMemcpy(h_buf.data(), d_row,
                          sizeof(float) * static_cast<size_t>(c_lim),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H");

    std::cout << "row " << r << ": ";
    for (int64_t c = 0; c < c_lim; ++c) {
      std::cout << h_buf[static_cast<size_t>(c)];
      if (c + 1 != c_lim)
        std::cout << ", ";
    }
    std::cout << "\n";
  }
}