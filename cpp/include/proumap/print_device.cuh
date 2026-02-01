#pragma once
#include <cstdint>

extern "C" void print_device_matrix_f32_impl(const float *d_ptr, int64_t n_rows,
                                             int64_t n_cols, int64_t ld,
                                             int max_rows, int max_cols);