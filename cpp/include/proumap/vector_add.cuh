#pragma once
#include <cstdint>

extern "C" void vector_add_f32_impl(const float *a, const float *b, float *out,
                                    int64_t n);