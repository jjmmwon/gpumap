#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

static void check_cuda(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
  }
}

__global__ void vector_add_f32_kernel(const float *a, const float *b,
                                      float *out, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
}

extern "C" void vector_add_f32_impl(const float *a, const float *b, float *out,
                                    int64_t n) {
  if (n <= 0)
    return;

  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);

  vector_add_f32_kernel<<<blocks, threads>>>(a, b, out, n);
  check_cuda(cudaGetLastError(), "vector_add_f32_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}