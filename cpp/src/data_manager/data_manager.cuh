#pragma once

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

struct ProuData {
  int64_t max_n;
  int64_t d;
  int64_t k;

  int64_t n_current;

  rmm::cuda_stream stream;

  rmm::device_uvector<double> streaming_data; // [max_n, d]
  rmm::device_uvector<int32_t> nng;           // [max_n, k]
  rmm::device_uvector<double> embedding;      // [max_n, k]
  rmm::device_uvector<double> projection;     // [max_n, 2]

  ProuData(int64_t max_n_, int64_t d_, int64_t k_)
      : max_n(max_n_), d(d_), k(k_), n_current(0), stream(),
        streaming_data(max_n * d, stream), nng(max_n * k, stream),
        embedding(max_n * k, stream), projection(max_n * 2, stream) {}
};
