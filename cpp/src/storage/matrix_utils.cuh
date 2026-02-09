#pragma once

#include "matrix2D.cuh"
#include <stdexcept>

namespace gpumap {

// Append device data to matrix
template <typename T, typename Idx_t>
void
DeviceMatrix2D<T, Idx_t>::append_device(T const *src_device, Idx_t n_new_rows,
                                        rmm::cuda_stream_view stream) {
  if (this->n_rows + n_new_rows > this->max_rows)
    throw std::runtime_error("Exceeding maximum number of rows");

  size_t dst_off =
      static_cast<size_t>(this->n_rows) * static_cast<size_t>(this->max_cols);

  size_t n_elem =
      static_cast<size_t>(n_new_rows) * static_cast<size_t>(this->max_cols);

  RAFT_CUDA_TRY(cudaMemcpyAsync(this->data() + dst_off, src_device,
                                n_elem * sizeof(T), cudaMemcpyDeviceToDevice,
                                stream.value()));

  this->n_rows += n_new_rows;
}

// Append host data to matrix
template <typename T, typename Idx_t>
void
DeviceMatrix2D<T, Idx_t>::append_host(T const *src_host, Idx_t n_new_rows,
                                      rmm::cuda_stream_view stream) {
  if (this->n_rows + n_new_rows > this->max_rows)
    throw std::runtime_error("Exceeding maximum number of rows");

  size_t dst_off =
      static_cast<size_t>(this->n_rows) * static_cast<size_t>(this->max_cols);

  size_t n_elem =
      static_cast<size_t>(n_new_rows) * static_cast<size_t>(this->max_cols);
  RAFT_CUDA_TRY(cudaMemcpyAsync(this->data() + dst_off, src_host,
                                n_elem * sizeof(T), cudaMemcpyHostToDevice,
                                stream.value()));

  this->n_rows += n_new_rows;
}

// Overwrite device data in matrix
template <typename T, typename Idx_t>
void
DeviceMatrix2D<T, Idx_t>::overwrite_device(T const *src_device, Idx_t row_start,
                                           Idx_t n_overwrite_rows,
                                           rmm::cuda_stream_view stream) {
  if (row_start + n_overwrite_rows > this->n_rows)
    throw std::runtime_error("Overwrite exceeds number of rows");
  size_t dst_off =
      static_cast<size_t>(row_start) * static_cast<size_t>(this->max_cols);

  size_t n_elem = static_cast<size_t>(n_overwrite_rows) *
                  static_cast<size_t>(this->max_cols);

  RAFT_CUDA_TRY(cudaMemcpyAsync(this->data() + dst_off, src_device,
                                n_elem * sizeof(T), cudaMemcpyDeviceToDevice,
                                stream.value()));
}

} // namespace gpumap