#include "gpumap_impl.cuh"

namespace gpumap {

GPUMAP::GPUMAP() : impl_(nullptr) {}

GPUMAP::GPUMAP(const GPUMAPConfig &cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}

GPUMAP::GPUMAP(GPUMAP &&) noexcept = default;
GPUMAP &GPUMAP::operator=(GPUMAP &&) noexcept = default;

GPUMAP::~GPUMAP() = default;

void GPUMAP::append_data(const float *ptr, Idx_t rows, Idx_t cols) {
  //
  impl_->append_data(ptr, rows, cols);
}

void GPUMAP::run_nng_build() { 
  //
  impl_->run_nng_build(); 
}

void GPUMAP::run_embedding_optimization() {
  //
  impl_->run_embedding_optimization();
}

GPUMAP::DataView GPUMAP::get_data_view() const {
  return impl_->get_data_view();
}

GPUMAP::EmbedView GPUMAP::get_embedding_view() const {
  return impl_->get_embedding_view();
}

const GPUMAPConfig &GPUMAP::get_config() const { return impl_->config; }

} // namespace gpumap
