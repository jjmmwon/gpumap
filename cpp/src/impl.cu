#include "impl.cuh"

namespace gpumap {

GPUMAP::GPUMAP(const GPUMAPParams &cfg) : impl_(std::make_unique<Impl>(cfg)) {}

GPUMAP::GPUMAP(GPUMAP &&) noexcept = default;
GPUMAP &GPUMAP::operator=(GPUMAP &&) noexcept = default;

GPUMAP::~GPUMAP() = default;

void GPUMAP::append_data(const float *ptr, Idx_t rows, Idx_t cols) {
  //
  impl_->append_data(ptr, rows, cols);
}

void GPUMAP::run(const size_t n_jobs_insertion, const size_t n_jobs_update,
                 const size_t n_jobs_embedding) {
  //
  impl_->insert(n_jobs_insertion);
  impl_->update(n_jobs_update);
  impl_->embed(n_jobs_embedding);
}

GPUMAP::DataView GPUMAP::get_data_view() const {
  return impl_->get_data_view();
}

GPUMAP::EmbedView GPUMAP::get_embedding_view() const {
  return impl_->get_embedding_view();
}

const GPUMAPParams &GPUMAP::get_config() const { return impl_->params; }

} // namespace gpumap
