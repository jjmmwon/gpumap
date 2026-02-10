#include "impl.cuh"

namespace gpumap {

GPUMAP::GPUMAP(const GPUMAPParams &cfg) : impl_(std::make_unique<Impl>(cfg)) {}

GPUMAP::GPUMAP(GPUMAP &&) noexcept = default;
GPUMAP &GPUMAP::operator=(GPUMAP &&) noexcept = default;

GPUMAP::~GPUMAP() = default;

void GPUMAP::initialize(size_t n_instances, size_t n_features) {
  // Initialization logic if needed
  // For now, initialization happens in constructor
}

void GPUMAP::append_data(const float *ptr, size_t n_new_instances,
                         bool is_device) {
  impl_->append_data(ptr, n_new_instances, is_device);
}

GPUMAPResult GPUMAP::run(size_t n_insertion_ops, size_t n_update_ops,
                         size_t n_embedding_ops) {
  return impl_->run(n_insertion_ops, n_update_ops, n_embedding_ops);
}

void GPUMAP::close() { impl_->close(); }

// GPUMAP::DataView GPUMAP::get_data_view() const {
//   return impl_->get_data_view();
// }

GPUMAP::EmbedView GPUMAP::get_embedding() const {
  return impl_->get_embedding();
}

// const GPUMAPParams &GPUMAP::get_config() const { return impl_->params; }

} // namespace gpumap
