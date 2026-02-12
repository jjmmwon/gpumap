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

void GPUMAP::append_data(const float *ptr, size_t n_new_instances) {
  impl_->append_data(ptr, n_new_instances);
}

GPUMAPResult GPUMAP::run(size_t n_insertion_ops, size_t n_update_ops,
                         size_t n_embedding_ops) {
  return impl_->run(n_insertion_ops, n_update_ops, n_embedding_ops);
}

void GPUMAP::close() { impl_->close(); }

GPUMAP::EmbedView GPUMAP::get_embedding() const {
  return impl_->get_embedding();
}

GPUMAP::NeighborIndicesView GPUMAP::get_neighbor_indices() const {
  return impl_->get_neighbor_indices();
}

GPUMAP::NeighborDistsView GPUMAP::get_neighbor_dists() const {
  return impl_->get_neighbor_dists();
}


} // namespace gpumap
