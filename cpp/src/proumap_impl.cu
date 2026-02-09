#include "proumap_impl.cuh"

namespace proumap {

ProUMAP::ProUMAP() : impl_(nullptr) {}

ProUMAP::ProUMAP(const ProUMAPConfig &cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}

ProUMAP::ProUMAP(ProUMAP &&) noexcept = default;
ProUMAP &ProUMAP::operator=(ProUMAP &&) noexcept = default;

ProUMAP::~ProUMAP() = default;

void ProUMAP::append_data(const float *ptr, Idx_t rows, Idx_t cols) {
  impl_->append_data(ptr, rows, cols);
}

void ProUMAP::run_nng_build() { impl_->run_nng_build(); }

void ProUMAP::run_embedding_optimization() {
  impl_->run_embedding_optimization();
}

ProUMAP::DataView ProUMAP::get_data_view() const {
  return impl_->get_data_view();
}

ProUMAP::EmbedView ProUMAP::get_embedding_view() const {
  return impl_->get_embedding_view();
}

const ProUMAPConfig &ProUMAP::get_config() const { return impl_->config; }

} // namespace proumap
