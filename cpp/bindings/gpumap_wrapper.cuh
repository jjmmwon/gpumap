#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gpumap/gpumap.cuh"
#include "utils/cai_wrapper.cuh"
#include "utils/numpy_utils.cuh"

namespace py = pybind11;

// ---------- GPUMAP wrapper for numpy input ----------
class GPUMAPWrapper {
public:
  using Data_t = float;
  using Idx_t = uint32_t;

  gpumap::GPUMAPParams params;
  gpumap::GPUMAP model;

  explicit GPUMAPWrapper(const gpumap::GPUMAPParams &params)
      : params(params), model(params) {}

  void initialize(size_t n_instances, size_t n_features) {
    model.initialize(n_instances, n_features);
  }

  void append_data(py::array x) {
    auto v = binding_utils::require_numpy_2d_f32(x);
    if (v.n_cols != params.n_features) {
      throw std::runtime_error(
          "n_features mismatch with numpy array second dimension.");
    }
    if (v.n_rows <= 0) {
      return;
    }

    model.append_data(v.ptr, static_cast<Idx_t>(v.n_rows));
  }

  gpumap::GPUMAPResult run(uint32_t n_insertion_ops, uint32_t n_update_ops,
                           uint32_t n_embedding_ops) {
    return model.run(n_insertion_ops, n_update_ops, n_embedding_ops);
  }

  void close() { model.close(); }

  py::object get_embedding() const {
    return binding_utils::to_cai_object(model.get_embedding());
  }

  py::object get_neighbor_indices() const {
    return binding_utils::to_cai_object(model.get_neighbor_indices());
  }

  py::object get_neighbor_dists() const {
    return binding_utils::to_cai_object(model.get_neighbor_dists());
  }
};
