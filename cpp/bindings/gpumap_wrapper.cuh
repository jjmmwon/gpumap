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

  gpumap::GPUMAPParams config;
  gpumap::GPUMAP model;

  explicit GPUMAPWrapper(const gpumap::GPUMAPParams &config)
      : config(config), model(config) {}

  void initialize(size_t n_instances, size_t n_features) {
    model.initialize(n_instances, n_features);
  }

  void append_data(py::array x) {
    auto v = binding_utils::require_numpy_2d_f32(x);
    if (v.n_cols != config.n_features) {
      throw std::runtime_error(
          "n_features mismatch with numpy array second dimension.");
    }
    if (v.n_rows <= 0) {
      return;
    }

    model.append_data(v.ptr, static_cast<Idx_t>(v.n_rows), false);
  }

  gpumap::GPUMAPResult run(size_t n_insertion_ops, size_t n_update_ops,
                           size_t n_embedding_ops) {
    return model.run(n_insertion_ops, n_update_ops, n_embedding_ops);
  }

  void close() { model.close(); }

  // py::object get_data_view() const {
  //   return binding_utils::to_cai_object(model.get_data_view());
  // }

  py::object get_embedding_view() const {
    return binding_utils::to_cai_object(model.get_embedding());
  }
};
