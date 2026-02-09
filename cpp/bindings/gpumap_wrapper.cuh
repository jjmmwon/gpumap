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

  gpumap::GPUMAPConfig config;
  gpumap::GPUMAP model;

  explicit GPUMAPWrapper(const gpumap::GPUMAPConfig &config)
      : config(config), model(config) {}

  void append_data(py::array x) {
    auto v = binding_utils::require_numpy_2d_f32(x);
    if (v.n_cols != config.n_features) {
      throw std::runtime_error(
          "n_features mismatch with numpy array second dimension.");
    }
    if (v.n_rows <= 0) {
      return;
    }

    model.append_data(v.ptr, static_cast<Idx_t>(v.n_rows),
                      static_cast<Idx_t>(v.n_cols));
  }

  void run_nng_build() { model.run_nng_build(); }
  void run_embedding_optimization() { model.run_embedding_optimization(); }

  py::object get_data_view() const {
    return binding_utils::to_cai_object(model.get_data_view());
  }

  py::object get_embedding_view() const {
    return binding_utils::to_cai_object(model.get_embedding_view());
  }
};
