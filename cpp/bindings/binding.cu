#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "proumap/proumap.cuh"
#include "proumap_wrapper.cuh"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "proumap CUDA extension module";

  py::class_<proumap::UMAPConfig>(m, "UMAPConfig")
      .def(py::init<>())
      .def_readwrite("n_neighbors", &proumap::UMAPConfig::n_neighbors)
      .def_readwrite("n_components", &proumap::UMAPConfig::n_components)
      .def_readwrite("min_dist", &proumap::UMAPConfig::min_dist)
      .def_readwrite("n_epochs", &proumap::UMAPConfig::n_epochs)
      .def_readwrite("negative_sample_rate",
                     &proumap::UMAPConfig::negative_sample_rate)
      .def_readwrite("verbose", &proumap::UMAPConfig::verbose);

  py::class_<proumap::ProUMAPConfig, proumap::UMAPConfig>(m, "ProUMAPConfig")
      .def(py::init<>())
      .def_readwrite("max_instances", &proumap::ProUMAPConfig::max_instances)
      .def_readwrite("batch_size_hint",
                     &proumap::ProUMAPConfig::batch_size_hint)
      .def_readwrite("n_features", &proumap::ProUMAPConfig::n_features);

  py::class_<ProUMAPWrapper>(m, "ProUMAP")
      .def(py::init<const proumap::ProUMAPConfig &>(), py::arg("config"))
      .def("append_data", &ProUMAPWrapper::append_data, py::arg("x"))
      .def("run_nng_build", &ProUMAPWrapper::run_nng_build)
      .def("run_embedding_optimization",
           &ProUMAPWrapper::run_embedding_optimization)
      .def("get_data_view", &ProUMAPWrapper::get_data_view)
      .def("get_embedding_view", &ProUMAPWrapper::get_embedding_view);
};
