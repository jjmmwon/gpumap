#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gpumap/gpumap.cuh"
#include "gpumap_wrapper.cuh"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "gpumap CUDA extension module";

  py::class_<gpumap::UMAPConfig>(m, "UMAPConfig")
      .def(py::init<>())
      .def_readwrite("n_neighbors", &gpumap::UMAPConfig::n_neighbors)
      .def_readwrite("n_components", &gpumap::UMAPConfig::n_components)
      .def_readwrite("min_dist", &gpumap::UMAPConfig::min_dist)
      .def_readwrite("n_epochs", &gpumap::UMAPConfig::n_epochs)
      .def_readwrite("negative_sample_rate",
                     &gpumap::UMAPConfig::negative_sample_rate)
      .def_readwrite("verbose", &gpumap::UMAPConfig::verbose);

  py::class_<gpumap::GPUMAPConfig, gpumap::UMAPConfig>(m, \"GPUMAPConfig\")
      .def(py::init<>())
      .def_readwrite(\"n_instances\", &gpumap::GPUMAPConfig::n_instances)
      .def_readwrite(\"n_features\",
                     &gpumap::GPUMAPConfig::n_features);


  py::class_<GPUMAPWrapper>(m, \"GPUMAP\")
      .def(py::init<const gpumap::GPUMAPConfig &>(), py::arg(\"config\"))
      .def("append_data", &GPUMAPWrapper::append_data, py::arg("x"))
      .def("run_nng_build", &GPUMAPWrapper::run_nng_build)
      .def("run_embedding_optimization",
           &GPUMAPWrapper::run_embedding_optimization)
      .def("get_data_view", &GPUMAPWrapper::get_data_view)
      .def("get_embedding_view", &GPUMAPWrapper::get_embedding_view);
};
