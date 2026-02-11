#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gpumap/gpumap.cuh"
#include "gpumap_wrapper.cuh"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "gpumap CUDA extension module";


  py::class_<gpumap::GPUMAPParams>(m, "GPUMAPParams")
      .def(py::init<>())
      .def_readwrite("n_neighbors", &gpumap::GPUMAPParams::n_neighbors)
      .def_readwrite("n_components", &gpumap::GPUMAPParams::n_components)
      .def_readwrite("min_dist", &gpumap::GPUMAPParams::min_dist)
      .def_readwrite("n_epochs", &gpumap::GPUMAPParams::n_epochs)
      .def_readwrite("negative_sample_rate",
                     &gpumap::GPUMAPParams::negative_sample_rate)
    .def_readwrite("max_epoch", &gpumap::GPUMAPParams::max_epoch)
    .def_readwrite("a", &gpumap::GPUMAPParams::a)
    .def_readwrite("b", &gpumap::GPUMAPParams::b)
    .def_readwrite("alpha", &gpumap::GPUMAPParams::alpha)
      .def_readwrite("verbose", &gpumap::GPUMAPParams::verbose)
      .def_readwrite("n_instances", &gpumap::GPUMAPParams::n_instances)
      .def_readwrite("n_features", &gpumap::GPUMAPParams::n_features);

  py::class_<gpumap::GPUMAPResult>(m, "GPUMAPResult")
      .def_readonly("n_insertion_ops", &gpumap::GPUMAPResult::n_insertion_ops)
      .def_readonly("n_update_ops", &gpumap::GPUMAPResult::n_update_ops)
      .def_readonly("n_embedding_ops", &gpumap::GPUMAPResult::n_embedding_ops)
      .def_readonly("insertion_time", &gpumap::GPUMAPResult::insertion_time)
      .def_readonly("update_time", &gpumap::GPUMAPResult::update_time)
      .def_readonly("embedding_time", &gpumap::GPUMAPResult::embedding_time)
      .def_readonly("insertion_completed",
                    &gpumap::GPUMAPResult::insertion_completed);

  py::class_<GPUMAPWrapper>(m, "GPUMAP")
      .def(py::init<const gpumap::GPUMAPParams &>(), py::arg("config"))
      .def("initialize", &GPUMAPWrapper::initialize, py::arg("n_instances"), py::arg("n_features"))
      .def("append_data", &GPUMAPWrapper::append_data, py::arg("x"))
      .def("run", &GPUMAPWrapper::run, py::arg("n_jobs_insertion"),
           py::arg("n_jobs_update"), py::arg("n_jobs_embedding"))
      .def("close", &GPUMAPWrapper::close)
      .def("get_embedding", &GPUMAPWrapper::get_embedding)
      .def("get_neighbor_indices", &GPUMAPWrapper::get_neighbor_indices)
      .def("get_neighbor_dists", &GPUMAPWrapper::get_neighbor_dists);
};
