#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cctype>
#include <string>

#include "gpumap/gpumap.cuh"
#include "gpumap_wrapper.cuh"

namespace py = pybind11;

namespace {

std::string metric_to_string(gpumap::Metric m) {
  switch (m) {
  case gpumap::Metric::Cosine:
    return "cosine";
  case gpumap::Metric::Euclidean:
  default:
    return "euclidean";
  }
}

// Accepts a metric name string (e.g. "cosine", case-insensitive -- matches
// umap-learn's `metric=` kwarg style), or a plain int / gpumap.Metric
// IntEnum (which pybind11 sees as an int either way).
gpumap::Metric metric_from_object(const py::object &v) {
  if (py::isinstance<py::str>(v)) {
    std::string s = v.cast<std::string>();
    for (char &c : s)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "euclidean")
      return gpumap::Metric::Euclidean;
    if (s == "cosine")
      return gpumap::Metric::Cosine;
    throw py::value_error("Unknown metric '" + s +
                          "' -- expected 'euclidean' or 'cosine'");
  }
  return static_cast<gpumap::Metric>(v.cast<int>());
}

} // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "gpumap CUDA extension module";

  // NOTE: gpumap::Metric is bound as a string-valued property (not
  // py::enum_) -- py::enum_<gpumap::Metric>'s generated value-constructor
  // triggers an ambiguous-template-instantiation error under this project's
  // nvcc/g++ toolchain (a known pybind11/nvcc incompatibility, unrelated to
  // Metric's definition). The C++ side keeps full enum type safety; Python
  // callers pass a metric name string (umap-learn style, e.g.
  // `params.metric = "cosine"`) or, for backwards compatibility, an int /
  // the gpumap.Metric IntEnum (gpumap/metric.py).
  py::class_<gpumap::GPUMAPParams>(m, "GPUMAPParams")
      .def(py::init<>())
      .def_readwrite("n_neighbors", &gpumap::GPUMAPParams::n_neighbors)
      .def_readwrite("n_components", &gpumap::GPUMAPParams::n_components)
      .def_readwrite("min_dist", &gpumap::GPUMAPParams::min_dist)
      .def_readwrite("negative_sample_rate",
                     &gpumap::GPUMAPParams::negative_sample_rate)
      .def_readwrite("max_epoch", &gpumap::GPUMAPParams::max_epoch)
      .def_readwrite("a", &gpumap::GPUMAPParams::a)
      .def_readwrite("b", &gpumap::GPUMAPParams::b)
      .def_readwrite("alpha", &gpumap::GPUMAPParams::alpha)
      .def_readwrite("verbose", &gpumap::GPUMAPParams::verbose)
      .def_property(
          "metric",
          [](const gpumap::GPUMAPParams &p) {
            return metric_to_string(p.metric);
          },
          [](gpumap::GPUMAPParams &p, const py::object &v) {
            p.metric = metric_from_object(v);
          })
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
                    &gpumap::GPUMAPResult::insertion_completed)
      .def_readonly("update_completed", &gpumap::GPUMAPResult::update_completed)
      .def_readonly("embedding_completed",
                    &gpumap::GPUMAPResult::embedding_completed)
      .def_readonly("n_inserted_instances",
                    &gpumap::GPUMAPResult::n_inserted_instances)
      .def_readonly("n_update_changed_edges",
                    &gpumap::GPUMAPResult::n_update_changed_edges)
      .def_readonly("embedding_pending_edges",
                    &gpumap::GPUMAPResult::embedding_pending_edges)
      .def_readonly("insertion_queue_size",
                    &gpumap::GPUMAPResult::insertion_queue_size)
      .def_readonly("update_queue_size",
                    &gpumap::GPUMAPResult::update_queue_size)
      .def_readonly("embedding_queue_size",
                    &gpumap::GPUMAPResult::embedding_queue_size);

  py::class_<GPUMAPWrapper>(m, "GPUMAP")
      .def(py::init<const gpumap::GPUMAPParams &>(), py::arg("config"))
      .def("initialize", &GPUMAPWrapper::initialize, py::arg("n_instances"),
           py::arg("n_features"))
      .def("append_data", &GPUMAPWrapper::append_data, py::arg("x"))
      .def("run", &GPUMAPWrapper::run, py::arg("n_jobs_insertion"),
           py::arg("n_jobs_update"), py::arg("n_jobs_embedding"),
           py::call_guard<py::gil_scoped_release>())
      .def("close", &GPUMAPWrapper::close)
      .def("get_embedding", &GPUMAPWrapper::get_embedding)
      .def("get_neighbor_indices", &GPUMAPWrapper::get_neighbor_indices)
      .def("get_neighbor_dists", &GPUMAPWrapper::get_neighbor_dists);
};
