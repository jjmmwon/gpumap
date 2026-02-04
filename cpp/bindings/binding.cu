#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ---------- CUDA array interface parsing helpers ----------

struct CudaArray2DViewF32 {
  const float *ptr = nullptr;
  int64_t n_rows = 0;
  int64_t n_cols = 0;
  int64_t ld = 0; // leading dimension in elements (row-major)
};

struct CudaArray1DViewF32 {
  const float *ptr = nullptr;
  int64_t n = 0;
};

static void require_cuda_array_interface(py::object arr) {
  if (!py::hasattr(arr, "__cuda_array_interface__")) {
    throw std::runtime_error(
        "Input must provide __cuda_array_interface__ (e.g., cupy.ndarray).");
  }
}

static void require_f32_typestr(py::dict cai) {
  std::string typestr = py::cast<std::string>(cai["typestr"]);
  if (!(typestr == "<f4" || typestr == "|f4")) {
    throw std::runtime_error("Only float32 is supported. Got typestr=" +
                             typestr);
  }
}

static uintptr_t get_ptr_from_cai(py::dict cai) {
  auto data_tuple = py::cast<py::tuple>(cai["data"]);
  return py::cast<uintptr_t>(data_tuple[0]);
}

static CudaArray2DViewF32 parse_2d_f32(py::object arr) {
  require_cuda_array_interface(arr);
  py::dict cai = arr.attr("__cuda_array_interface__");
  require_f32_typestr(cai);

  auto shape = py::cast<std::vector<int64_t>>(cai["shape"]);
  if (shape.size() != 2)
    throw std::runtime_error("Expected 2D array.");
  int64_t n_rows = shape[0];
  int64_t n_cols = shape[1];

  int64_t ld = n_cols;
  if (cai.contains("strides") && !cai["strides"].is_none()) {
    auto strides = py::cast<std::vector<int64_t>>(cai["strides"]);
    if (strides.size() != 2)
      throw std::runtime_error("Invalid strides.");
    if (strides[1] != static_cast<int64_t>(sizeof(float))) {
      throw std::runtime_error(
          "Require contiguous last dimension (stride[1]==4).");
    }
    ld = strides[0] / static_cast<int64_t>(sizeof(float));
  }

  CudaArray2DViewF32 v;
  v.ptr = reinterpret_cast<const float *>(get_ptr_from_cai(cai));
  v.n_rows = n_rows;
  v.n_cols = n_cols;
  v.ld = ld;
  return v;
}

static CudaArray1DViewF32 parse_1d_f32(py::object arr) {
  require_cuda_array_interface(arr);
  py::dict cai = arr.attr("__cuda_array_interface__");
  require_f32_typestr(cai);

  auto shape = py::cast<std::vector<int64_t>>(cai["shape"]);
  if (shape.size() != 1)
    throw std::runtime_error("Expected 1D array.");
  int64_t n = shape[0];

  if (cai.contains("strides") && !cai["strides"].is_none()) {
    auto strides = py::cast<std::vector<int64_t>>(cai["strides"]);
    if (strides.size() != 1)
      throw std::runtime_error("Invalid strides.");
    if (strides[0] != static_cast<int64_t>(sizeof(float))) {
      throw std::runtime_error("Require contiguous (stride==4 bytes).");
    }
  }

  CudaArray1DViewF32 v;
  v.ptr = reinterpret_cast<const float *>(get_ptr_from_cai(cai));
  v.n = n;
  return v;
}

// ---------- Existing: print matrix ----------
static void print_device_matrix_f32(py::object x, int max_rows, int max_cols) {
  auto v = parse_2d_f32(x);
  print_device_matrix_f32_impl(v.ptr, v.n_rows, v.n_cols, v.ld, max_rows,
                               max_cols);
}

// ---------- New: vector add ----------
static void add_f32(py::object a, py::object b, py::object out) {
  auto va = parse_1d_f32(a);
  auto vb = parse_1d_f32(b);
  auto vo = parse_1d_f32(out);

  if (va.n != vb.n || va.n != vo.n) {
    throw std::runtime_error("a, b, out must have the same length.");
  }

  vector_add_f32_impl(va.ptr, vb.ptr, const_cast<float *>(vo.ptr), va.n);
}

PYBIND11_MODULE(_backend, m) {
  m.doc() = "proumap CUDA extension: print matrix + vector add";

  m.def("print_device_matrix_f32", &print_device_matrix_f32, py::arg("x"),
        py::arg("max_rows") = 4, py::arg("max_cols") = 8);

  m.def("add_f32", &add_f32, py::arg("a"), py::arg("b"), py::arg("out"));
}