#include <pybind11/pybind11.h>
#include <raft/core/device_mdspan.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace py = pybind11;

namespace binding_utils {
template <typename T> inline std::string get_cai_type_str() {
  if constexpr (std::is_same_v<T, float>)
    return "<f4";
  else if constexpr (std::is_same_v<T, double>)
    return "<f8";
  else if constexpr (std::is_same_v<T, int8_t>)
    return "|i1";
  else if constexpr (std::is_same_v<T, uint8_t>)
    return "|u1";
  else if constexpr (std::is_same_v<T, int16_t>)
    return "<i2";
  else if constexpr (std::is_same_v<T, uint16_t>)
    return "<u2";
  else if constexpr (std::is_same_v<T, int32_t>)
    return "<i4";
  else if constexpr (std::is_same_v<T, uint32_t>)
    return "<u4";
  else if constexpr (std::is_same_v<T, int64_t>)
    return "<i8";
  else if constexpr (std::is_same_v<T, uint64_t>)
    return "<u8";
  else {
    throw std::runtime_error("Unsupported dtype for __cuda_array_interface__");
  }
}

template <typename T>
py::object make_cuda_array_interface_wrapper(
    const raft::device_matrix_view<T, uint32_t, raft::row_major> &view,
    py::object owner) {
  // shape
  py::tuple shape = py::make_tuple(view.extent(0), view.extent(1));
  // strides
  py::tuple strides =
      py::make_tuple(sizeof(T) * view.stride(0), sizeof(T) * view.stride(1));
  // data pointer
  uintptr_t data_ptr = reinterpret_cast<uintptr_t>(view.data());
  // typestr
  std::string typestr = get_cai_type_str<T>();
  // Construct the __cuda_array_interface__ dictionary
  py::dict cai_dict;
  cai_dict["shape"] = shape;
  cai_dict["strides"] = strides;
  cai_dict["data"] = py::make_tuple(data_ptr, false); // false: not read-only
  cai_dict["typestr"] = typestr;
  cai_dict["version"] = 2;
  cai_dict["owner"] = owner; // keep a reference to the owner object
  return cai_dict;
}

} // namespace binding_utils