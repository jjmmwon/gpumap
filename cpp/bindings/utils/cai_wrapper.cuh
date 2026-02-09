#pragma once

#include <cstdint>
#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>

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

template <typename Data_t, typename Idx_t, typename LayoutPolicy>
inline pybind11::object to_cai_object(
    const raft::device_matrix_view<const Data_t, Idx_t, LayoutPolicy> &view) {
  static_assert(std::is_same<Data_t, float>::value,
                "Only float32 is supported for CAI export.");

  const auto *ptr = view.data_handle();
  const std::size_t rows = static_cast<std::size_t>(view.extent(0));
  const std::size_t cols = static_cast<std::size_t>(view.extent(1));

  const std::size_t row_stride = cols * sizeof(Data_t);
  const std::size_t col_stride = sizeof(Data_t);

  pybind11::dict cai;
  cai["version"] = 3;
  cai["shape"] = pybind11::make_tuple(rows, cols);
  cai["typestr"] = get_cai_type_str<Data_t>();
  cai["strides"] = pybind11::make_tuple(row_stride, col_stride);
  cai["data"] = pybind11::make_tuple(reinterpret_cast<std::uintptr_t>(ptr),
                                     true /* readonly */);

  pybind11::object ns =
      pybind11::module_::import("types").attr("SimpleNamespace")();
  ns.attr("__cuda_array_interface__") = cai;
  return ns;
}

} // namespace binding_utils
