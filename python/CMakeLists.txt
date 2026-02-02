cmake_minimum_required(VERSION 3.24)

find_package(pybind11 CONFIG REQUIRED)

find_package(CUDAToolkit REQUIRED)

pybind11_add_module(_backend MODULE
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/binding.cu
  ${PROJECT_SOURCE_DIR}/cpp/src/print_device.cu
  ${PROJECT_SOURCE_DIR}/cpp/src/vector_add.cu
)

target_link_libraries(_backend PRIVATE pybind11::headers)

target_include_directories(_backend PRIVATE
  ${PROJECT_SOURCE_DIR}/cpp/include
)

target_link_libraries(_backend PRIVATE
  CUDA::cudart
)


set_target_properties(_backend PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


install(TARGETS _backend
  LIBRARY DESTINATION proumap
  RUNTIME DESTINATION proumap
  ARCHIVE DESTINATION proumap
)