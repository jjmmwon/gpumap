cmake_minimum_required(VERSION 3.24)

# Use system GCC instead of conda GCC for CUDA compatibility
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-9")

find_package(CUDAToolkit REQUIRED)

find_package(rmm CONFIG REQUIRED)
find_package(raft CONFIG REQUIRED)
find_package(cuvs CONFIG REQUIRED)

pybind11_add_module(_core MODULE
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/binding.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/impl.cu
)

target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings
)

target_link_libraries(_core PRIVATE pybind11::headers)

target_link_libraries(_core PRIVATE
  CUDA::cudart
  rmm::rmm
  raft::raft
  cuvs::cuvs
)

target_compile_definitions(_core PRIVATE
  LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
)


set_target_properties(_core PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


install(TARGETS _core
  LIBRARY DESTINATION gpumap
  RUNTIME DESTINATION gpumap
  ARCHIVE DESTINATION gpumap
)
