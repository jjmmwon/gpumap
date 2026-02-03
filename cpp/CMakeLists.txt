cmake_minimum_required(VERSION 3.24)

find_package(CUDAToolkit REQUIRED)

find_package(rmm CONFIG REQUIRED)
find_package(raft CONFIG REQUIRED)
find_package(cuvs CONFIG REQUIRED)

pybind11_add_module(_core MODULE
  ${PROJECT_SOURCE_DIR}/bindings/binding.cu
  ${PROJECT_SOURCE_DIR}/cpp/src/proumap.cu
)

target_include_directories(_core PRIVATE
  ${PROJECT_SOURCE_DIR}/cpp/include
)

target_link_libraries(_core PRIVATE pybind11::headers)

target_link_libraries(_core PRIVATE
  CUDA::cudart
  rmm::rmm
  raft::raft
  cuvs::cuvs
)


set_target_properties(_core PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


install(TARGETS _core
  LIBRARY DESTINATION proumap
  RUNTIME DESTINATION proumap
  ARCHIVE DESTINATION proumap
)