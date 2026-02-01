# cmake_minimum_required(VERSION 3.18)

# # set(CMAKE_CUDA_ARCHITECTURES native)
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
# set(CMAKE_BUILD_RPATH "$ENV{CONDA_PREFIX}/lib")
# set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
# set(CMAKE_CUDA_ARCHITECTURES 86)

# project(proumap LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(CUDAToolkit REQUIRED)

# # add_subdirectory(cpp)
# add_subdirectory(python)

cmake_minimum_required(VERSION 3.24)

project(proumap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA arch 비어있을 때만 기본값
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_subdirectory(cpp)
add_subdirectory(python)