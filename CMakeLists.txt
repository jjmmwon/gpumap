cmake_minimum_required(VERSION 3.24)

# ── Conda env auto-detection ─────────────────────────────────────────────────
# Must happen before project() so compiler detection picks up these values.
if(DEFINED ENV{CONDA_PREFIX})
  set(_conda "$ENV{CONDA_PREFIX}")
  message(STATUS "[gpumap] conda env: ${_conda}")

  # Prepend conda prefix so find_package resolves rmm/raft/cuvs without
  # requiring CMAKE_PREFIX_PATH to be set manually.
  list(PREPEND CMAKE_PREFIX_PATH "${_conda}")

  # Use conda g++ as CXX compiler to get conda's GCC ABI, headers, and OpenMP.
  if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${_conda}/bin/x86_64-conda-linux-gnu-g++"
        CACHE FILEPATH "CXX compiler" FORCE)
  endif()

  # Use conda nvcc unless the caller already set CUDACXX or CMAKE_CUDA_COMPILER.
  if(NOT DEFINED ENV{CUDACXX} AND NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${_conda}/bin/nvcc"
        CACHE FILEPATH "CUDA compiler" FORCE)
  endif()

  # Use conda g++ as nvcc host compiler to match the GCC ABI of conda headers.
  if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER
        "${_conda}/bin/x86_64-conda-linux-gnu-g++"
        CACHE FILEPATH "CUDA host compiler" FORCE)
  endif()
endif()

# ── ccache ────────────────────────────────────────────────────────────────────
# Must be set before project() to register as a compiler launcher.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER  "${CCACHE_PROGRAM}" CACHE STRING "" FORCE)
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "" FORCE)
  message(STATUS "[gpumap] ccache: ${CCACHE_PROGRAM}")
endif()

# Compile for the GPU actually installed on this machine.
# Override with -DCMAKE_CUDA_ARCHITECTURES=xx if cross-compiling.
# Must be set before project()/enable_language(CUDA) -- that call gives
# CMAKE_CUDA_ARCHITECTURES its own built-in (non-native) default if unset,
# after which this guard's "NOT DEFINED" check no longer triggers.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ── Project ───────────────────────────────────────────────────────────────────
project(gpumap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(cpp)
add_subdirectory(python)
