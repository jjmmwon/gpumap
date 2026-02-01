# set(_conda "$ENV{CONDA_PREFIX}")



# set(PROUMAP_CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/print_device.cu)

# message(STATUS "PROUMAP_CPP_SOURCES = ${PROUMAP_CPP_SOURCES}")
# list(LENGTH PROUMAP_CPP_SOURCES _n)
# message(STATUS "Number of sources: ${_n}")


# add_library(proumap_core STATIC ${PROUMAP_CPP_SOURCES})

# set_target_properties(proumap_core PROPERTIES
#   POSITION_INDEPENDENT_CODE ON
#   CUDA_SEPARABLE_COMPILATION ON
# )
# # BUILD_RPATH "${_conda}/lib;${_conda}/targets/x86_64-linux/lib"
#   # INSTALL_RPATH "${_conda}/lib;${_conda}/targets/x86_64-linux/lib"
#   # BUILD_WITH_INSTALL_RPATH ON

# target_include_directories(proumap_core
#   PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#   PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/src 
# )


# find_package(raft CONFIG REQUIRED)
# find_package(cuvs CONFIG REQUIRED)
# find_package(RMM CONFIG REQUIRED)

# target_link_libraries(proumap_core
#   PRIVATE
#     CUDA::cudart
#     raft::raft
#     cuvs::cuvs
#     rmm::rmm
# )

# install(TARGETS proumap_core
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin
# )

# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
#   DESTINATION include
# )

add_library(proumap_cpp INTERFACE)
target_include_directories(proumap_cpp INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)