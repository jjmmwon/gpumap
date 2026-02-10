#pragma once

#include "details/nndescent.cuh"

namespace gumap {
namespace nndescent {

/**
 * @brief Kernel to initialize random neighbors for each instance.
 * @tparam Idx_t Index type (e.g., size_t, int)
 * @param neighbor_indices Pointer to uninitialized neighbor indices matrix
 * @param n_instances Number of instances to initialize
 * @param n_neighbors Number of neighbors per instance
 * @param total_instances Total number of instances in the dataset
 * @param seed Seed for random number generation
 */

template <typename Idx_t>
__global__ void
initialize_random_neighbors_kernel(Idx_t *neighbor_indices, size_t n_instances,
                                   size_t n_neighbors, size_t total_instances,
                                   unsigned long long seed);

/**
 * @brief Kernel to calculate squared L2 norms for each instance to avoid
 * repetitve distance calculations.
 * kernel:<<< n_instances, threads_per_block, threads*sizeof(Data_t), stream >>>
 * @tparam Data_t Data type of the input matrix (e.g., float, double)
 * @tparam Idx_t Index type (e.g., size_t, int)
 * @param input_data Pointer to the newly added input data matrix
 * @param n_instances Number of instances to
 * @param n_features Number of features per instance
 */
template <typename Data_t, typename Idx_t>
__global__ void
calculate_l2_norms_sqr_kernel(const Data_t *input_data, // 입력 데이터 매트릭스
                              size_t n_instances,       // 인스턴스 개수
                              size_t n_features         // 특성 개수
);

} // namespace nndescent
} // namespace gumap
