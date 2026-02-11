#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace gpumap {

template <typename Idx_t>
__device__ __forceinline__ bool is_neighbor(const Idx_t *neighbors, int n_neighbors, Idx_t dst) {
	for (int i = 0; i < n_neighbors; ++i) {
		if (neighbors[i] == dst) {
			return true;
		}
	}
	return false;
}

template <typename Idx_t>
__global__ void filter_edges_by_neighbors_kernel(
		const Idx_t *__restrict__ srcs, const Idx_t *__restrict__ dsts,
		Idx_t *__restrict__ out_srcs, Idx_t *__restrict__ out_dsts,
		size_t n_edges, const Idx_t *__restrict__ neighbor_indices,
		int n_neighbors, unsigned long long *__restrict__ out_count) {

	const size_t tid = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
	if (tid >= n_edges) {
		return;
	}

	const Idx_t src = srcs[tid];
	const Idx_t dst = dsts[tid];
	const Idx_t *my_neighbors = neighbor_indices + src * n_neighbors;

	if (!is_neighbor(my_neighbors, n_neighbors, dst)) {
		return;
	}

	const unsigned long long out_idx = atomicAdd(out_count, 1ULL);
	out_srcs[out_idx] = src;
	out_dsts[out_idx] = dst;
}

template <typename Idx_t>
void filter_edges_by_neighbors(const Idx_t *srcs, const Idx_t *dsts,
															 Idx_t *out_srcs, Idx_t *out_dsts,
															 size_t n_edges,
															 const Idx_t *neighbor_indices,
															 int n_neighbors, unsigned long long *out_count,
															 cudaStream_t stream) {

	const int threads_per_block = 256;
	const int blocks = static_cast<int>((n_edges + threads_per_block - 1) /
																			threads_per_block);

	cudaMemsetAsync(out_count, 0, sizeof(unsigned long long), stream);

	filter_edges_by_neighbors_kernel<Idx_t>
			<<<blocks, threads_per_block, 0, stream>>>(
					srcs, dsts, out_srcs, out_dsts, n_edges, neighbor_indices,
					n_neighbors, out_count);
}

} // namespace gpumap
