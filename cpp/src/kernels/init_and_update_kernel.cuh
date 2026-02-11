#include <vector>

namespace gpumap {

template <typename Data_t, typename Idx_t>
void init_and_update_kernel(
    Data_t *input_data, Idx_t *neighbor_indices, float *neighbor_distances,
    Data_t *embedding,

    Idx_t n_neighbors, size_t n_instances,

    const std::vector<size_t> &batch_indices,
    std::vector<size_t> &affected_indices,
    std::vector<std::tuple<size_t, size_t, float>> &new_edges,

    cudaStream_t *insertion_stream, cudaStream_t *update_stream,
    cudaStream_t *embedding_stream) {}

} // namespace gpumap