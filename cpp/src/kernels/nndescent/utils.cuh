namespace gpumap::nndescent {

template <typename Idx_t, int MAX_K>
__device__ bool contains_id(const Idx_t *arr, int k, Idx_t v) {

#pragma unroll
  for (int i = 0; i < MAX_K; ++i) {
    if (i >= k)
      break;
    if (arr[i] == v) {
      return true;
    }
  }
  return false;
}

template <typename Idx_t> __device__ bool is_invalid_id(Idx_t id) {
  return id == std::numeric_limits<Idx_t>::max() || id < 0;
}

// L2 distance calculation with precomputed norms
template <typename Data_t, typename Idx_t>
__device__ float compute_l2_distance(const Data_t *__restrict__ data,
                                     Idx_t idx1, Idx_t idx2,
                                     size_t n_features) {

  size_t offset1 = static_cast<size_t>(idx1) * (n_features + 1);
  size_t offset2 = static_cast<size_t>(idx2) * (n_features + 1);

  // Get precomputed norms (stored in last position)
  float norm1 = static_cast<float>(data[offset1 + n_features]);
  float norm2 = static_cast<float>(data[offset2 + n_features]);

  // Compute dot product
  float dot = 0.0f;
  for (size_t i = 0; i < n_features; ++i) {
    dot += static_cast<float>(data[offset1 + i]) *
           static_cast<float>(data[offset2 + i]);
  }

  // L2 distance squared: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
  Data_t dist_sq = norm1 + norm2 - 2.0f * dot;
  return dist_sq;
}

// Insert a new neighbor maintaining sorted order
template <typename Data_t, typename Idx_t>
__device__ bool insert_neighbor_sorted(Idx_t *neighbors, Data_t *distances,
                                       int n_neighbors, Idx_t new_id,
                                       Data_t new_dist) {

  // Check if already exists
  for (int i = 0; i < n_neighbors; ++i) {
    if (neighbors[i] == new_id &&
        distances[i] != std::numeric_limits<Data_t>::max()) {
      return false;
    }
  }

  if (new_dist >= distances[n_neighbors - 1]) {
    return false;
  }

  // Find insertion position
  int insert_pos = n_neighbors - 1;
  for (int i = n_neighbors - 2; i >= 0; --i) {
    if (new_dist < distances[i]) {
      insert_pos = i;
    } else {
      break;
    }
  }

  // Shift and insert
  for (int i = n_neighbors - 1; i > insert_pos; --i) {
    neighbors[i] = neighbors[i - 1];
    distances[i] = distances[i - 1];
  }

  neighbors[insert_pos] = new_id;
  distances[insert_pos] = new_dist;

  return true;
}
} // namespace gpumap::nndescent