#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <vector_types.h>

namespace gpumap::nndescent {

/**
 * @brief CPU-side Bloom filter used to deduplicate candidate edges per list.
 *
 * This filter keeps `num_sets_per_list` independent bitsets for each list.
 * A key is mapped to one set using `key % num_sets_per_list`, then inserted
 * and queried with `num_hashes` hashes inside that set. The implementation is
 * optimized for low-overhead membership checks and insertions, and may return
 * false positives but never false negatives.
 *
 * @tparam Idx_t Integral key type used for node ids.
 */

template <typename Idx_t = size_t> class BloomFilter {
public:
  BloomFilter(size_t n_row, size_t num_sets_per_list, size_t num_hashes)
      : nrow_(n_row), num_sets_per_list_(num_sets_per_list),
        num_hashes_(num_hashes),
        bitsets_(nrow_ * num_sets_per_list_ * num_bits_per_set_) {};

  void set_nrow(size_t nrow) { nrow_ = nrow; }

  void clear() {
    if (is_cleared_)
      return;
#pragma omp parallel for
    for (size_t i = 0; i < nrow_ * num_sets_per_list_ * num_bits_per_set_;
         ++i) {
      bitsets_[i] = 0;
    }
    is_cleared_ = true;
  }

  bool check(size_t list_id, Idx_t key) const {
    bool is_present = true;
    uint32_t hash = hash0(static_cast<uint32_t>(key));
    size_t global_set_idx = list_id * num_sets_per_list_ +
                            (static_cast<size_t>(key) % num_sets_per_list_);

    is_present &= bitsets_[global_set_idx + hash % num_bits_per_set_];

    if (!is_present)
      return false;

    for (size_t i = 1; i < num_hashes_; ++i) {
      hash = hash + hash1(static_cast<uint32_t>(key));
      is_present &= bitsets_[global_set_idx + hash % num_bits_per_set_];
      if (!is_present)
        return false;
    }
    return true;
  }

  void add(size_t list_id, Idx_t key) {
    if (is_cleared_)
      is_cleared_ = false;

    const uint32_t k = static_cast<uint32_t>(key);
    uint32_t hash = hash0(k);
    const size_t global_set_idx =
        list_id * num_bits_per_set_ * num_sets_per_list_ +
        (k % num_sets_per_list_) * num_bits_per_set_;

    bitsets_[global_set_idx + (hash % num_bits_per_set_)] = 1;
    for (size_t i = 1; i < num_hashes_; ++i) {
      hash = hash + hash1(k);
      bitsets_[global_set_idx + (hash % num_bits_per_set_)] = 1;
    }
  }

private:
  uint32_t hash0(uint32_t value) {
    value *= 1103515245;
    value += 12345;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
  }

  uint32_t hash1(uint32_t value) {
    value *= 1664525;
    value += 1013904223;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
  }

  static constexpr int num_bits_per_set_ = 512;
  bool is_cleared_ = true;

  std::vector<bool> bitsets_;
  size_t nrow_;
  size_t num_sets_per_list_;
  size_t num_hashes_;
};

struct InternalID_t {
private:
  int id_{std::numeric_limits<int>::max()};

public:
  InternalID_t(int id) : id_(id) {}

  inline __host__ __device__ bool is_new() const { return id_ >= 0; }
  inline __host__ __device__ int &id_with_flag() { return id_; }
  inline __host__ __device__ int id() const {
    if (is_new())
      return id_;
    return -id_ - 1;
  }
  inline __host__ __device__ void mark_old() {
    if (id_ >= 0)
      id_ = -id_ - 1;
  }
  inline __host__ __device__ bool operator==(const InternalID_t &other) const {
    return id() == other.id();
  }
};

template <typename Data_t, typename Idx_t> struct NNDGraph {
  raft::resources const &res;
  static constexpr int segment_size = 32;
  InternalID_t *h_graph;

  size_t n_row;
  size_t n_neighbors;
  size_t n_inserted_instances_;
  size_t n_internal_neighbors;
  size_t num_samples;

  raft::host_matrix<Data_t, size_t, raft::row_major> h_dists;

  raft::pinned_matrix<Idx_t, size_t> h_graph_new;
  raft::pinned_vector<int2, size_t> h_list_sizes_new;

  raft::pinned_matrix<Idx_t, size_t> h_graph_old;
  raft::pinned_vector<int2, size_t> h_list_sizes_old;
  BloomFilter<Idx_t> bloom_filter;

  NNDGraph(const NNDGraph &) = delete;
  NNDGraph &operator=(const NNDGraph &) = delete;
  NNDGraph(raft::resources const &res, const size_t n_row,
           const size_t n_neighbors, const size_t n_internal_neighbors,
           const size_t n_inserted_instances, const size_t num_samples);

  void sample_graph_new(InternalID_t *new_neighbors, const size_t width);
  void sample_graph(bool sample_new);
  void update_graph(const InternalID_t *new_neighbors, const Data_t *new_dists,
                    const size_t width, std::atomic<int64_t> &update_counter);
  void clear();
  ~NNDGraph();
};

template <typename Data_t, typename Idx_t, int NUM_SAMPLES> class NNDUpdate {
public:
  NNDUpdate(raft::resources const &res);
  NNDUpdate(const NNDUpdate &) = delete;
  NNDUpdate &operator=(const NNDUpdate &) = delete;

  template <typename DistEpilogue_t = raft::identity_op>
  void build(Data_t *data, const Idx_t nrow, Idx_t *output_graph,
             Data_t *output_distances);
  ~NNDUpdate() = default;

  void reset(raft::resources const &res);

private:
  void add_reverse_edges(Idx_t *graph_ptr, Idx_t *h_rev_graph_ptr,
                         Idx_t *d_rev_graph_ptr, int2 *list_sizes,
                         cudaStream_t stream = 0);

  void local_join(cudaStream_t stream = 0);

  raft::resources const &res;
  NNDGraph<Data_t, Idx_t> *graph_;
  std::atomic<int64_t> update_counter_;

  size_t n_row;
  size_t n_features;

  raft::device_matrix<Data_t, size_t, raft::row_major> d_data;
  raft::device_matrix<InternalID_t, size_t, raft::row_major> graph_buffer_;
  raft::device_matrix<Data_t, size_t, raft::row_major> dists_buffer_;

  raft::pinned_matrix<InternalID_t, size_t> graph_host_buffer_;
  raft::pinned_matrix<Data_t, size_t> dists_host_buffer_;

  raft::device_vector<int, size_t> d_locks_;

  raft::pinned_matrix<Idx_t, size_t> h_rev_graph_new_;
  raft::pinned_matrix<Idx_t, size_t> h_graph_old_;
  raft::pinned_matrix<Idx_t, size_t> h_rev_graph_old_;

  // int2.x is the number of forward edges, int2.y is the number of reverse
  // edges
  raft::device_vector<int2, size_t> d_list_sizes_new_;
  raft::device_vector<int2, size_t> d_list_sizes_old_;
};
} // namespace gpumap::nndescent

//   void update_graph(const InternalID_t *new_neighbors, const Data_t
//   *new_dists,
//                     std::atomic<int64_t> &update_counter) {
//     for (size_t i = 0; i < batch_indices_.size(); ++i) {
//       size_t global_row_idx = batch_indices_[i] *
//       neighbor_indices_block_size_;

//       auto h_graph = h_graph.data_handle() + i * n_internal_neighbors_;
//       auto dist_list = h_dists.data_handle() + i * n_internal_neighbors_;

//       Data_t worst = dist_list[n_internal_neighbors_ - 1];

//       for (size_t j = 0; j < NUM_SAMPLES; ++j) {
//         InternalID_t new_id = new_neighbors[i * NUM_SAMPLES + j];
//         Data_t new_dist = new_dists[i * NUM_SAMPLES + j];

//         if (new_dist == std::numeric_limits<Data_t>::max())
//           break;
//         if (static_cast<size_t>(new_id.id()) ==
//             batch_indices_[i]) // skip self-loops
//           continue;

//         if (new_dist >= worst)
//           break;

//         if (utils::insert_neighbor_sorted<Data_t, Idx_t>(
//                 reinterpret_cast<Idx_t *>(h_graph), dist_list,
//                 static_cast<int>(n_internal_neighbors_),
//                 static_cast<Idx_t>(new_id.id()), new_dist)) {
//           worst = dist_list[n_internal_neighbors_ - 1];
//           update_counter++;
//         }
//       }
//     }