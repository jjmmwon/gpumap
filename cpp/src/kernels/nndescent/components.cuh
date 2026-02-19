#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <driver_types.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <sys/types.h>
#include <utility>
#include <vector>
#include <vector_types.h>

namespace gpumap::nndescent {

constexpr int MAX_NEIGHBORS = 32;
constexpr int NUM_SAMPLES = 32;

/**
 * @brief CPU-side Bloom filter used to deduplicate candidate edges per list.
 *
 * This filter keeps `num_sets_per_list` independent bitsets for each list.
 * A key is mapped to one set using `key % num_sets_per_list`, then inserted
 * and queried with `num_hashes` hashes inside that set. The implementation is
 * optimized for low-overhead membership checks and insertions, and may return
 * false positives but never false negatives.
 *
 * @tparam Index_t Integral key type used for node ids.
 */
template <typename Index_t> class BloomFilter {
public:
  BloomFilter(size_t n_instances, size_t num_sets_per_list, size_t num_hashes)
      : n_instances_(n_instances), num_sets_per_list_(num_sets_per_list),
        num_hashes_(num_hashes),
        bitsets_(n_instances_ * num_sets_per_list_ * num_bits_per_set_) {};

  void set_nrow(size_t nrow) { n_instances_ = nrow; }

  void clear() {
    if (is_cleared_)
      return;
#pragma omp parallel for
    for (size_t i = 0;
         i < n_instances_ * num_sets_per_list_ * num_bits_per_set_; ++i) {
      bitsets_[i] = 0;
    }
    is_cleared_ = true;
  }

  bool check(size_t list_id, Index_t key) const {
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

  void add(size_t list_id, Index_t key) {
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
  size_t n_instances_;
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

template <typename Data_t, typename Index_t = int> struct UpdateBuffers {
  UpdateBuffers() = delete;
  UpdateBuffers(const UpdateBuffers &) = delete;
  UpdateBuffers &operator=(const UpdateBuffers &) = delete;

  UpdateBuffers(raft::resources const &res, uint32_t n_instances,
                uint32_t n_neighbors)

      : bloom_filter{n_instances, 4, 4}, res(res), n_instances(n_instances),
        n_neighbors(n_neighbors),

        // Buffers used in device kernels
        l2_norms_{raft::make_device_vector<Data_t, size_t>(res, n_instances)},
        graph_buffer_{
            raft::make_device_matrix<InternalID_t, size_t, raft::row_major>(
                res, n_instances, MAX_NEIGHBORS)},
        dists_buffer_{raft::make_device_matrix<Data_t, size_t, raft::row_major>(
            res, n_instances, MAX_NEIGHBORS)},
        graph_host_buffer_{
            raft::make_pinned_matrix<InternalID_t, size_t, raft::row_major>(
                res, n_instances, MAX_NEIGHBORS)},
        dists_host_buffer_{
            raft::make_pinned_matrix<Data_t, size_t, raft::row_major>(
                res, n_instances, MAX_NEIGHBORS)},

        d_locks_{raft::make_device_vector<int, size_t>(res, n_instances)},
        d_dirty_flags_{raft::make_device_vector<int, size_t>(res, n_instances)},
        h_dirty_flags_{raft::make_pinned_vector<int, size_t>(res, n_instances)},

        h_rev_graph_new_{
            raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(
                res, n_instances, NUM_SAMPLES)},
        h_graph_old_{raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(
            res, n_instances, NUM_SAMPLES)},
        h_rev_graph_old_{
            raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(
                res, n_instances, NUM_SAMPLES)},

        d_list_sizes_new_{
            raft::make_device_vector<int2, size_t>(res, n_instances)},
        d_list_sizes_old_{
            raft::make_device_vector<int2, size_t>(res, n_instances)},

        // Buffers used in host side
        // (sample_graph/sample_graph_new/update_graph)
        h_graph{raft::make_host_matrix<InternalID_t, size_t, raft::row_major>(
            n_instances, n_neighbors)},
        h_dists{raft::make_host_matrix<Data_t, size_t, raft::row_major>(
            n_instances, n_neighbors)},
        h_graph_new{raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(
            res, n_instances, NUM_SAMPLES)},
        h_list_sizes_new{
            raft::make_pinned_vector<int2, size_t>(res, n_instances)},
        h_graph_old{raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(
            res, n_instances, NUM_SAMPLES)},
        h_list_sizes_old{
            raft::make_pinned_vector<int2, size_t>(res, n_instances)} {}

  raft::resources const &res;
  uint32_t n_instances;
  uint32_t n_neighbors;
  uint32_t n_internal_neighbors;

  // =========================================================================
  // Buffers used in device kernels. These are allocated once and reused across
  // iterations to avoid repeated allocation overhead.
  // =========================================================================
  raft::device_vector<Data_t, size_t> l2_norms_;
  raft::device_matrix<InternalID_t, size_t, raft::row_major> graph_buffer_;
  raft::device_matrix<Data_t, size_t, raft::row_major> dists_buffer_;

  raft::pinned_matrix<InternalID_t, size_t> graph_host_buffer_;
  raft::pinned_matrix<Data_t, size_t> dists_host_buffer_;

  raft::device_vector<int, size_t> d_locks_;
  raft::device_vector<int, size_t> d_dirty_flags_;
  raft::pinned_vector<int, size_t> h_dirty_flags_;

  raft::pinned_matrix<Index_t, size_t> h_rev_graph_new_;
  raft::pinned_matrix<Index_t, size_t> h_graph_old_;
  raft::pinned_matrix<Index_t, size_t> h_rev_graph_old_;

  raft::device_vector<int2, size_t> d_list_sizes_new_;
  raft::device_vector<int2, size_t> d_list_sizes_old_;

  // =========================================================================
  // Buffers used in host side. These are used for CPU-side processing and
  // DMA transfers to/from device.
  // =========================================================================

  raft::host_matrix<Index_t, size_t> h_graph;
  raft::host_matrix<Data_t, size_t> h_dists;
  raft::pinned_matrix<Index_t, size_t> h_graph_new;
  raft::pinned_vector<int2, size_t> h_list_sizes_new;

  raft::pinned_matrix<Index_t, size_t> h_graph_old;
  raft::pinned_vector<int2, size_t> h_list_sizes_old;
  BloomFilter<Index_t> bloom_filter;
};

template <typename Data_t, typename Idx_t, typename Index_t = int>
class UpdateAlgo {
public:
  UpdateAlgo() = delete;
  UpdateAlgo(const UpdateAlgo &) = delete;
  UpdateAlgo &operator=(const UpdateAlgo &) = delete;

  UpdateAlgo(UpdateBuffers<Data_t, Index_t> &buffers,
             std::vector<Idx_t> batch_indices,

             Data_t *d_input_data, Idx_t *d_neighbor_indices,
             Data_t *d_neighbor_distances, uint32_t data_block_size,
             uint32_t neighbor_indices_block_size,
             uint32_t neighbor_dists_block_size, uint32_t n_instances,
             uint32_t n_features, uint32_t n_neighbors,
             uint32_t n_inserted_instances, uint32_t max_iter = 30,
             float termination_threshold = 0.001f)
      : buffers_(buffers), update_counter_{0},
        batch_indices_(std::move(batch_indices)),

        d_input_data_(d_input_data), d_neighbor_indices_(d_neighbor_indices),
        d_neighbor_distances_(d_neighbor_distances),
        data_block_size_(data_block_size),
        neighbor_indices_block_size_(neighbor_indices_block_size),
        neighbor_dists_block_size_(neighbor_dists_block_size),

        n_instances_(n_instances), n_features_(n_features),
        n_neighbors_(n_neighbors), n_inserted_instances_(n_inserted_instances),
        max_iter_(max_iter), termination_threshold_(termination_threshold) {}

  std::vector<Idx_t> build(cudaStream_t const stream);
  void local_join(cudaStream_t const stream);
  void add_reverse_edges(Index_t *graph_ptr, Index_t *h_rev_graph_ptr,
                         Index_t *d_rev_graph_ptr, int2 *list_sizes,
                         cudaStream_t stream);

  void sample_graph_new();
  void sample_graph(bool sample_new);
  void update_graph(std::atomic<int64_t> &update_counter);
  void sort_lists();

private:
  UpdateBuffers<Data_t, int> &buffers_;
  std::atomic<int64_t> update_counter_;
  // Row mapping for partial update mode: local row i -> global row
  // batch_indices_[i].
  std::vector<Idx_t> batch_indices_;

  Data_t *d_input_data_;
  Idx_t *d_neighbor_indices_;
  Data_t *d_neighbor_distances_;

  uint32_t data_block_size_;
  uint32_t neighbor_indices_block_size_;
  uint32_t neighbor_dists_block_size_;

  uint32_t n_instances_;
  uint32_t n_features_;
  uint32_t n_neighbors_;
  uint32_t n_inserted_instances_;
  uint32_t max_iter_;
  float termination_threshold_;
};

} // namespace gpumap::nndescent
