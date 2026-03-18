#pragma once

#include <cstddef>
#include <cstdint>
#include <driver_types.h>
#include <limits>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <sys/types.h>
#include <vector>
#include <vector_types.h>

namespace gpumap {

constexpr int MAX_NEIGHBORS = 32;
constexpr int MAX_CANDIDATES = 256;
constexpr int NUM_SAMPLES = 32;
constexpr int MAX_NUM_BI_SAMPLES = 64;
constexpr int SKEWED_MAX_NUM_BI_SAMPLES = 68;
constexpr int BLOCK_SIZE = 512;

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
    std::fill(bitsets_.begin(), bitsets_.end(), false);
    is_cleared_ = true;
  }

  bool check(size_t list_id, Index_t key) const {
    if (list_id >= n_instances_) {
      return false;
    }
    bool is_present = true;
    uint32_t hash = hash0(static_cast<uint32_t>(key));
    const size_t global_set_idx =
        list_id * num_bits_per_set_ * num_sets_per_list_ +
        (static_cast<size_t>(key) % num_sets_per_list_) * num_bits_per_set_;
    const size_t first_idx = global_set_idx + (hash % num_bits_per_set_);
    if (first_idx >= bitsets_.size()) {
      return false;
    }

    is_present &= bitsets_[first_idx];

    if (!is_present)
      return false;

    for (size_t i = 1; i < num_hashes_; ++i) {
      hash = hash + hash1(static_cast<uint32_t>(key));
      const size_t idx = global_set_idx + (hash % num_bits_per_set_);
      if (idx >= bitsets_.size()) {
        return false;
      }
      is_present &= bitsets_[idx];
      if (!is_present)
        return false;
    }
    return true;
  }

  void add(size_t list_id, Index_t key) {
    if (list_id >= n_instances_) {
      return;
    }
    if (is_cleared_)
      is_cleared_ = false;

    const uint32_t k = static_cast<uint32_t>(key);
    uint32_t hash = hash0(k);
    const size_t global_set_idx =
        list_id * num_bits_per_set_ * num_sets_per_list_ +
        (k % num_sets_per_list_) * num_bits_per_set_;

    const size_t first_idx = global_set_idx + (hash % num_bits_per_set_);
    if (first_idx >= bitsets_.size()) {
      return;
    }
    bitsets_[first_idx] = 1;
    for (size_t i = 1; i < num_hashes_; ++i) {
      hash = hash + hash1(k);
      const size_t idx = global_set_idx + (hash % num_bits_per_set_);
      if (idx >= bitsets_.size()) {
        return;
      }
      bitsets_[idx] = 1;
    }
  }

private:
  static uint32_t hash0(uint32_t value) {
    value *= 1103515245;
    value += 12345;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
  }

  static uint32_t hash1(uint32_t value) {
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
  inline __host__ __device__ InternalID_t(int id) : id_(id) {}

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

template <typename Data_t, typename Index_t = int> struct NNDBuffers {
  NNDBuffers() = delete;
  NNDBuffers(const NNDBuffers &) = delete;
  NNDBuffers &operator=(const NNDBuffers &) = delete;

  NNDBuffers(raft::resources const &res, uint32_t n_instances,
             uint32_t n_neighbors)

      : bloom_filter{n_instances, 4, 4}, res(res), n_instances(n_instances),
        n_neighbors(n_neighbors),

        l2_norms_{raft::make_device_vector<Data_t, size_t>(res, n_instances)},
        graph_buffer_{
            raft::make_device_matrix<InternalID_t, size_t, raft::row_major>(
                res, n_instances, MAX_NEIGHBORS)},
        dists_buffer_{raft::make_device_matrix<Data_t, size_t, raft::row_major>(
            res, n_instances, MAX_NEIGHBORS)},
        d_rev_graph_buffer_{
            raft::make_device_matrix<Index_t, size_t, raft::row_major>(
                res, n_instances, NUM_SAMPLES)},

        /// buffers used for storing candidates during insertion.
        candidate_buffer_{
            raft::make_device_matrix<int, size_t, raft::row_major>(
                res, n_instances, MAX_CANDIDATES)},

        graph_host_buffer_{
            raft::make_host_matrix<InternalID_t, size_t, raft::row_major>(
                n_instances, MAX_NEIGHBORS)},
        dists_host_buffer_{
            raft::make_host_matrix<Data_t, size_t, raft::row_major>(
                n_instances, MAX_NEIGHBORS)},

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
            raft::make_pinned_vector<int2, size_t>(res, n_instances)},

        reverse_cand_ids_{raft::make_device_matrix<int, size_t, raft::row_major>(
            res, n_instances, MAX_NEIGHBORS)},
        reverse_cand_dists_{
            raft::make_device_matrix<Data_t, size_t, raft::row_major>(
                res, n_instances, MAX_NEIGHBORS)},
        reverse_cand_counts_{
            raft::make_device_vector<int, size_t>(res, n_instances)} {}

  raft::resources const &res;
  uint32_t n_instances;
  uint32_t n_neighbors;

  raft::device_vector<Data_t, size_t> l2_norms_;

  // Working KNN graph on the GPU used throughout NND. Each row holds up to
  // MAX_NEIGHBORS entries stored as InternalID_t, which encodes the neighbor
  // id and a new/old flag in the sign bit (non-negative = new, negative = old).
  // Written by initialize_random_neighbors_kernel, forward_nndescent_kernel,
  // and local_join_kernel; read by sample_forward_candidates_kernel and
  // initialize_embeddings_kernel. Mirrored to graph_host_buffer_ after each
  // NND iteration.
  raft::device_matrix<InternalID_t, size_t, raft::row_major> graph_buffer_;
  // Distances corresponding to graph_buffer_. Always updated in lockstep with
  // graph_buffer_.
  raft::device_matrix<Data_t, size_t, raft::row_major> dists_buffer_;
  // Scratch device buffer used by add_reverse_edges() so reverse-edge
  // construction never aliases dists_buffer_.
  raft::device_matrix<Index_t, size_t, raft::row_major> d_rev_graph_buffer_;

  // Device buffer for candidate neighbors during insertion.
  raft::device_matrix<int, size_t, raft::row_major> candidate_buffer_;

  // CPU mirrors of graph_buffer_ and dists_buffer_.
  // Kept pageable (not pinned) to reduce locked-memory pressure under WSL2
  // and environments with low RLIMIT_MEMLOCK.
  raft::host_matrix<InternalID_t, size_t> graph_host_buffer_;
  raft::host_matrix<Data_t, size_t> dists_host_buffer_;

  raft::device_vector<int, size_t> d_locks_;
  raft::device_vector<int, size_t> d_dirty_flags_;
  raft::pinned_vector<int, size_t> h_dirty_flags_;

  // Pinned buffers passed directly to local_join_kernel as kernel arguments.
  // h_rev_graph_new_: reverse edges of the new-neighbor sample for the current
  //   iteration. Built on GPU by add_rev_edges_kernel, then copied here so the
  //   kernel can read them via zero-copy from device code.
  // h_graph_old_: copy of h_graph_old (the old-neighbor sample) transferred
  //   here each iteration so local_join_kernel can access it while the CPU
  //   thread simultaneously refills h_graph_old for the next iteration.
  // h_rev_graph_old_: reverse edges of the old-neighbor sample, built the same
  //   way as h_rev_graph_new_.
  raft::pinned_matrix<Index_t, size_t> h_rev_graph_new_;
  raft::pinned_matrix<Index_t, size_t> h_graph_old_;
  raft::pinned_matrix<Index_t, size_t> h_rev_graph_old_;

  // Per-node sampled list sizes for the current NND iteration, stored on GPU.
  // int2::x = number of forward edges in the sample (filled by sample_graph).
  // int2::y = number of reverse edges added by add_rev_edges_kernel.
  // Copied from h_list_sizes_new / h_list_sizes_old before each local_join,
  // then passed to add_rev_edges_kernel (which increments .y atomically) and
  // to local_join_kernel (which reads both fields to bound loop iterations).
  raft::device_vector<int2, size_t> d_list_sizes_new_;
  raft::device_vector<int2, size_t> d_list_sizes_old_;

  // CPU-only authoritative graph maintained across all NND iterations.
  // Stores InternalID_t so the new/old flag survives across calls to
  // sample_graph(), which reads each entry's flag to classify it as new or old
  // and calls mark_old() on new entries in-place. Not used for GPU transfers.
  raft::host_matrix<InternalID_t, size_t> h_graph;
  // Distances corresponding to h_graph.
  raft::host_matrix<Data_t, size_t> h_dists;

  // Pinned buffer holding the "new" neighbor sample for the current iteration.
  // Filled by sample_graph() (up to NUM_SAMPLES entries per node). Passed
  // directly to local_join_kernel as a kernel argument and to
  // add_reverse_edges() to build h_rev_graph_new_. int2::x of
  // h_list_sizes_new records the number of valid entries.
  raft::pinned_matrix<Index_t, size_t> h_graph_new;
  raft::pinned_vector<int2, size_t> h_list_sizes_new;

  // Pinned buffer holding the "old" neighbor sample for the current iteration.
  // Filled by sample_graph() from h_graph entries whose new/old flag is old.
  // At the start of each iteration, copied into h_graph_old_ (which is passed
  // to local_join_kernel) so the CPU thread can refill h_graph_old for the
  // next iteration in parallel with the running GPU kernel.
  raft::pinned_matrix<Index_t, size_t> h_graph_old;
  raft::pinned_vector<int2, size_t> h_list_sizes_old;

  // Pre-allocated scratch buffers for update_reverse_neighbors, sized
  // [n_instances × MAX_NEIGHBORS] and [n_instances], to avoid per-call
  // cudaMallocAsync inside the insertion iteration loop.
  raft::device_matrix<int, size_t, raft::row_major> reverse_cand_ids_;
  raft::device_matrix<Data_t, size_t, raft::row_major> reverse_cand_dists_;
  raft::device_vector<int, size_t> reverse_cand_counts_;

  BloomFilter<Index_t> bloom_filter;
};

} // namespace gpumap
