#include <cstddef>
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
  InternalID_t *h_indices;

  size_t n_row;
  size_t n_neighbors;
  size_t n_inserted_instances;
  size_t n_internal_neighbors;
  size_t num_samples; // number of samples of new/old neighbors to consider
                      // during graph update
  size_t seed;

  raft::host_matrix<Data_t, size_t, raft::row_major>
      h_dists; // 호스트 측 거리 저장

  raft::pinned_matrix<Idx_t, size_t>
      h_indices_new; // 새로운 그래프 연결 정보 (pinned 메모리)
  raft::pinned_vector<int2, size_t>
      h_list_sizes_new; // 새 그래프의 forward/reverse edge 개수

  raft::pinned_matrix<Idx_t, size_t>
      h_indices_old; // 기존 그래프 연결 정보 (pinned 메모리)
  raft::pinned_vector<int2, size_t>
      h_list_sizes_old;            // 기존 그래프의 forward/reverse edge 개수
  BloomFilter<Idx_t> bloom_filter; // 중복 에지 제거용 Bloom 필터

  NNDGraph(const NNDGraph &) = delete;
  NNDGraph &operator=(const NNDGraph &) = delete;
  NNDGraph(raft::resources const &res, const size_t n_row,
           const size_t n_neighbors, const size_t n_internal_neighbors,
           const size_t n_inserted_instances, const size_t num_samples,
           const size_t seed);

  /**
   * @brief Samples new neighbors from the candidate list and adds them to the
   * new graph.
   *
   * Filters new neighbors using a bloom filter to avoid duplicates, marks them
   * as old, and populates the new graph list up to num_samples entries.
   *
   * @param new_neighbors Array of new candidate neighbors with their IDs
   * @param width Width of the new_neighbors array (number of candidates per
   * row)
   */
  void sample_graph_new(InternalID_t *new_neighbors, const size_t width);

  /**
   * @brief Separates neighbors into old and new lists based on their update
   * status.
   *
   * Iterates through internal neighbors and distributes them into old_graph and
   * new_graph lists, respecting the num_samples capacity limit for each list.
   *
   * @param sample_new If true, also samples new neighbors; if false, only
   * samples old neighbors
   */
  void sample_graph(bool sample_new);

  /**
   * @brief Updates the internal graph with new neighbors and their distances.
   *
   * Inserts new neighbors into the sorted neighbor list if their distances are
   * better than the current worst distance, and increments the update counter
   * when successful. Skips self-loops and neighbors that are worse than the
   * current worst neighbor.
   *
   * @param new_neighbors Array of new neighbor IDs to consider for insertion
   * @param new_dists Array of distances corresponding to new neighbors
   * @param width Number of new neighbors per row
   * @param update_counter Atomic counter tracking the number of successful
   * updates
   */
  void update_graph(const InternalID_t *new_neighbors, const Data_t *new_dists,
                    const size_t width, std::atomic<int64_t> &update_counter);
  void clear();
  ~NNDGraph();
};

template <typename Data_t, typename Idx_t> class NNDUpdate {
public:
  NNDUpdate(raft::resources const &res, std::vector<size_t> batch_indices,
            Data_t *input_data, Idx_t *neighbor_indices,
            Data_t *neighbor_distances, Idx_t data_block_size,
            Idx_t neighbor_indices_block_size, Idx_t neighbor_dists_block_size,
            BloomFilter<Idx_t> &bloom_filter, size_t n_features,
            size_t n_neighbors, size_t n_internal_neighbors,
            size_t n_inserted_instances, size_t num_samples,
            float termination_threshold = 0.001, size_t max_iterations = 100,
            size_t seed = 42);

  NNDUpdate(const NNDUpdate &) = delete;
  NNDUpdate &operator=(const NNDUpdate &) = delete;

  void build();
  ~NNDUpdate() = default;

  void reset(raft::resources const &res);

private:
  void initialize();
  void add_reverse_edges(Idx_t *graph_ptr, Idx_t *h_rev_graph_ptr,
                         Idx_t *d_rev_graph_ptr, int2 *list_sizes,
                         cudaStream_t stream = 0);

  void local_join(cudaStream_t stream = 0);
  void update_neighbor_info();

  raft::resources const &res_;
  NNDGraph<Data_t, Idx_t> graph_;
  std::atomic<int64_t> update_counter_;
  float termination_threshold_;
  size_t max_iterations_ = 100;

  std::vector<size_t> batch_indices_;
  Data_t *d_input_data_; // a.k.a. d_data_
  Idx_t *d_neighbor_indices_;
  Data_t *d_neighbor_distances_;
  Idx_t data_block_size_;
  Idx_t neighbor_indices_block_size_;
  Idx_t neighbor_dists_block_size_;

  BloomFilter<Idx_t> &bloom_filter_;

  size_t n_row_;
  size_t n_features_;
  size_t n_neighbors_;
  size_t n_internal_neighbors_;
  size_t n_inserted_instances_;

  size_t num_samples_;

  size_t seed_;

  raft::device_matrix<InternalID_t, size_t, raft::row_major>
      indices_buffer_; // 디바이스에서 계산된 그래프 이웃 정보
  raft::device_matrix<Data_t, size_t, raft::row_major>
      dists_buffer_; // 디바이스에서 계산된 거리 정보

  raft::pinned_matrix<InternalID_t, size_t>
      indices_host_buffer_; // 호스트와 디바이스 간 빠른 전송을 위한 고정
                            // 메모리의 그래프 버퍼
  raft::pinned_matrix<Data_t, size_t>
      dists_host_buffer_; // 호스트와 디바이스 간 빠른 전송을 위한 고정 메모리의
                          // 거리 버퍼

  raft::device_vector<int, size_t>
      d_locks_; // 디바이스에서 동시성 제어용 락 벡터

  raft::pinned_matrix<Idx_t, size_t>
      h_rev_indices_new_; // 새 그래프의 역방향 에지 정보 (pinned 메모리)
  raft::pinned_matrix<Idx_t, size_t>
      h_indices_old_; // 기존 그래프의 정방향 에지 정보 (pinned 메모리)
  raft::pinned_matrix<Idx_t, size_t>
      h_rev_indices_old_; // 기존 그래프의 역방향 에지 정보 (pinned 메모리)

  // int2.x는 정방향 에지 개수, int2.y는 역방향 에지 개수
  raft::device_vector<int2, size_t>
      d_list_sizes_new_; // 새 그래프의 각 노드별 에지 개수 (디바이스)
  raft::device_vector<int2, size_t>
      d_list_sizes_old_; // 기존 그래프의 각 노드별 에지 개수 (디바이스)
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