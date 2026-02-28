#pragma once
#include <algorithm>
#include <array>
#include <queue>
#include <unordered_set>
#include <vector>

namespace gpumap {

// Base Queue template class with common functionality
template <typename T> class Queue {
protected:
  std::queue<T> queue_;

public:
  size_t size() const { return queue_.size(); }
  bool empty() const { return queue_.empty(); }

  void push(const T &item) { queue_.push(item); }

  void push_batch(const std::vector<T> &batch_items) {
    for (const T &item : batch_items) {
      queue_.push(item);
    }
  }

  void push_front(const std::vector<T> &batch_items) {
    std::queue<T> new_queue;
    for (const T &item : batch_items) {
      new_queue.push(item);
    }
    while (!queue_.empty()) {
      new_queue.push(queue_.front());
      queue_.pop();
    }
    queue_ = std::move(new_queue);
  }

  std::vector<T> pop_n(size_t n) {
    size_t actual_n = std::min(n, queue_.size());

    std::vector<T> batch;
    batch.reserve(actual_n);

    for (size_t i = 0; i < actual_n; ++i) {
      batch.push_back(queue_.front());
      queue_.pop();
    }

    return batch;
  }
};

struct InsertionQueue : public Queue<uint32_t> {
  InsertionQueue() = default;
};

// UpdateQueue deduplicates entries: pushing an index already in the queue is a
// no-op. This prevents redundant NND work and unbounded queue growth when
// neighbor cascading adds the same node multiple times.
struct UpdateQueue {
private:
  std::queue<uint32_t> queue_;
  std::unordered_set<uint32_t> in_queue_;

public:
  size_t size() const { return queue_.size(); }
  bool empty() const { return queue_.empty(); }

  void push(uint32_t item) {
    if (in_queue_.insert(item).second)
      queue_.push(item);
  }

  void push_batch(const std::vector<uint32_t> &items) {
    for (auto item : items)
      push(item);
  }

  std::vector<uint32_t> pop_n(size_t n) {
    size_t actual_n = std::min(n, queue_.size());
    std::vector<uint32_t> batch;
    batch.reserve(actual_n);
    for (size_t i = 0; i < actual_n; ++i) {
      uint32_t item = queue_.front();
      queue_.pop();
      in_queue_.erase(item);
      batch.push_back(item);
    }
    return batch;
  }
};

template <typename Idx_t = uint32_t>
using EdgeT = std::tuple<Idx_t, Idx_t, float, float, float>;

using Edge = EdgeT<uint32_t>;

// Multi-level embedding queue: divides [0, max_epoch] into N_LEVELS bins.
// Edges are routed by their current_epoch so that pop_n drains lower-epoch
// bins first, preserving temporal ordering across requeues.
struct MultiLevelEmbeddingQueue {
  static constexpr int N_LEVELS = 10;

  std::array<std::queue<Edge>, N_LEVELS> levels_;
  float max_epoch_;
  size_t total_size_ = 0;

  explicit MultiLevelEmbeddingQueue(float max_epoch) : max_epoch_(max_epoch) {}

  int get_level(float current_epoch) const {
    if (max_epoch_ <= 0.0f)
      return 0;
    float bin_size = max_epoch_ / static_cast<float>(N_LEVELS);
    int level = static_cast<int>(current_epoch / bin_size);
    return std::clamp(level, 0, N_LEVELS - 1);
  }

  size_t size() const { return total_size_; }
  bool empty() const { return total_size_ == 0; }

  std::vector<uint32_t> level_sizes() const {
    std::vector<uint32_t> sizes;
    sizes.reserve(N_LEVELS);
    for (int lvl = 0; lvl < N_LEVELS; ++lvl) {
      sizes.push_back(static_cast<uint32_t>(levels_[lvl].size()));
    }
    return sizes;
  }

  void push(const Edge &e) {
    int level = get_level(std::get<2>(e)); // current_epoch is field 2
    levels_[level].push(e);
    ++total_size_;
  }

  void push_batch(const std::vector<Edge> &edges) {
    for (const Edge &e : edges) {
      push(e);
    }
  }

  // Drains from the highest-epoch bin first.
  // Each level receives an equal base quota of actual_n / N_LEVELS items.
  // If a level has fewer items than its quota, the shortfall carries over to
  // the next lower level. A final sweep collects any items missed due to
  // integer-division remainder or carry that accumulated past level 0.
  std::vector<Edge> pop_n(size_t n, std::vector<Edge> &result) {
    size_t actual_n = std::min(n, total_size_);
    result.clear();
    result.reserve(actual_n);

    size_t base_quota = actual_n / N_LEVELS;
    size_t carry = 0;

    for (int lvl = N_LEVELS - 1; lvl >= 0; --lvl) {
      size_t effective_quota = base_quota + carry;
      carry = 0;

      size_t pulled = 0;
      while (!levels_[lvl].empty() && pulled < effective_quota) {
        result.push_back(levels_[lvl].front());
        levels_[lvl].pop();
        --total_size_;
        ++pulled;
      }

      carry =
          effective_quota - pulled; // unmet quota carries to next lower level
    }

    // Collect items missed due to integer-division remainder or carry from
    // level 0
    if (result.size() < actual_n) {
      size_t still_needed = actual_n - result.size();
      for (int lvl = N_LEVELS - 1; lvl >= 0 && still_needed > 0; --lvl) {
        while (!levels_[lvl].empty() && still_needed > 0) {
          result.push_back(levels_[lvl].front());
          levels_[lvl].pop();
          --total_size_;
          --still_needed;
        }
      }
    }

    return result;
  }
};

} // namespace gpumap
