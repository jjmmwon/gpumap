#pragma once
#include <algorithm>
#include <queue>
#include <tuple>
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

} // namespace gpumap
