#pragma once
#include <queue>

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

struct InsertionQueue : public Queue<size_t> {
  InsertionQueue(size_t n_instances) {
    for (size_t i = 0; i < n_instances; ++i) {
      push(i);
    }
  }
};

struct UpdateQueue : public Queue<size_t> {
  // Inherits all functionality from Queue<size_t>
  // No additional methods needed as all are provided by base class
};

struct EmbeddingQueue : public Queue<std::tuple<size_t, size_t, float>> {
  using Edge = std::tuple<size_t, size_t, float>; // (src, dst, weight)
  // Inherits all functionality from Queue<Edge>
  // No additional methods needed as all are provided by base class
};

} // namespace gpumap