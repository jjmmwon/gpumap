#pragma once
#include "rmm/cuda_stream_view.hpp"
#include <functional>
#include <queue>

namespace gpumap {
namespace queue {

struct Queue {

}

struct InsertionQueue : public Queue {
  vector<size_t> indices;

  InsertionQueue(size_t n_instances): Queue() {
    indices.reserve(n_instances);
    for(size_t i = n_instances-1; i-- > 0; ){
      indices.push_back(i);
    }
  }

  size_t size() const {
    return indices.size();
  }

  vector<size_t> pop_n(size_t n){
    size_t actual_n = std::min(n, indices.size());
    vector<size_t> batch_indices;
    batch_indices.reserve(actual_n);
    for(size_t i = 0; i < actual_n; ++i){
      batch_indices.push_back(indices.back());
      indices.pop_back();
    }
    return batch_indices;
  }
  
};

struct UpdateQueue : public Queue {
  vector<size_t> indices;

  void push_batch(const vector<size_t> &batch_indices){
    for(auto idx : batch_indices){
      indices.push_back(idx);
    }
  }
};

struct EmbeddingQueue : public Queue {

  priority_queue<tuple<size_t, size_t, float>, vector<tuple<size_t, size_t, float>>, greater<> pq; // (index, n_updates, priority)

};


// template <typename TaskType> struct Task {
//   TaskType type;

//   std::function<void()> run;

//   std::function<void()> on_complete;
// };

// template <typename TaskType> struct Queue {
//   std::queue<Task<TaskType>> tasks;
//   rmm::cuda_stream_view stream;

//   Queue(rmm::cuda_stream_view stream_) : stream(stream_) {}

//   void push(Task<TaskType> const &task) { tasks.push(task); }

//   Task<TaskType> pop() {
//     Task<TaskType> task = tasks.front();
//     tasks.pop();
//     return task;
//   }

//   bool is_empty() const { return tasks.empty(); };
//   size_t size() const { return tasks.size(); };

//   void execute_all() {};
// };
} // namespace queue
} // namespace gpumap