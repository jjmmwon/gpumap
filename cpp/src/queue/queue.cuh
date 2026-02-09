#pragma once
#include "rmm/cuda_stream_view.hpp"
#include <functional>
#include <queue>

namespace proumap {
namespace queue {

template <typename TaskType> struct Task {
  TaskType type;

  std::function<void()> run;

  std::function<void()> on_complete;
};

template <typename TaskType> struct Queue {
  std::queue<Task<TaskType>> tasks;
  rmm::cuda_stream_view stream;

  Queue(rmm::cuda_stream_view stream_) : stream(stream_) {}

  void push(Task<TaskType> const &task) { tasks.push(task); }

  Task<TaskType> pop() {
    Task<TaskType> task = tasks.front();
    tasks.pop();
    return task;
  }

  bool is_empty() const { return tasks.empty(); };
  size_t size() const { return tasks.size(); };

  void execute_all() {};
};
} // namespace queue
} // namespace proumap