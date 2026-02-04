#pragma once

#include "storage/storage_manager.cuh"
#include "stroage/matrix2D.cuh"

namespace proumap {
struct UMAPConfig {
  int n_neighbors = 15;
  int n_components = 2;
  float min_dist = 0.1f;
  int n_epochs = 200;
  float learning_rate = 1.0f;
  bool verbose = false;
};

class ProUMAP {
public:
  UMAPConfig config;
  int max_instances = 1000000;
  int batch_size_hint = 1024;

  ProUMAP() = default;
  ProUMAP(const UMAPConfig &config_, int max_instances_, int batch_size_hint_)
      : config(config_), max_instances(max_instances_),
        batch_size_hint(batch_size_hint_) {}

  // Move-only semantics for
  ProUMAP(const ProUMAP &) = delete;
  ProUMAP &operator=(const ProUMAP &) = delete;
  ProUMAP(ProUMAP &&) = default;
  ProUMAP &operator=(ProUMAP &&) = default;

  ~ProUMAP() = default;
};
} // namespace proumap