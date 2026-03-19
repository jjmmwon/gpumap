#pragma once

#include "gpumap/gpumap.cuh"

#include "kernels/embedding_kernel.cuh"
#include "kernels/insertion_kernel.cuh"
#include "kernels/update_kernel.cuh"

#include "queue.cuh"
#include "raft/core/device_mdspan.hpp"
#include "storage.cuh"
#include "utils/matrix_utils.cuh"
#include <algorithm>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <numeric>
#include <random>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

namespace gpumap {

using clock = std::chrono::high_resolution_clock;

struct GPUMAP::Impl {
  using Data_t = float;
  using Idx_t = uint32_t;

  GPUMAPParams params;

  Storage<Data_t, Idx_t> storage;

  cudaStream_t insertion_stream, update_stream, embedding_stream;

  InsertionQueue insertion_queue;
  UpdateQueue update_queue;

  static constexpr size_t EMBEDDING_MAX_EPOCH = 200;

  // Sequential cursor-based embedding state
  // Stores the id of the last processed instance; next run starts from
  // cursor+1. Initialized to SIZE_MAX so that (cursor+1)%n == 0 on the first
  // call.
  size_t embedding_cursor_ = SIZE_MAX;
  size_t embedding_sweeps_done_ = 0;

  // Recent-window tracking for initialize_random_neighbors.
  static constexpr int WINDOW_N_BATCHES = 3;
  std::deque<uint32_t> batch_start_history_;

  bool first_insertion = true;
  std::mt19937 rng_{42};

  Impl(const GPUMAPParams &params_)
      : params(params_), storage(params_.n_instances, params_.n_features,
                                 params_.n_neighbors, params_.n_components),
        insertion_queue() {
    initialize();
  }

  void initialize() {
    cudaStreamCreate(&insertion_stream);
    cudaStreamCreate(&update_stream);
    cudaStreamCreate(&embedding_stream);
  }

  template <typename... Args> void log(const char *fmt, Args... args) const {
    if (params.verbose) {
      fprintf(stderr, fmt, args...);
      fflush(stderr);
    }
  }

  void append_data(const Data_t *ptr, Idx_t n_new_instances) {

    if (storage.n_inserted_instances + n_new_instances > storage.n_instances) {
      throw std::runtime_error("Exceeding maximum number of rows in storage");
    }

    auto data_ptr = storage.data.data() +
                    static_cast<size_t>(storage.n_inserted_instances) *
                        storage.data_block_size;

    utils::append_host(data_ptr, ptr, n_new_instances, storage.n_features,
                       storage.data_block_size,
                       rmm::cuda_stream_view(insertion_stream));

    // Add newly inserted indices to the insertion queue
    Idx_t old_count = storage.n_inserted_instances;
    storage.n_inserted_instances += n_new_instances;

    std::vector<Idx_t> new_indices(n_new_instances);
    std::iota(new_indices.begin(), new_indices.end(), old_count);
    insertion_queue.push_batch(new_indices);
  }

  GPUMAPResult run(uint32_t n_insertion_ops, uint32_t n_update_ops,
                   uint32_t n_embedding_ops) {
    GPUMAPResult result{};
    log("[GPUMAP] run() start: insert=%u update=%u embed=%u inserted=%u\n",
        n_insertion_ops, n_update_ops, n_embedding_ops,
        storage.n_inserted_instances);

    if (storage.insertion_completed) {
      result.n_insertion_ops = 0;
      result.insertion_time = 0.0f;
    } else {
      log("[GPUMAP] insert start: n_ops=%u queue=%zu\n", n_insertion_ops,
          insertion_queue.size());
      auto start = clock::now();
      result.n_insertion_ops = insert(n_insertion_ops);
      cudaStreamSynchronize(insertion_stream);
      auto end = clock::now();
      result.insertion_time = std::chrono::duration<float>(end - start).count();
      storage.insertion_completed =
          insertion_queue.size() == 0 &&
          storage.n_inserted_instances == storage.n_instances;
      log("[GPUMAP] insert done: actual=%zu time=%.3fs\n",
          result.n_insertion_ops,
          std::chrono::duration<float>(end - start).count());
    }

    log("[GPUMAP] update queue size=%zu\n", update_queue.size());
    if (n_update_ops == 0 || storage.update_completed) {
      result.n_update_ops = 0;
      result.n_update_changed_edges = 0;
      result.update_time = 0.0f;
    } else {
      size_t n_changed_edges = 0;
      auto start = clock::now();
      result.n_update_ops = update(n_update_ops, n_changed_edges);
      cudaStreamSynchronize(update_stream);
      auto end = clock::now();
      result.n_update_changed_edges = n_changed_edges;
      result.update_time = std::chrono::duration<float>(end - start).count();
      log("[GPUMAP] update done: actual=%zu changed_edges=%zu time=%.3fs\n",
          result.n_update_ops, result.n_update_changed_edges,
          std::chrono::duration<float>(end - start).count());
    }

    result.embedding_pending_edges = storage.n_inserted_instances;
    log("[GPUMAP] embedding instances=%zu cursor=%zu\n",
        storage.n_inserted_instances, embedding_cursor_);
    if (n_embedding_ops == 0 || storage.n_inserted_instances == 0 ||
        storage.embedding_completed) {
      result.n_embedding_ops = 0;
      result.embedding_time = 0.0f;
    } else {
      auto start = clock::now();
      result.n_embedding_ops = embed(n_embedding_ops);
      cudaStreamSynchronize(embedding_stream);
      auto end = clock::now();
      result.embedding_time = std::chrono::duration<float>(end - start).count();
      log("[GPUMAP] embed done: actual=%zu time=%.3fs\n",
          result.n_embedding_ops,
          std::chrono::duration<float>(end - start).count());
    }

    result.n_inserted_instances = storage.n_inserted_instances;
    result.insertion_queue_size = insertion_queue.size();
    result.update_queue_size = update_queue.size();
    result.embedding_queue_size = result.embedding_pending_edges;
    result.insertion_completed = storage.insertion_completed;
    result.update_completed = storage.update_completed;
    result.embedding_completed = storage.embedding_completed;

    return result;
  }

  uint32_t insert(uint32_t n_ops) {
    if (!insertion_queue.size() || n_ops == 0) {
      return 0;
    }

    std::vector<Idx_t> batch_indices = insertion_queue.pop_n(n_ops);
    std::vector<Idx_t> affected_indices;
    std::vector<gpumap::Edge> new_edges;

    // Record this batch's start and maintain a sliding window of size W.
    batch_start_history_.push_back(batch_indices.front());
    if (static_cast<int>(batch_start_history_.size()) > WINDOW_N_BATCHES)
      batch_start_history_.pop_front();
    const uint32_t window_start = batch_start_history_.front();

    run_insertion<Data_t, Idx_t>(
        storage.nnd_buffer, batch_indices, storage.data.data(),
        storage.data_block_size, storage.n_features, storage.n_neighbors,
        storage.n_inserted_instances,
        /*max_iter=*/5, /*seed=*/42, insertion_stream, affected_indices,
        new_edges, storage.embedding.data(), storage.n_components,
        window_start);

    update_queue.push_batch(affected_indices);

    return static_cast<uint32_t>(batch_indices.size());
  }

  uint32_t update(uint32_t n_ops, size_t &n_changed_edges_out) {
    n_changed_edges_out = 0;
    if (n_ops == 0 || storage.update_completed ||
        storage.n_inserted_instances == 0)
      return 0;

    constexpr float termination_sigma = 0.001f;
    const size_t convergence_threshold = static_cast<size_t>(
        termination_sigma * static_cast<float>(storage.n_instances));

    uint32_t left_ops = n_ops;

    while (left_ops > 0) {
      std::vector<Idx_t> batch_indices;

      if (static_cast<size_t>(left_ops) >= storage.n_inserted_instances) {
        update_queue.pop_n(update_queue.size());
        batch_indices.resize(storage.n_inserted_instances);
        std::iota(batch_indices.begin(), batch_indices.end(), 0);
      } else if (static_cast<size_t>(left_ops) >= update_queue.size()) {
        batch_indices = update_queue.pop_n(update_queue.size());
        std::unordered_set<Idx_t> seen(batch_indices.begin(),
                                       batch_indices.end());

        std::uniform_int_distribution<Idx_t> dist(
            0, static_cast<Idx_t>(storage.n_inserted_instances - 1));

        auto start_offset = dist(rng_);
        while (batch_indices.size() < static_cast<size_t>(left_ops)) {
          Idx_t candidate = (start_offset++) % storage.n_inserted_instances;
          if (seen.insert(candidate).second) {
            batch_indices.push_back(candidate);
          }
        }
      } else {
        batch_indices = update_queue.pop_n(left_ops);
      }

      if (batch_indices.empty())
        break;

      std::vector<Idx_t> affected_indices;
      std::vector<gpumap::Edge> new_edges;

      size_t n_changed = update_kernel<Data_t, Idx_t>(
          storage.nnd_buffer, batch_indices, affected_indices, new_edges,
          update_stream,

          storage.data.data(), storage.neighbor_indices.data(),
          storage.neighbor_dists.data(), storage.data_block_size,
          storage.neighbor_indices_block_size,
          storage.neighbor_dists_block_size,

          storage.n_features, storage.n_neighbors,
          storage.n_inserted_instances);

      n_changed_edges_out += new_edges.size();
      left_ops -= batch_indices.size();

      // Always propagate affected nodes and their 1-hop neighbors into the
      // update queue regardless of convergence.
      update_queue.push_batch(affected_indices);

      const auto *h_graph = storage.nnd_buffer.h_graph.data_handle();
      std::vector<Idx_t> cascade_indices;
      cascade_indices.reserve(affected_indices.size() * storage.n_neighbors);
      for (auto idx : affected_indices) {
        const auto *nbrs =
            h_graph + static_cast<size_t>(idx) * storage.n_neighbors;
        for (uint32_t j = 0; j < storage.n_neighbors; ++j) {
          int nid = nbrs[j].id();
          if (nid >= 0 &&
              static_cast<size_t>(nid) < storage.n_inserted_instances)
            cascade_indices.push_back(static_cast<Idx_t>(nid));
        }
      }
      update_queue.push_batch(cascade_indices);

      // Convergence: KNN changes dropped below threshold AND all instances
      // have been inserted (no more data will arrive to disturb the graph).
      if (update_queue.size() < convergence_threshold &&
          storage.n_inserted_instances == storage.n_instances) {
        log("[GPUMAP] update done: n_changed=%zu < threshold=%zu, "
            "all %zu instances inserted\n",
            n_changed, convergence_threshold, storage.n_instances);
        storage.update_completed = true;
        break;
      }
    }

    return n_ops - left_ops;
  }

  uint32_t embed(uint32_t n_ops) {
    if (storage.n_inserted_instances == 0 || n_ops == 0)
      return 0;

    const size_t n = storage.n_inserted_instances;
    const size_t n_ops_sz = static_cast<size_t>(n_ops);

    // Determine the two (possibly one) sequential ranges to embed.
    // Full sweep if n_ops >= n_inserted_instances.
    size_t range1_start, range1_end, range2_end;
    if (n_ops_sz >= n) {
      range1_start = 0;
      range1_end = n;
      range2_end = 0;
      embedding_cursor_ = n - 1;
      embedding_sweeps_done_++;
      if (storage.update_completed) {
        if (embedding_sweeps_done_ > 5) {
          storage.embedding_completed = true;
        }
      }
    } else {
      const size_t start = (embedding_cursor_ + 1) % n;
      range1_start = start;
      range1_end = std::min(start + n_ops_sz, n);
      size_t wrap = n_ops_sz - (range1_end - range1_start);
      if (storage.update_completed) {
        if (embedding_sweeps_done_ > 5) {
          wrap = 0;
          storage.embedding_completed = true;
        }
      }
      range2_end = wrap;
      if (range1_end == n) {
        embedding_cursor_ = (wrap > 0) ? wrap - 1 : n - 1;
        embedding_sweeps_done_++;
      } else {
        embedding_cursor_ = range1_end - 1;
      }
    }

    const size_t sel_count =
        (range1_end - range1_start) + range2_end; // range2: [0, range2_end)

    // Build srcs/dsts on host from NND graph (kernel handles invalid indices).
    const auto *h_graph = storage.nnd_buffer.h_graph.data_handle();
    const size_t max_edges = sel_count * storage.n_neighbors;
    std::vector<Idx_t> h_srcs, h_dsts;
    h_srcs.reserve(max_edges);
    h_dsts.reserve(max_edges);
    auto fill_edges = [&](size_t inst_start, size_t inst_end) {
      for (size_t inst = inst_start; inst < inst_end; ++inst) {
        const auto *nbrs = h_graph + inst * storage.n_neighbors;
        for (uint32_t j = 0; j < storage.n_neighbors; ++j) {
          int nid = nbrs[j].id();
          if (nid < 0)
            continue;
          h_srcs.push_back(static_cast<Idx_t>(inst));
          h_dsts.push_back(static_cast<Idx_t>(nid));
        }
      }
    };
    fill_edges(range1_start, range1_end);
    fill_edges(0, range2_end);

    const size_t n_edges = h_srcs.size();
    if (n_edges == 0)
      return static_cast<uint32_t>(sel_count);

    rmm::device_uvector<Idx_t> d_srcs(n_edges, embedding_stream),
        d_dsts(n_edges, embedding_stream);
    cudaMemcpyAsync(d_srcs.data(), h_srcs.data(), n_edges * sizeof(Idx_t),
                    cudaMemcpyHostToDevice, embedding_stream);
    cudaMemcpyAsync(d_dsts.data(), h_dsts.data(), n_edges * sizeof(Idx_t),
                    cudaMemcpyHostToDevice, embedding_stream);

    // Precompute sigma and max_w for the full graph.
    update_sigma_all<Data_t, Idx_t>(
        storage.neighbor_indices.data(), storage.neighbor_indices_block_size,
        storage.neighbor_dists.data(), storage.neighbor_dists_block_size,
        storage.n_neighbors, storage.n_inserted_instances, embedding_stream);
    cudaMemsetAsync(storage.max_w.data(), 0, sizeof(float), embedding_stream);
    compute_max_w_from_knn<Data_t, Idx_t>(
        storage.neighbor_indices.data(), storage.neighbor_indices_block_size,
        storage.neighbor_dists.data(), storage.neighbor_dists_block_size,
        storage.n_neighbors, storage.n_inserted_instances, storage.max_w.data(),
        embedding_stream);
    cudaStreamSynchronize(embedding_stream);

    // Initialize epoch state to zero; run all epochs in one loop.
    rmm::device_uvector<float> d_ce(n_edges, embedding_stream),
        d_nse(n_edges, embedding_stream), d_nne(n_edges, embedding_stream);
    cudaMemsetAsync(d_ce.data(), 0, n_edges * sizeof(float), embedding_stream);
    cudaMemsetAsync(d_nse.data(), 0, n_edges * sizeof(float), embedding_stream);
    cudaMemsetAsync(d_nne.data(), 0, n_edges * sizeof(float), embedding_stream);

    for (size_t epoch = 0; epoch < EMBEDDING_MAX_EPOCH; ++epoch) {
      update_embeddings<Data_t, Idx_t>(
          d_srcs.data(), d_dsts.data(), d_ce.data(), d_nse.data(), d_nne.data(),
          n_edges, storage.embedding.data(), params.n_components,
          storage.n_inserted_instances, storage.neighbor_indices.data(),
          storage.neighbor_indices_block_size, storage.neighbor_dists.data(),
          storage.neighbor_dists_block_size, storage.n_neighbors,
          storage.max_w.data(), params.a, params.b, params.alpha,
          EMBEDDING_MAX_EPOCH, params.negative_sample_rate, embedding_stream);
    }

    return static_cast<uint32_t>(sel_count);
  }

  void close() {
    cudaDeviceSynchronize();
    cudaStreamDestroy(insertion_stream);
    cudaStreamDestroy(update_stream);
    cudaStreamDestroy(embedding_stream);
  }

  auto get_embedding() const
      -> raft::device_matrix_view<const Data_t, Idx_t, raft::row_major> {
    auto embedding_ptr = storage.embedding.data();

    return raft::make_device_matrix_view<const Data_t, Idx_t, raft::row_major>(
        embedding_ptr, storage.n_instances, storage.n_components);
  }

  auto get_neighbor_indices() const
      -> raft::device_matrix_view<const Idx_t, Idx_t, raft::row_major> {
    auto neighbor_indices_ptr = storage.neighbor_indices.data();

    return raft::make_device_matrix_view<const Idx_t, Idx_t, raft::row_major>(
        neighbor_indices_ptr, storage.n_instances,
        storage.neighbor_indices_block_size);
  }

  auto get_neighbor_dists() const
      -> raft::device_matrix_view<const Data_t, Idx_t, raft::row_major> {
    auto neighbor_dists_ptr = storage.neighbor_dists.data();

    return raft::make_device_matrix_view<const Data_t, Idx_t, raft::row_major>(
        neighbor_dists_ptr, storage.n_instances,
        storage.neighbor_dists_block_size);
  }
};
} // namespace gpumap
