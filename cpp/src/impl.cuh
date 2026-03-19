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

  // First Option: (1, 2, 3)
  static constexpr float FIRST_MAX_EPOCH = 35.f;
  static constexpr float SECOND_MAX_EPOCH = 70.f;
  static constexpr float LAST_MAX_EPOCH = 105.f;

  Queue<Edge> first_embed_queue;
  Queue<Edge> second_embed_queue;
  Queue<Edge> last_embed_queue;

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

    result.embedding_pending_edges = first_embed_queue.size() +
                                     second_embed_queue.size() +
                                     last_embed_queue.size();
    log("[GPUMAP] embedding queue size=%zu\n", result.embedding_pending_edges);
    if (n_embedding_ops == 0 || result.embedding_pending_edges == 0) {
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

    // embedding_completed: all data inserted, KNN converged, no pending edges.
    storage.embedding_completed =
        storage.insertion_completed && storage.update_completed &&
        (first_embed_queue.size() + second_embed_queue.size() +
             last_embed_queue.size() ==
         0);

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
    first_embed_queue.push_batch(new_edges);

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
      first_embed_queue.push_batch(new_edges);
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

  uint32_t embed_stage(uint32_t n_ops, Queue<Edge> &src_queue,
                       float stage_max_epoch, Queue<Edge> *dst_queue) {
    if (n_ops == 0 || src_queue.empty())
      return 0;

    // ── 1. Drain entire queue
    // ─────────────────────────────────────────────────
    std::vector<gpumap::Edge> all_edges = src_queue.pop_n(src_queue.size());
    const size_t n_all = all_edges.size();

    std::vector<Idx_t> h_srcs(n_all), h_dsts(n_all), h_indices(n_all);
    std::vector<float> h_ce(n_all), h_nse(n_all), h_nne(n_all);
    for (size_t i = 0; i < n_all; ++i) {
      h_srcs[i] = std::get<0>(all_edges[i]);
      h_dsts[i] = std::get<1>(all_edges[i]);
      h_indices[i] = static_cast<Idx_t>(i);
      h_ce[i] = std::get<2>(all_edges[i]);
      h_nse[i] = std::get<3>(all_edges[i]);
      h_nne[i] = std::get<4>(all_edges[i]);
    }

    // ── 2. Upload and filter all edges once
    // ───────────────────────────────────
    rmm::device_uvector<Idx_t> d_srcs(n_all, embedding_stream),
        d_dsts(n_all, embedding_stream), d_indices(n_all, embedding_stream);
    rmm::device_uvector<Idx_t> d_out_srcs(n_all, embedding_stream),
        d_out_dsts(n_all, embedding_stream),
        d_out_indices(n_all, embedding_stream);
    rmm::device_scalar<unsigned long long> d_out_count(embedding_stream);

    cudaMemcpyAsync(d_srcs.data(), h_srcs.data(), n_all * sizeof(Idx_t),
                    cudaMemcpyHostToDevice, embedding_stream);
    cudaMemcpyAsync(d_dsts.data(), h_dsts.data(), n_all * sizeof(Idx_t),
                    cudaMemcpyHostToDevice, embedding_stream);
    cudaMemcpyAsync(d_indices.data(), h_indices.data(), n_all * sizeof(Idx_t),
                    cudaMemcpyHostToDevice, embedding_stream);

    filter_edges_by_neighbors<Idx_t, Idx_t>(
        d_srcs.data(), d_dsts.data(), d_indices.data(), d_out_srcs.data(),
        d_out_dsts.data(), d_out_indices.data(), n_all,
        storage.neighbor_indices.data(), storage.neighbor_indices_block_size,
        storage.n_neighbors, d_out_count.data(), embedding_stream);

    unsigned long long h_out_count = 0;
    cudaMemcpyAsync(&h_out_count, d_out_count.data(),
                    sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                    embedding_stream);
    cudaStreamSynchronize(embedding_stream);

    if (h_out_count == 0)
      return 0;

    std::vector<Idx_t> h_out_srcs(h_out_count), h_out_dsts(h_out_count),
        h_out_indices(h_out_count);
    cudaMemcpyAsync(h_out_srcs.data(), d_out_srcs.data(),
                    h_out_count * sizeof(Idx_t), cudaMemcpyDeviceToHost,
                    embedding_stream);
    cudaMemcpyAsync(h_out_dsts.data(), d_out_dsts.data(),
                    h_out_count * sizeof(Idx_t), cudaMemcpyDeviceToHost,
                    embedding_stream);
    cudaMemcpyAsync(h_out_indices.data(), d_out_indices.data(),
                    h_out_count * sizeof(Idx_t), cudaMemcpyDeviceToHost,
                    embedding_stream);
    cudaStreamSynchronize(embedding_stream);

    // Restore epoch state for filtered edges.
    const size_t n_valid = static_cast<size_t>(h_out_count);
    std::vector<float> h_out_ce(n_valid), h_out_nse(n_valid),
        h_out_nne(n_valid);
    for (size_t i = 0; i < n_valid; ++i) {
      const Idx_t idx = h_out_indices[i];
      h_out_ce[i] = h_ce[idx];
      h_out_nse[i] = h_nse[idx];
      h_out_nne[i] = h_nne[idx];
    }

    // ── 3. Water-fill alloc: find target_epoch so all edges reach the same
    // level.
    //    Pass 1 — find min/max current epoch and total cost to reach max.
    size_t total_ops = 0;
    size_t target_epoch = 0;

    size_t ce_min = static_cast<size_t>(h_out_ce[0]);
    size_t ce_max = ce_min;
    size_t sum_ce = 0;

    const size_t hard_max = static_cast<size_t>(stage_max_epoch) + 1;
    {
      for (size_t i = 0; i < n_valid; ++i) {
        const size_t ce = static_cast<size_t>(h_out_ce[i]);
        ce_min = std::min(ce_min, ce);
        ce_max = std::max(ce_max, ce);
        sum_ce += ce;
      }

      // Cost to bring every edge up to ce_max.
      const size_t cost_to_max = n_valid * ce_max - sum_ce;

      // Remaining ops after alignment; extra full epochs above ce_max.
      size_t extra = 0;
      if (static_cast<size_t>(n_ops) > cost_to_max) {
        const size_t remaining = static_cast<size_t>(n_ops) - cost_to_max;
        // ceil(remaining / n_valid) so total_alloc >= n_ops.
        extra = (remaining + n_valid - 1) / n_valid;
      }
      target_epoch = std::min(ce_max + extra, hard_max);
      // Recompute actual total_ops from the final capped target_epoch.
      total_ops = n_valid * target_epoch - sum_ce;
    }

    const size_t global_min_epoch = ce_min;
    const size_t global_max_epoch = target_epoch;

    // ── 4. Upload edge data to GPU
    // ──────────────────────────────────────────
    rmm::device_uvector<Idx_t> d_p_srcs(n_valid, embedding_stream),
        d_p_dsts(n_valid, embedding_stream);
    rmm::device_uvector<float> d_p_ce(n_valid, embedding_stream),
        d_p_nse(n_valid, embedding_stream), d_p_nne(n_valid, embedding_stream);

    cudaMemcpyAsync(d_p_srcs.data(), h_out_srcs.data(), n_valid * sizeof(Idx_t),
                    cudaMemcpyHostToDevice, embedding_stream);
    cudaMemcpyAsync(d_p_dsts.data(), h_out_dsts.data(), n_valid * sizeof(Idx_t),
                    cudaMemcpyHostToDevice, embedding_stream);
    cudaMemcpyAsync(d_p_ce.data(), h_out_ce.data(), n_valid * sizeof(float),
                    cudaMemcpyHostToDevice, embedding_stream);
    cudaMemcpyAsync(d_p_nse.data(), h_out_nse.data(), n_valid * sizeof(float),
                    cudaMemcpyHostToDevice, embedding_stream);
    cudaMemcpyAsync(d_p_nne.data(), h_out_nne.data(), n_valid * sizeof(float),
                    cudaMemcpyHostToDevice, embedding_stream);

    // ── 5. Host epoch loop — call update_embeddings once per epoch tick
    // ──────
    for (size_t epoch = global_min_epoch; epoch < global_max_epoch; ++epoch) {
      update_embeddings<Data_t, Idx_t>(
          d_p_srcs.data(), d_p_dsts.data(), d_p_ce.data(), d_p_nse.data(),
          d_p_nne.data(), n_valid, storage.embedding.data(),
          params.n_components, storage.n_inserted_instances,
          storage.neighbor_indices.data(), storage.neighbor_indices_block_size,
          storage.neighbor_dists.data(), storage.neighbor_dists_block_size,
          storage.n_neighbors, storage.max_w.data(), params.a, params.b,
          params.alpha, static_cast<size_t>(stage_max_epoch),
          params.negative_sample_rate, embedding_stream);
    }

    raft::copy(h_out_ce.data(), d_p_ce.data(), n_valid, embedding_stream);
    raft::copy(h_out_nse.data(), d_p_nse.data(), n_valid, embedding_stream);
    raft::copy(h_out_nne.data(), d_p_nne.data(), n_valid, embedding_stream);
    cudaStreamSynchronize(embedding_stream);

    // ── 6. Requeue / promote
    // ──────────────────────────────────────────────────
    std::vector<gpumap::Edge> requeue_edges, promote_edges;
    for (size_t i = 0; i < n_valid; ++i) {
      if (h_out_ce[i] <= stage_max_epoch) {
        requeue_edges.emplace_back(h_out_srcs[i], h_out_dsts[i], h_out_ce[i],
                                   h_out_nse[i], h_out_nne[i]);
      } else if (dst_queue != nullptr) {
        promote_edges.emplace_back(h_out_srcs[i], h_out_dsts[i], h_out_ce[i],
                                   h_out_nse[i], h_out_nne[i]);
      }
      // else: exceeded last stage max, drop
    }

    if (!requeue_edges.empty())
      src_queue.push_batch(requeue_edges);
    if (!promote_edges.empty())
      dst_queue->push_batch(promote_edges);

    return total_ops;
  }

  uint32_t embed(uint32_t n_ops) {
    // Precompute sigma for all dirty nodes and global max_w from full KNN
    // graph. This replaces the per-batch update_sigma_for_dirty calls in
    // embed_stage.
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

    uint32_t done = 0;
    done += embed_stage(n_ops - done, first_embed_queue, FIRST_MAX_EPOCH,
                        &second_embed_queue);
    done += embed_stage(n_ops - done, second_embed_queue, SECOND_MAX_EPOCH,
                        &last_embed_queue);
    done +=
        embed_stage(n_ops - done, last_embed_queue, LAST_MAX_EPOCH, nullptr);
    return done;
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
