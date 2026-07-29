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
#include <limits>
#include <numeric>
#include <random>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <vector>

namespace gpumap {

using clock = std::chrono::high_resolution_clock;

struct GPUMAP::Impl {
  using Data_t = float;
  using Idx_t = uint32_t;

  GPUMAPParams params;

  Storage<Data_t, Idx_t> storage;

  cudaStream_t insertion_stream, update_stream, embedding_stream;

  InsertionQueue insertion_queue;
  InsertionQueue embedding_new_queue;
  UpdateQueue update_queue;

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
    if (n_new_instances == 0)
      return;

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
    embedding_new_queue.push_batch(new_indices);
  }

  void sync_knn_rows_to_device(const std::vector<Idx_t> &rows,
                               cudaStream_t stream) {
    static_assert(std::is_same<Idx_t, uint32_t>::value,
                  "sync_knn_rows_to_device scatter path assumes Idx_t == "
                  "uint32_t");
    if (rows.empty())
      return;

    std::vector<Idx_t> unique_rows(rows.begin(), rows.end());
    std::sort(unique_rows.begin(), unique_rows.end());
    unique_rows.erase(std::unique(unique_rows.begin(), unique_rows.end()),
                      unique_rows.end());

    const size_t n = unique_rows.size();
    const size_t k = storage.n_neighbors;
    auto &scratch = storage.nnd_buffer;

    std::vector<uint8_t> valid_row(n);
    utils::stage_and_scatter_knn_rows<Data_t>(
        reinterpret_cast<uint32_t *>(storage.neighbor_indices.data()),
        storage.neighbor_indices_block_size, storage.neighbor_dists.data(),
        storage.neighbor_dists_block_size,
        storage.nnd_buffer.h_graph.data_handle(),
        storage.nnd_buffer.h_dists.data_handle(), k,
        reinterpret_cast<uint32_t *>(unique_rows.data()), n,
        storage.n_inserted_instances,
        scratch.sync_row_ids_scratch_.data_handle(),
        scratch.sync_idx_scratch_.data_handle(),
        scratch.sync_dist_scratch_.data_handle(), valid_row,
        rmm::cuda_stream_view(stream));

    // Unlike the incremental-update path, this initial sync always writes
    // the dirty flag (0 or 1) for every row passed in, since these rows have
    // never been synced to device before.
    std::vector<uint32_t> stg_dirty(valid_row.begin(), valid_row.end());
    utils::scatter_rows_host<uint32_t, uint32_t>(
        reinterpret_cast<uint32_t *>(storage.neighbor_indices.data()) + k,
        storage.neighbor_indices_block_size,
        reinterpret_cast<uint32_t *>(unique_rows.data()), stg_dirty.data(), 1,
        static_cast<uint32_t>(n), scratch.sync_row_ids_scratch_.data_handle(),
        scratch.sync_idx_scratch_.data_handle(), rmm::cuda_stream_view(stream));
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
    log("[GPUMAP] embedding instances=%zu cursor=%zu new_queue=%zu\n",
        storage.n_inserted_instances, embedding_cursor_,
        embedding_new_queue.size());
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

    int seed = n_ops * 9973 + storage.n_inserted_instances * 7919;

    run_insertion<Data_t, Idx_t>(
        storage.nnd_buffer, batch_indices, storage.data.data(),
        storage.data_block_size, storage.n_features, storage.n_neighbors,
        storage.n_inserted_instances, params.metric,
        /*max_iter=*/10, /*seed=*/seed, insertion_stream, affected_indices,
        new_edges, storage.embedding.data(), storage.n_components,
        window_start);

    // Embedding reads storage.neighbor_indices immediately after insertion.
    // Sync the authoritative host-side KNN rows now so new batch rows do not
    // look like all-zero neighbors until a later update pass touches them.
    sync_knn_rows_to_device(affected_indices, insertion_stream);

    update_queue.push_batch(affected_indices);

    return static_cast<uint32_t>(batch_indices.size());
  }

  uint32_t update(uint32_t n_ops, size_t &n_changed_edges_out) {
    n_changed_edges_out = 0;
    if (n_ops == 0 || storage.update_completed ||
        storage.n_inserted_instances == 0)
      return 0;

    constexpr float termination_sigma = 0.0001f;
    const size_t convergence_threshold = static_cast<size_t>(
        termination_sigma * static_cast<float>(storage.n_inserted_instances));

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

      // Convergence: KNN changes dropped below threshold AND all instances
      // have been inserted (no more data will arrive to disturb the graph).
      bool converged = (update_queue.size() < convergence_threshold);

      if (converged) {
        log("[GPUMAP] update convergence check: n_changed=%zu < threshold=%zu, "
            "queue_size=%zu\n",
            n_changed, convergence_threshold, update_queue.size());
        storage.update_completed =
            (storage.n_inserted_instances == storage.n_instances);
        break;
      }
    }

    return n_ops - left_ops;
  }

  uint32_t embed(uint32_t n_ops) {
    if (n_ops == 0 || storage.n_inserted_instances == 0)
      return 0;

    const size_t n = storage.n_inserted_instances;

    // ── Build batch_indices (no duplicates) ──────────────────────────────
    const size_t total_n = static_cast<size_t>(storage.n_instances);
    std::vector<Idx_t> batch_indices;
    if (static_cast<size_t>(n_ops) >= n) {
      // Full sweep: drain queue entirely, then [0, n) in order
      embedding_new_queue.pop_n(embedding_new_queue.size());
      batch_indices.resize(n);
      std::iota(batch_indices.begin(), batch_indices.end(), Idx_t{0});
      embedding_cursor_ = n - 1;
      if (embedding_cursor_ == total_n - 1 && storage.update_completed)
        ++embedding_sweeps_done_;
    } else {
      // Step 1: drain new_queue entirely (may exceed n_ops)
      if (embedding_new_queue.size() > 0) {
        auto new_items = embedding_new_queue.pop_n(embedding_new_queue.size());
        batch_indices.insert(batch_indices.end(), new_items.begin(),
                             new_items.end());
      }

      // Step 2: fill remaining budget with cursor sweep, skipping duplicates
      size_t left_ops = (batch_indices.size() >= static_cast<size_t>(n_ops))
                            ? 0
                            : static_cast<size_t>(n_ops) - batch_indices.size();
      if (left_ops > 0) {
        std::unordered_set<Idx_t> seen(batch_indices.begin(),
                                       batch_indices.end());
        size_t cursor =
            (embedding_cursor_ == SIZE_MAX) ? 0 : (embedding_cursor_ + 1) % n;
        for (size_t checked = 0; left_ops > 0 && checked < n; ++checked) {
          if (seen.insert(static_cast<Idx_t>(cursor)).second) {
            batch_indices.push_back(static_cast<Idx_t>(cursor));
            embedding_cursor_ = cursor;
            if (embedding_cursor_ == total_n - 1 && storage.update_completed)
              ++embedding_sweeps_done_;
            --left_ops;
          }
          cursor = (cursor + 1) % n;
        }
      }
    }

    log("[GPUMAP] embed batch_size=%zu new_queue=%zu cursor=%zu\n",
        batch_indices.size(), embedding_new_queue.size(), embedding_cursor_);

    const int n_int = static_cast<int>(n);
    const int k = static_cast<int>(storage.n_neighbors);
    const int n_batch = static_cast<int>(batch_indices.size());
    // ── Upload batch_indices to GPU ───────────────────────────────────────
    rmm::device_uvector<Idx_t> d_batch(n_batch, embedding_stream);
    cudaMemcpyAsync(d_batch.data(), batch_indices.data(),
                    n_batch * sizeof(Idx_t), cudaMemcpyHostToDevice,
                    embedding_stream);

    // ── Build embed-time KNN with self prepended at col 0 ────────────────
    // Traditional UMAP: knn_indices[i][0]=i, knn_dists[i][0]=0.
    // Self is inserted at col 0; original neighbors shifted to 1..k-1,
    // dropping the k-th (last) neighbor so total column count stays k.
    rmm::device_uvector<Idx_t> d_embed_indices(static_cast<size_t>(n_int) * k,
                                               embedding_stream);
    rmm::device_uvector<float> d_embed_dists(static_cast<size_t>(n_int) * k,
                                             embedding_stream);
    {
      const int total = n_int * k;
      const int tpb = 256;
      const int blks = (total + tpb - 1) / tpb;
      prepend_self_knn_kernel<Idx_t><<<blks, tpb, 0, embedding_stream>>>(
          storage.neighbor_indices.data(),
          static_cast<int>(storage.neighbor_indices_block_size),
          storage.neighbor_dists.data(),
          static_cast<int>(storage.neighbor_dists_block_size),
          d_embed_indices.data(), d_embed_dists.data(), n_int, k,
          params.metric == Metric::Euclidean);
    }

    // ── Allocate COO (2*n_batch*k: first half directed, second half extras) ──
    const int max_nnz = 2 * n_batch * k;
    rmm::device_uvector<int> coo_rows(max_nnz, embedding_stream);
    rmm::device_uvector<int> coo_cols(max_nnz, embedding_stream);
    rmm::device_uvector<float> coo_vals(max_nnz, embedding_stream);

    // ── Build fuzzy simplicial set graph ─────────────────────────────────
    // sigma/rho computed for all n; COO built only for batch rows.
    // Returns actual nnz (directed + asymmetric reverse entries).
    const int actual_nnz = compute_fuzzy_graph<Idx_t>(
        d_embed_indices.data(), k, d_embed_dists.data(), k, n_int, k,
        d_batch.data(), n_batch, coo_rows.data(), coo_cols.data(),
        coo_vals.data(), embedding_stream);

    if (embedding_sweeps_done_ > 2) {
      storage.embedding_completed = storage.update_completed;
    }

    float noise_scale = 0.001f;

    if (noise_scale > 0.0f) {
      rmm::device_uvector<Idx_t> d_selected(batch_indices.size(),
                                            embedding_stream);
      cudaMemcpyAsync(d_selected.data(), batch_indices.data(),
                      batch_indices.size() * sizeof(Idx_t),
                      cudaMemcpyHostToDevice, embedding_stream);

      const unsigned long long noise_seed =
          42ULL ^ static_cast<unsigned long long>(embedding_sweeps_done_);
      add_embedding_noise(storage.embedding.data(), d_selected.data(),
                          batch_indices.size(), params.n_components, n_int,
                          noise_scale, noise_seed, embedding_stream);
    }

    // ── Optimize layout (optimize_layout equivalent) ──────────────────────
    run_optimize_layout(storage.embedding.data(), n_int, coo_rows.data(),
                        coo_cols.data(), coo_vals.data(), actual_nnz,
                        params.alpha, // initial_alpha
                        1.0f,         // gamma (repulsion_strength)
                        params.a, params.b, static_cast<int>(params.max_epoch),
                        static_cast<int>(params.negative_sample_rate),
                        42ULL, // seed
                        embedding_stream);

    return static_cast<uint32_t>(batch_indices.size());
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
