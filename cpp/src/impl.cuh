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

  static constexpr float FIRST_MAX_EPOCH = 20.f;
  static constexpr float SECOND_MAX_EPOCH = 40.f;
  static constexpr float THIRD_MAX_EPOCH = 60.f;
  static constexpr float LAST_MAX_EPOCH = 80.f;

  Queue<Edge> first_embed_queue;
  Queue<Edge> second_embed_queue;
  Queue<Edge> third_embed_queue;
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

    std::vector<Idx_t> new_indices;
    for (Idx_t i = old_count; i < storage.n_inserted_instances; ++i) {
      new_indices.push_back(i);
    }
    insertion_queue.push_batch(new_indices);
  }

  GPUMAPResult run(size_t n_insertion_ops, size_t n_update_ops,
                   size_t n_embedding_ops) {
    GPUMAPResult result{};
    printf("Debugging Trial #3\n");
    log("[GPUMAP] run() start: insert=%zu update=%zu embed=%zu inserted=%u\n",
        n_insertion_ops, n_update_ops, n_embedding_ops,
        storage.n_inserted_instances);

    if (!insertion_queue.size() &&
        storage.n_inserted_instances == storage.n_instances) {
      result.insertion_completed = true;
      result.n_insertion_ops = 0;
      result.insertion_time = 0.0f;
    } else {
      log("[GPUMAP] insert start: n_ops=%zu queue=%zu\n", n_insertion_ops,
          insertion_queue.size());
      auto start = clock::now();
      result.n_insertion_ops = insert(n_insertion_ops);
      cudaStreamSynchronize(insertion_stream);
      auto end = clock::now();
      log("[GPUMAP] insert done: actual=%zu time=%.3fs\n",
          result.n_insertion_ops,
          std::chrono::duration<float>(end - start).count());

      result.insertion_completed =
          insertion_queue.size() == 0 &&
          storage.n_inserted_instances == storage.n_instances;
      result.insertion_time = std::chrono::duration<float>(end - start).count();
    }

    log("[GPUMAP] update queue size=%zu\n", update_queue.size());
    if (n_update_ops == 0) {
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
      log("[GPUMAP] update done: actual=%zu changed_edges=%zu time=%.3fs\n",
          result.n_update_ops, result.n_update_changed_edges,
          std::chrono::duration<float>(end - start).count());

      result.update_time = std::chrono::duration<float>(end - start).count();
    }

    result.embedding_pending_edges =
        first_embed_queue.size() + second_embed_queue.size() +
        third_embed_queue.size() + last_embed_queue.size();
    log("[GPUMAP] embedding queue size=%zu\n", result.embedding_pending_edges);
    if (n_embedding_ops == 0 || !result.embedding_pending_edges) {
      result.n_embedding_ops = 0;
      result.embedding_time = 0.0f;
    } else {
      auto start = clock::now();
      result.n_embedding_ops = embed(n_embedding_ops);
      cudaStreamSynchronize(embedding_stream);
      auto end = clock::now();
      log("[GPUMAP] embed done: actual=%zu time=%.3fs\n",
          result.n_embedding_ops,
          std::chrono::duration<float>(end - start).count());

      result.embedding_time = std::chrono::duration<float>(end - start).count();
    }

    result.n_inserted_instances = storage.n_inserted_instances;
    result.insertion_queue_size = insertion_queue.size();
    result.update_queue_size = update_queue.size();
    result.embedding_queue_level_sizes = {
        static_cast<uint32_t>(first_embed_queue.size()),
        static_cast<uint32_t>(second_embed_queue.size()),
        static_cast<uint32_t>(third_embed_queue.size()),
        static_cast<uint32_t>(last_embed_queue.size()),
    };

    return result;
  }

  size_t insert(size_t n_ops) {
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
        /*max_iter=*/3, /*seed=*/42, insertion_stream, affected_indices,
        new_edges, storage.embedding.data(), storage.n_components,
        window_start);

    update_queue.push_batch(affected_indices);
    first_embed_queue.push_batch(new_edges);

    return batch_indices.size();
  }

  size_t update(size_t n_ops, size_t &n_changed_edges_out) {
    n_changed_edges_out = 0;
    if (n_ops == 0 || storage.n_inserted_instances == 0)
      return 0;

    constexpr float termination_sigma = 0.001f;
    const size_t convergence_threshold = static_cast<size_t>(
        termination_sigma * static_cast<float>(storage.n_inserted_instances) *
        static_cast<float>(storage.n_neighbors));

    size_t left_ops = n_ops;

    while (left_ops > 0) {
      std::vector<Idx_t> batch_indices;

      if (!update_queue.empty()) {
        size_t pop_count = std::min(left_ops, update_queue.size());
        batch_indices = update_queue.pop_n(pop_count);
      } else {
        size_t sample_size = std::min(
            left_ops, static_cast<size_t>(storage.n_inserted_instances));
        std::uniform_int_distribution<Idx_t> dist(
            0, static_cast<Idx_t>(storage.n_inserted_instances - 1));
        batch_indices.resize(sample_size);
        auto start_offset = dist(rng_);
        for (size_t i = 0; i < sample_size; ++i) {
          batch_indices[i] = (start_offset + i) % storage.n_inserted_instances;
        }
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

          storage.n_instances, storage.n_features, storage.n_neighbors,
          storage.n_inserted_instances,

          /*termination_threshold=*/termination_sigma, /*seed=*/42);

      n_changed_edges_out += new_edges.size();
      first_embed_queue.push_batch(new_edges);
      left_ops -= batch_indices.size();

      // Convergence check: stop loop when neighbor changes drop below
      // threshold.
      if (n_changed < convergence_threshold) {
        log("[GPUMAP] update convergence: n_changed=%zu < threshold=%zu\n",
            n_changed, convergence_threshold);
        break;
      }

      update_queue.push_batch(affected_indices);

      // Cascade 1-hop neighbors of affected nodes into the queue.
      const auto *h_graph = storage.nnd_buffer.h_graph.data_handle();
      for (auto idx : affected_indices) {
        const auto *nbrs =
            h_graph + static_cast<size_t>(idx) * storage.n_neighbors;
        for (uint32_t j = 0; j < storage.n_neighbors; ++j) {
          int nid = nbrs[j].id();
          if (nid >= 0 &&
              static_cast<size_t>(nid) < storage.n_inserted_instances)
            update_queue.push(static_cast<Idx_t>(nid));
        }
      }
    }

    return n_ops - left_ops;
  }

  // 하나의 stage에 대해 embedding SGD를 수행한다.
  // src_queue에서 current_ops개의 edge를 꺼내 update_embeddings를 실행한다.
  // stage가 끝나지 않은 edge는 src_queue에 다시 넣고,
  // stage_max_epoch을 넘은 edge는 epoch을 초기화해 dst_queue로 승격한다
  // (dst_queue가 nullptr이면 마지막 stage이므로 해당 edge는 그냥 버린다).
  size_t embed_stage(size_t n_ops, Queue<Edge> &src_queue,
                     float stage_max_epoch, Queue<Edge> *dst_queue) {
    size_t left_ops = n_ops;

    while (left_ops > 0 && !src_queue.empty()) {
      size_t current_ops = std::min(left_ops, src_queue.size());
      std::vector<gpumap::Edge> batch_edges = src_queue.pop_n(current_ops);

      // src, dst, epoch 리스트를 만든다. h for host
      const size_t n_edges = batch_edges.size();
      std::vector<Idx_t> h_srcs(n_edges), h_dsts(n_edges), h_indices(n_edges);
      std::vector<float> h_current_epochs(n_edges),
          h_next_sample_epochs(n_edges), h_next_neg_epochs(n_edges);

      for (size_t i = 0; i < n_edges; ++i) {
        h_srcs[i] = std::get<0>(batch_edges[i]);
        h_dsts[i] = std::get<1>(batch_edges[i]);
        h_indices[i] = static_cast<Idx_t>(i);
        h_current_epochs[i] = std::get<2>(batch_edges[i]);
        h_next_sample_epochs[i] = std::get<3>(batch_edges[i]);
        h_next_neg_epochs[i] = std::get<4>(batch_edges[i]);
      }

      // 커널 실행 준비를 위해 src, dst, index 디바이스 vector 업로드
      rmm::device_uvector<Idx_t> d_srcs(n_edges, embedding_stream),
          d_dsts(n_edges, embedding_stream),
          d_indices(n_edges, embedding_stream);
      rmm::device_uvector<Idx_t> d_out_srcs(n_edges, embedding_stream),
          d_out_dsts(n_edges, embedding_stream),
          d_out_indices(n_edges, embedding_stream);
      rmm::device_scalar<unsigned long long> d_out_count(embedding_stream);

      cudaMemcpyAsync(d_srcs.data(), h_srcs.data(), n_edges * sizeof(Idx_t),
                      cudaMemcpyHostToDevice, embedding_stream);
      cudaMemcpyAsync(d_dsts.data(), h_dsts.data(), n_edges * sizeof(Idx_t),
                      cudaMemcpyHostToDevice, embedding_stream);
      cudaMemcpyAsync(d_indices.data(), h_indices.data(),
                      n_edges * sizeof(Idx_t), cudaMemcpyHostToDevice,
                      embedding_stream);

      // 이웃인 엣지들만 남긴다. payload = index → 살아남은 index를 복원에 사용
      filter_edges_by_neighbors<Idx_t, Idx_t>(
          d_srcs.data(), d_dsts.data(), d_indices.data(), d_out_srcs.data(),
          d_out_dsts.data(), d_out_indices.data(), n_edges,
          storage.neighbor_indices.data(), storage.neighbor_indices_block_size,
          storage.n_neighbors, d_out_count.data(), embedding_stream);

      // 필터링 하고 남은 개수 읽어오기 (즉 아직 src, dst가 이웃인 엣지의 수)
      unsigned long long h_out_count = 0;
      cudaMemcpyAsync(&h_out_count, d_out_count.data(),
                      sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                      embedding_stream);
      cudaStreamSynchronize(embedding_stream);

      if (h_out_count == 0) {
        left_ops -= current_ops;
        continue;
      }

      // 살아남은 edge의 src, dst, index 복사
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

      // 살아남은 index를 이용해 epoch 배열 복원
      std::vector<float> h_out_current_epochs(h_out_count),
          h_out_next_sample_epochs(h_out_count),
          h_out_next_neg_epochs(h_out_count);
      for (size_t i = 0; i < h_out_count; ++i) {
        const Idx_t idx = h_out_indices[i];
        h_out_current_epochs[i] = h_current_epochs[idx];
        h_out_next_sample_epochs[i] = h_next_sample_epochs[idx];
        h_out_next_neg_epochs[i] = h_next_neg_epochs[idx];
      }

      // epoch 배열을 디바이스로 업로드
      rmm::device_uvector<float> d_out_current_epochs(h_out_count,
                                                      embedding_stream),
          d_out_next_sample_epochs(h_out_count, embedding_stream),
          d_out_next_neg_epochs(h_out_count, embedding_stream);
      cudaMemcpyAsync(d_out_current_epochs.data(), h_out_current_epochs.data(),
                      h_out_count * sizeof(float), cudaMemcpyHostToDevice,
                      embedding_stream);
      cudaMemcpyAsync(d_out_next_sample_epochs.data(),
                      h_out_next_sample_epochs.data(),
                      h_out_count * sizeof(float), cudaMemcpyHostToDevice,
                      embedding_stream);
      cudaMemcpyAsync(d_out_next_neg_epochs.data(),
                      h_out_next_neg_epochs.data(), h_out_count * sizeof(float),
                      cudaMemcpyHostToDevice, embedding_stream);

      cudaStreamSynchronize(embedding_stream);

      // 남은 엣지의 src, dst 인덱스 중에서 중복을 제거한다.
      std::vector<Idx_t> unique_indices;
      unique_indices.reserve(h_out_count * 2);
      for (size_t i = 0; i < h_out_count; ++i) {
        unique_indices.push_back(h_out_srcs[i]);
        unique_indices.push_back(h_out_dsts[i]);
      }
      std::sort(unique_indices.begin(), unique_indices.end());
      unique_indices.erase(
          std::unique(unique_indices.begin(), unique_indices.end()),
          unique_indices.end());

      rmm::device_uvector<Idx_t> d_unique_indices(unique_indices.size(),
                                                  embedding_stream);
      cudaMemcpyAsync(d_unique_indices.data(), unique_indices.data(),
                      unique_indices.size() * sizeof(Idx_t),
                      cudaMemcpyHostToDevice, embedding_stream);

      update_sigma_for_dirty<Data_t, Idx_t>(
          d_unique_indices.data(), unique_indices.size(),
          storage.neighbor_indices.data(), storage.neighbor_indices_block_size,
          storage.neighbor_dists.data(), storage.neighbor_dists_block_size,
          storage.n_neighbors, embedding_stream);

      update_embeddings<Data_t, Idx_t>(
          d_out_srcs.data(), d_out_dsts.data(), d_out_current_epochs.data(),
          d_out_next_sample_epochs.data(), d_out_next_neg_epochs.data(),
          h_out_count, storage.embedding.data(), params.n_components,
          storage.n_inserted_instances, storage.neighbor_indices.data(),
          storage.neighbor_indices_block_size, storage.neighbor_dists.data(),
          storage.neighbor_dists_block_size, storage.n_neighbors,
          storage.max_w.data(), params.a, params.b, params.alpha,
          static_cast<size_t>(stage_max_epoch), params.negative_sample_rate,
          embedding_stream);

      raft::copy(h_out_current_epochs.data(), d_out_current_epochs.data(),
                 h_out_count, embedding_stream);
      raft::copy(h_out_next_sample_epochs.data(),
                 d_out_next_sample_epochs.data(), h_out_count,
                 embedding_stream);
      raft::copy(h_out_next_neg_epochs.data(), d_out_next_neg_epochs.data(),
                 h_out_count, embedding_stream);
      cudaStreamSynchronize(embedding_stream);

      // stage 내에서 아직 덜 끝난 edge는 src_queue로 돌려보내고,
      // stage를 완료한 edge는 epoch을 초기화해 dst_queue로 승격한다.
      std::vector<gpumap::Edge> requeue_edges, promote_edges;
      for (size_t i = 0; i < h_out_count; ++i) {
        if (h_out_current_epochs[i] <= stage_max_epoch) {
          requeue_edges.emplace_back(
              h_out_srcs[i], h_out_dsts[i], h_out_current_epochs[i],
              h_out_next_sample_epochs[i], h_out_next_neg_epochs[i]);
        } else if (dst_queue != nullptr) {
          promote_edges.emplace_back(h_out_srcs[i], h_out_dsts[i], 0.f, -1.f,
                                     -1.f);
        }
      }

      if (!requeue_edges.empty())
        src_queue.push_batch(requeue_edges);
      if (!promote_edges.empty())
        dst_queue->push_batch(promote_edges);

      left_ops -= current_ops;
    }
    return n_ops - left_ops;
  }

  size_t embed(size_t n_ops) {
    // first stage를 먼저 소진하고, 남은 ops를 순서대로 다음 stage로 넘긴다.
    size_t done = 0;
    done += embed_stage(n_ops - done, first_embed_queue, FIRST_MAX_EPOCH,
                        &second_embed_queue);
    done += embed_stage(n_ops - done, second_embed_queue, SECOND_MAX_EPOCH,
                        &third_embed_queue);
    done += embed_stage(n_ops - done, third_embed_queue, THIRD_MAX_EPOCH,
                        &last_embed_queue);

    done +=
        embed_stage(n_ops - done, last_embed_queue, LAST_MAX_EPOCH, nullptr);
    return done;
  }

  void close() {
    // 스트림 destroy 전에 device 전체를 동기화한다.
    // 각 스트림을 개별 sync하는 것만으로는 스트림 간 dependency나
    // 비동기 메모리 오퍼레이션이 남아있을 수 있다.
    // 특히 WSL2에서 이후 cudaFreeHost()가 WDDM 레이어에 완전히
    // 반납되려면 모든 GPU 작업이 종료된 상태여야 한다.
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
