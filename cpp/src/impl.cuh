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
  EmbeddingQueue embedding_queue;

  bool first_insertion = true;

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
    // fprintf(
    //     stderr,
    //     "[GPUMAP] run() start: insert=%zu update=%zu embed=%zu
    //     inserted=%u\n", n_insertion_ops, n_update_ops, n_embedding_ops,
    //     storage.n_inserted_instances);
    // fflush(stderr);

    if (!insertion_queue.size()) {
      result.insertion_completed = true;
      result.n_insertion_ops = 0;
      result.insertion_time = 0.0f;
    } else {
      // fprintf(stderr, "[GPUMAP] starting insert(%zu), queue size=%zu\n",
      //         n_insertion_ops, insertion_queue.size());
      // fflush(stderr);
      auto start = clock::now();
      insert(n_insertion_ops);
      cudaStreamSynchronize(insertion_stream);

      // fprintf(stderr, "[GPUMAP] insert done\n");
      // fflush(stderr);
      auto end = clock::now();

      result.insertion_completed = false;
      result.n_insertion_ops = n_insertion_ops;
      result.insertion_time = std::chrono::duration<float>(end - start).count();
    }

    // fprintf(stderr, "[GPUMAP] update queue size=%zu\n", update_queue.size());
    // fflush(stderr);
    if (n_update_ops == 0 || !update_queue.size()) {
      result.n_update_ops = 0;
      result.update_time = 0.0f;
    } else {
      auto start = clock::now();
      update(n_update_ops);
      cudaStreamSynchronize(update_stream);
      // fprintf(stderr, "[GPUMAP] update done\n");
      // fflush(stderr);
      auto end = clock::now();

      result.n_update_ops = n_update_ops;
      result.update_time = std::chrono::duration<float>(end - start).count();
    }

    // fprintf(stderr, "[GPUMAP] embedding queue size=%zu\n",
    //         embedding_queue.size());
    // fflush(stderr);

    if (n_embedding_ops == 0 || !embedding_queue.size()) {
      result.n_embedding_ops = 0;
      result.embedding_time = 0.0f;
    } else {
      auto start = clock::now();
      embed(n_embedding_ops);
      cudaStreamSynchronize(embedding_stream);
      auto end = clock::now();

      result.n_embedding_ops = n_embedding_ops;
      result.embedding_time = std::chrono::duration<float>(end - start).count();
    }

    return result;
  }

  size_t insert(size_t n_ops) {
    if (!insertion_queue.size() || n_ops == 0) {
      return 0;
    }

    std::vector<Idx_t> batch_indices = insertion_queue.pop_n(n_ops);
    std::vector<Idx_t> affected_indices;
    std::vector<gpumap::Edge> new_edges;

    insertion_kernel<Data_t, Idx_t>(
        storage.nnd_buffer, batch_indices,

        storage.data.data(), storage.neighbor_indices.data(),
        storage.neighbor_dists.data(), storage.embedding.data(),

        storage.data_block_size, storage.neighbor_indices_block_size,
        storage.neighbor_dists_block_size, storage.n_components,

        storage.n_instances, storage.n_features, storage.n_neighbors,
        storage.n_inserted_instances, affected_indices, new_edges,
        insertion_stream);

    update_queue.push_batch(affected_indices);
    embedding_queue.push_batch(new_edges);

    return batch_indices.size();
  }

  void update(size_t n_ops) {
    if (!update_queue.size() || n_ops == 0) {
      return;
    }
    std::vector<Idx_t> batch_indices = update_queue.pop_n(n_ops);

    std::vector<Idx_t> affected_indices;
    std::vector<gpumap::Edge> new_edges;

    update_kernel<Data_t, Idx_t>(
        storage.nnd_buffer, batch_indices, affected_indices, new_edges,
        update_stream,

        storage.data.data(), storage.neighbor_indices.data(),
        storage.neighbor_dists.data(), storage.data_block_size,
        storage.neighbor_indices_block_size, storage.neighbor_dists_block_size,

        storage.n_instances, storage.n_features, storage.n_neighbors,
        storage.n_inserted_instances,

        /*max_iter=*/2,
        /*termination_threshold=*/0.001f, /*seed=*/42);

    // printf("Affected indices count: %zu\n", affected_indices.size());
    update_queue.push_batch(affected_indices);
    embedding_queue.push_batch(new_edges);
  }

  void embed(size_t n_ops) {
    size_t left_ops = n_ops;

    while (left_ops > 0 && embedding_queue.size() > 0) {
      size_t current_ops = std::min(left_ops, embedding_queue.size());
      std::vector<gpumap::Edge> batch_edges =
          embedding_queue.pop_n(current_ops);
      // shufle batch_edges
      // std::shuffle(batch_edges.begin(), batch_edges.end(),
      //              std::default_random_engine(42));
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
        // 여전히 이웃인 pair가 하나도 없다면..
        // TODO 만약 진짜 하나도 없다면 몇 번 더봐야 하지 않나?
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
      // fprintf(stderr, "[GPUMAP] unique indices count=%zu\n",
      //         unique_indices.size());

      // 남은 엣지들 중에서 dirty한 녀석들을 보고 dirty하다면 sigma_i를
      // 업데이트해준다.
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
          params.max_epoch, params.negative_sample_rate, embedding_stream);

      raft::copy(h_out_current_epochs.data(), d_out_current_epochs.data(),
                 h_out_count, embedding_stream);
      raft::copy(h_out_next_sample_epochs.data(),
                 d_out_next_sample_epochs.data(), h_out_count,
                 embedding_stream);
      raft::copy(h_out_next_neg_epochs.data(), d_out_next_neg_epochs.data(),
                 h_out_count, embedding_stream);

      std::vector<gpumap::Edge> requeue_edges;
      requeue_edges.reserve(h_out_count);
      for (size_t i = 0; i < h_out_count; ++i) {
        const float next_epoch = h_out_current_epochs[i];
        if (next_epoch <= static_cast<float>(params.max_epoch)) {
          requeue_edges.emplace_back(
              h_out_srcs[i], h_out_dsts[i], h_out_current_epochs[i],
              h_out_next_sample_epochs[i], h_out_next_neg_epochs[i]);
        }
      }

      if (!requeue_edges.empty()) {
        embedding_queue.push_batch(requeue_edges);
      }
      left_ops -= current_ops;
    }
  }

  void close() {
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
