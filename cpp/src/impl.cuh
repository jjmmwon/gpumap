#pragma once

#include "gpumap/gpumap.cuh"
#include "queue.cuh"
#include "raft/core/device_mdspan.hpp"
#include "storage.cuh"
#include "utils/matrix_utils.cuh"
#include "kernels/embedding_kernel.cuh"
#include <algorithm>
#include <csignal>
#include <rmm/device_scalar.hpp>
#include <tuple>

namespace gpumap {

using clock = std::chrono::high_resolution_clock;

struct GPUMAP::Impl {
  using Data_t = float;
  using Idx_t = size_t;

  GPUMAPParams params;

  Storage<Data_t, Idx_t> storage;

  cudaStream_t insertion_stream, update_stream, embedding_stream;

  InsertionQueue insertion_queue;
  UpdateQueue update_queue;
  EmbeddingQueue embedding_queue;

  Impl(const GPUMAPParams &params_)
      : params(params_),
        storage(params_.n_instances, params_.n_features, params_.n_neighbors,
                params_.n_components),
        insertion_queue(params_.n_instances) {
    initialize();
  }

  void initialize() {
    cudaStreamCreate(&insertion_stream);
    cudaStreamCreate(&update_stream);
    cudaStreamCreate(&embedding_stream);
  }

  void append_data(const Data_t *ptr, size_t n_new_instances,
                   bool is_device = false) {

    if (storage.n_inserted_instances + n_new_instances > storage.n_instances) {
      throw std::runtime_error("Exceeding maximum number of rows in storage");
    }

    auto data_ptr =
        storage.data.data() + static_cast<size_t>(storage.n_inserted_instances) *
                                  static_cast<size_t>(storage.n_features + 1);

    if (is_device) {
      utils::append_device(data_ptr, ptr, n_new_instances,
                           storage.n_features + 1, insertion_stream);
    } else {
      utils::append_host(data_ptr, ptr, n_new_instances, storage.n_features + 1,
                         insertion_stream);
    }

    storage.n_inserted_instances += n_new_instances;
  }

  GPUMAPResult run(size_t n_insertion_ops, size_t n_update_ops,
                   size_t n_embedding_ops) {
    GPUMAPResult result{};

    if (!insertion_queue.size()) {
      result.insertion_completed = true;
      result.n_insertion_ops = 0;
      result.insertion_time = 0.0f;
    } else {
      auto start = clock::now();
      insert(n_insertion_ops);
      cudaStreamSynchronize(insertion_stream);
      auto end = clock::now();

      result.insertion_completed = false;
      result.n_insertion_ops = n_insertion_ops;
      result.insertion_time = std::chrono::duration<float>(end - start).count();
    }

    if (!update_queue.size()) {
      result.n_update_ops = 0;
      result.update_time = 0.0f;
    } else {
      auto start = clock::now();
      update(n_update_ops);
      cudaStreamSynchronize(update_stream);
      auto end = clock::now();

      result.n_update_ops = n_update_ops;
      result.update_time = std::chrono::duration<float>(end - start).count();
    }

    if (!embedding_queue.size()) {
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
    if (!insertion_queue.size()) {
      return 0;
    }

    std::vector<size_t> batch_indices = insertion_queue.pop_n(n_ops);
    std::vector<size_t> affected_indices;
    std::vector<gpumap::Edge> new_edges;

    // run_init_and_update_kernel(
    //     batch_indices, &insertion_stream, &update_stream, &embedding_stream,
    //     affected_indices,
    //     new_edges); // randomly  initialize neighbors of incies
                    // batch_indices and do descent k times

    // batch indices의 임베딩을 이웃들의 평균으로 초기화
    // affected_indices = batch_indices의 이웃을 unique
    update_queue.push_batch(affected_indices);

    // bath_indices의 index와 이웃 n에 대한 에지 (i, n_0), (i, n_k-1) 까지를
    // 넣어야됨 지금은 넣으면 무조건 하게되는데, 사실 그러면 안됨. 유맵은 스킵이
    // 있음(차례가 안되면). 우리도 CPU에서 업데이트할 녀석들을 앞에서 뽑을 때,
    // 스킵이되면 다시 카운트+1해서 뒤로 넣어주고, 아니면 벡터에넣어서
    // 그 벡터를 임베딩의 인자로 줘야함.

    embedding_queue.push_batch(new_edges);

    return batch_indices.size();
  }

  void update(size_t n_ops) {
    if (!update_queue.size()) {
      return;
    }

    std::vector<size_t> batch_indices = update_queue.pop_n(n_ops);
    std::vector<gpumap::Edge> affected_edges;

    // run_update_kernel(
    //     batch_indices, &update_stream, &embedding_stream,
    //     affected_edges); // batch_indices의 이웃들을 다시
    //                      // 업데이트하고, 그 이웃들을 affected_edges에 추가
    // affected_indices = batch_indices의 이웃을 unique

    embedding_queue.push_batch(affected_edges);
  }

  void embed(size_t n_ops) {
    // embedding 단계에서 queue는 엣지들을 담고있다. 여기서 e = (src, dst, iter)임

    // 먼저 n_ops 개수만큼 엣지를 빼자
    std::vector<gpumap::Edge> batch_edges = embedding_queue.pop_n(n_ops);

    // src, dst, epoch 리스트를 만든다. h for host
    std::vector<Idx_t> h_srcs, h_dsts;
    std::vector<size_t> h_epochs;
    h_srcs.reserve(batch_edges.size());
    h_dsts.reserve(batch_edges.size());
    h_epochs.reserve(batch_edges.size());

    for (const auto& e : batch_edges) {
      h_srcs.push_back(std::get<0>(e));
      h_dsts.push_back(std::get<1>(e));
      h_epochs.push_back(std::get<2>(e));
    }

    // 커널 실행 준비를 위해 src,dst에 대한 디바이스 vector
    rmm::device_uvector<Idx_t> d_srcs(h_srcs.size(), embedding_stream),
                               d_dsts(h_dsts.size(), embedding_stream);

    rmm::device_uvector<Idx_t> d_out_srcs(h_srcs.size(), embedding_stream),
      d_out_dsts(h_dsts.size(), embedding_stream);
    rmm::device_uvector<size_t> d_epochs(h_epochs.size(), embedding_stream);
    rmm::device_uvector<size_t> d_out_epochs(h_epochs.size(), embedding_stream);

    rmm::device_scalar<unsigned long long> d_out_count(embedding_stream);

    // 호스트 -> 디바이스 복사
    cudaMemcpyAsync(d_srcs.data(), h_srcs.data(), h_srcs.size() * sizeof(Idx_t), cudaMemcpyHostToDevice, embedding_stream);

    cudaMemcpyAsync(d_dsts.data(), h_dsts.data(), h_dsts.size() * sizeof(Idx_t),
            cudaMemcpyHostToDevice, embedding_stream);
    cudaMemcpyAsync(d_epochs.data(), h_epochs.data(),
            h_epochs.size() * sizeof(size_t),
            cudaMemcpyHostToDevice, embedding_stream);

    // 이웃인 엣지들만 남긴다. (필터링 커널 실행)
    filter_edges_by_neighbors<Idx_t, size_t>(
      d_srcs.data(), d_dsts.data(), d_epochs.data(), d_out_srcs.data(),
      d_out_dsts.data(), d_out_epochs.data(), batch_edges.size(),
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
      return;
    }

    // 실제 src, dst 호스트로 복사
    std::vector<Idx_t> h_out_srcs(h_out_count);
    std::vector<Idx_t> h_out_dsts(h_out_count);
    std::vector<size_t> h_out_epochs(h_out_count);

    cudaMemcpyAsync(h_out_srcs.data(), d_out_srcs.data(),
                    h_out_count * sizeof(Idx_t), cudaMemcpyDeviceToHost,
                    embedding_stream);
    cudaMemcpyAsync(h_out_dsts.data(), d_out_dsts.data(),
                    h_out_count * sizeof(Idx_t), cudaMemcpyDeviceToHost,
                    embedding_stream);
    cudaMemcpyAsync(h_out_epochs.data(), d_out_epochs.data(),
            h_out_count * sizeof(size_t), cudaMemcpyDeviceToHost,
            embedding_stream);

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

    // 남은 엣지들 중에서 dirty한 녀석들을 보고 dirty하다면 sigma_i를 업데이트해준다.
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
        d_out_srcs.data(), d_out_dsts.data(), h_out_count,
        storage.embedding.data(), params.n_components, storage.n_inserted_instances,
        storage.neighbor_indices.data(), storage.neighbor_indices_block_size,
        storage.neighbor_dists.data(), storage.neighbor_dists_block_size,
        storage.n_neighbors, params.a, params.b, params.alpha,
        params.negative_sample_rate, embedding_stream);

    std::vector<gpumap::Edge> requeue_edges;
    requeue_edges.reserve(h_out_count);
    for (size_t i = 0; i < h_out_count; ++i) {
      const size_t next_epoch = h_out_epochs[i] + 1;
      if (next_epoch <= params.max_epoch) {
        requeue_edges.emplace_back(h_out_srcs[i], h_out_dsts[i], next_epoch);
      }
    }

    if (!requeue_edges.empty()) {
      embedding_queue.push_batch(requeue_edges);
    }



    // 남은것
    // 1) epoch을 보면서 임베딩 업데이트
    // 2) epoch이 max_epoch보다 크면 걔네는 죽이는거 관리 (큐에서 빼는거)
    // 3) negative sampling 처리


    // embedding_queue로부터 최대 n_ops 의 에지를 하나 씩 뺀다.
    // 이를 e라고 하면,
    // 무르익은 것 처리 (에폭과 넥스트에폭을 관리))
    // e = (start, end, weight, current_epoch, target_epoch)
    // e.start, e.end보고 이게 여전히 e.end \in nn[e.start]인지 확인 아니면 버림
    // current_epoch < target_epoch 이면 current_epoch += 1 다시 뒤로
    // e.start, e.end를 업데이트 + 네거티브 n회
    // current_epoch +=1, target_epoch += edge weight이 반비례하는 값
    // max_epoch으로 영원히 죽이는 것관리

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
};
} // namespace gpumap
