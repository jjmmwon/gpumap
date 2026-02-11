#pragma once

#include "gpumap/gpumap.cuh"
#include "queue.cuh"
#include "raft/core/device_mdspan.hpp"
#include "storage.cuh"
#include "utils/matrix_utils.cuh"
#include <csignal>

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
        storage(params_.n_instances, params_.n_features, params_.n_neighbors),
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
    // TODO: 아래 에러남! device랑 host랑 교통정리하고 개선할것 (호스트에 데이터
    // 추가가 필요한지?) storage.data.append_device(ptr, n_new_instances,
    // insertion_stream);

    // 현재 numpy array 입력만 처리하는 관계로 append_host로 구현해야합니다.
    // wrapper 단에서 device로 옮겨서 넣어준다면 host는 필요없을거로 보입니다.
    size_t max_rows = storage.capacity;
    size_t n_rows = storage.n_data_rows;
    if (n_rows + n_new_instances > max_rows) {
      throw std::runtime_error("Exceeding maximum number of rows in storage");
    }

    auto data_ptr =
        storage.data.data() + static_cast<size_t>(storage.n_data_rows) *
                                  static_cast<size_t>(storage.n_features + 1);

    if (is_device) {
      utils::append_device(data_ptr, ptr, n_new_instances,
                           storage.n_features + 1, insertion_stream);
    } else {
      utils::append_host(data_ptr, ptr, n_new_instances, storage.n_features + 1,
                         insertion_stream);
    }

    storage.n_data_rows += n_new_instances;
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
    std::vector<std::tuple<size_t, size_t, float>> new_edges;

    run_init_and_update_kernel(
        batch_indices, &insertion_stream, &update_stream, &embedding_stream,
        affected_indices,
        new_edges); // randomly  initialize neighbors of incies
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
    std::vector<std::tuple<size_t, size_t, float>> affected_edges;

    run_update_kernel(
        batch_indices, &update_stream, &embedding_stream,
        affected_edges); // batch_indices의 이웃들을 다시
                         // 업데이트하고, 그 이웃들을 affected_edges에 추가
    // affected_indices = batch_indices의 이웃을 unique

    embedding_queue.push_batch(affected_edges);
  }

  void embed(size_t n_ops) {
    // embedding 단계에서 queue는 엣지들을 담고있다. 여기서 e = (src, dst, iter)임

    // 먼저 n_ops 개수만큼 엣지를 빼자
    std::vector<GPUMAP::Edge> batch_edges = embedding_queue.pop_n(n_ops);

    // 이 엣지 중에서 src, dst가 여전히 이웃인지 확인

    



    // 사실 embedding queue에는 엣지가 들어가야 유맵 구현과 같음

    // embedding_queue는 그냥 벡터임

    // embedding_queue로부터 최대 n_ops 의 에지를 하나 씩 뺀다.
    // 이를 e라고 하면,
    // 무르익은 것 처리 (에폭과 넥스트에폭을 관리))
    // e = (start, end, weight, current_epoch, target_epoch)
    // e.start, e.end보고 이게 여전히 e.end \in nn[e.start]인지 확인 아니면 버림
    // current_epoch < target_epoch 이면 current_epoch += 1 다시 뒤로
    // e.start, e.end를 업데이트 + 네거티브 n회
    // current_epoch +=1, target_epoch += edge weight이 반비례하는 값
    // max_epoch으로 영원히 죽이는 것관리

    run_embed_kernel(batch_edges, &embedding_stream);
  }

  void run_init_and_update_kernel(
      const std::vector<size_t> &batch_indices, cudaStream_t *insertion_stream,
      cudaStream_t *update_stream, cudaStream_t *embedding_stream,
      std::vector<size_t> &affected_indices,
      std::vector<std::tuple<size_t, size_t, float>> &new_edges) {}

  void run_update_kernel(
      const std::vector<size_t> &batch_indices, cudaStream_t *update_stream,
      cudaStream_t *embedding_stream,
      std::vector<std::tuple<size_t, size_t, float>> &affected_edges) {}

  void run_embed_kernel(
      const std::vector<std::tuple<size_t, size_t, float>> &batch_edges,
      cudaStream_t *embedding_stream) {}

  void close() {
    cudaStreamDestroy(insertion_stream);
    cudaStreamDestroy(update_stream);
    cudaStreamDestroy(embedding_stream);
  }

  auto get_embedding() const
      -> raft::device_matrix_view<const Data_t, Idx_t, raft::row_major> {
    auto embedding_ptr = storage.embedding.data();
    Idx_t n_rows = storage.n_rows;
    Idx_t n_cols = 2;

    return raft::make_device_matrix_view<const Data_t, Idx_t, raft::row_major>(
        embedding_ptr, n_rows, n_cols);
  }
};
} // namespace gpumap
