#pragma once

#include "gpumap/gpumap.cuh"
#include "queue.cuh"
#include "gpumap_storage.cuh"

namespace gpumap {

struct GPUMAP::Impl {
  using Data_t = float;
  using Idx_t = uint32_t;

  GPUMAPConfig config;
  storage::Storage<Data_t, Idx_t> storage;

  cudaStream_t insertion_stream, update_stream, embedding_stream;


  InsertionQueue insertion_queue;
  UpdateQueue update_queue;
  EmbeddingQueue embedding_queue;
  
  Impl(const GPUMAPConfig &config_)
      : config(config_), storage(config_.n_instances, config_.n_features,
                                 config_.n_neighbors), 
                                 insertion_queue(config_.n_instances) {
                                  
                                 }

                                
  void initialize(){
    insertion_stream = cudaStreamCreateWithFlags(cudaStreamNonBlocking);
    update_stream = cudaStreamCreateWithFlags(cudaStreamNonBlocking);
    embedding_stream = cudaStreamCreateWithFlags(cudaStreamNonBlocking);
  }

  void append_data(const Data_t *ptr, Idx_t n_new_rows, Idx_t n_new_cols) {
    storage.data.append_device(ptr, n_new_rows, rmm::cuda_stream_default);
  }

  void insert(size_t n_jobs) {
    if(!insertion_queue.size()){
      return;
    }

    vector<size_t> batch_indices = insertion_queue.pop_n(n_jobs);

    initAndUpdate(batch_indices); // randomly  initialize neighbors of incies batch_indices and do descent k times

    // batch indices의 임베딩을 이웃들의 평균으로 초기화
    // affected_indices = batch_indices의 이웃을 unique

    update_queue.push_batch(affected_indices);

    // bath_indices의 index와 이웃 n에 대한 에지 (i, n_0), (i, n_k-1) 까지를 넣어야됨
    // 지금은 넣으면 무조건 하게되는데, 사실 그러면 안됨. 유맵은 스킵이 있음(차례가 안되면). 우리도 CPU에서 업데이트할 녀석들을 앞에서 뽑을 때,
    // 스킵이되면 다시 카운트+1해서 뒤로 넣어주고, 아니면 벡터에넣어서
    // 그 벡터를 임베딩의 인자로 줘야함. 
    embedding_queue.push_batch(batch_indices);    
  }

  void update(size_t n_jobs) {
    if(!update_queue.size()){
      return;
    }

    vector<size_t> batch_indices = update_queue.pop_n(n_jobs);

    update(batch_indices);
    // affected_indices = batch_indices의 이웃을 unique

    embedding_queue.push_batch(batch_indices);
  }

  void embed(size_t n_jobs) {
    // 사실 embedding queue에는 엣지가 들어가야 유맵 구현과 같음

    // embedding_queue는 그냥 벡터임

    // embedding_queue로부터 최대 n_jobs 의 에지를 하나 씩 뺀다.
    // 이를 e라고 하면,
    // 무르익은 것 처리 (에폭과 넥스트에폭을 관리))
    // e = (start, end, weight, current_epoch, target_epoch)
    // e.start, e.end보고 이게 여전히 e.end \in nn[e.start]인지 확인 아니면 버림
    // current_epoch < target_epoch 이면 current_epoch += 1 다시 뒤로
    // e.start, e.end를 업데이트 + 네거티브 n회
    // current_epoch +=1, target_epoch += edge weight이 반비례하는 값
    // max_epoch으로 영원히 죽이는 것관리
    
    embed(); // 얘는 i 번째 인스턴의 임베딩을 업데이트,
  }


  void run_nng_build() {
    // NNG build implementation placeholder
  }
  void run_embedding_optimization() {
    // Embedding optimization implementation placeholder
  }

  auto get_data_view() const
      -> raft::device_matrix_view<const Data_t, Idx_t, raft::row_major> {
    return storage.data.view();
  }

  auto get_embedding_view() const
      -> raft::device_matrix_view<const Data_t, Idx_t, raft::row_major> {
    return storage.embedding.view();
  }
};
} // namespace gpumap
