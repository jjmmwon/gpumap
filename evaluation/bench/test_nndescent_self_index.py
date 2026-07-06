"""
NNDescent self-index test: check whether index.graph[:,0] == arange(n).

Run from core/:
    python -m evaluation.bench.test_nndescent_self_index
"""
import numpy as np
import cupy as cp
from cuvs.neighbors import nn_descent

N, D, K = 1000, 32, 15

rng = np.random.default_rng(42)
data = rng.random((N, D), dtype=np.float32)
data_gpu = cp.asarray(data)

params = nn_descent.IndexParams(graph_degree=K, intermediate_graph_degree=2 * K)
index = nn_descent.build(params, data_gpu)
graph = np.asarray(index.graph).astype(np.int32)  # (N, K)

print(f"graph shape : {graph.shape}")
print(f"graph dtype : {graph.dtype}")
print(f"graph[0]    : {graph[0]}")

first_col = graph[:, 0]
self_count = int((first_col == np.arange(N)).sum())
print(f"\n[0th column == self] : {self_count} / {N}  ({self_count/N*100:.1f}%)")

if self_count == N:
    print("→ 첫 번째 인덱스가 항상 본인 (자기 자신 포함)")
elif self_count == 0:
    print("→ 첫 번째 인덱스가 본인이 아님 (자기 자신 미포함)")
else:
    print(f"→ 일부만 본인 ({self_count}개)")
