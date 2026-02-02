import cupy as cp
from proumap import add_f32, print_device_matrix_f32

a = cp.arange(10, dtype=cp.float32)
b = cp.ones(10, dtype=cp.float32) * 2

c = add_f32(a, b)
print(c)
print(cp.asnumpy(c))

# (옵션) matrix print도 확인
x = cp.arange(5 * 12, dtype=cp.float32).reshape(5, 12)
print_device_matrix_f32(x, max_rows=3, max_cols=6)
