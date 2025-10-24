import numpy as np
np.random.seed(0)
a = np.array([1,2,3], dtype=np.float64)
b = np.array([10,20,30], dtype=np.float64)
print(a.shape, a.dtype)
print(b.shape, b.dtype)
print(a + b)
print(a * b)
