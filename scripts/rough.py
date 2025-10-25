import numpy as np

rng = np.random.default_rng()
x = np.random.randn(1_000_00000)

# Vectorized sum of squares
sum_vec = (x**2).sum()
print("Vectorized sum:", sum_vec)

# Loop version (may take time for large arrays)
s = 0.0
for v in x:
    s += v*v
print("Loop sum:", s)