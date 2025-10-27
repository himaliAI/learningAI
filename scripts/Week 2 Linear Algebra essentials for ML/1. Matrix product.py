import numpy as np

u = np.array([1, 3, -2])
v = np.array([4, -1, 5])

# dot product vector.vector
dot = np.dot(u, v) # -9

A = np.array([[1, 2, 3],
              [4, 5, 6]]) # shape 2x3
x = np.array([1, 0, -1]) # shape 3,

# matrix-vector product
y = A @ x #[1*1 + 2*0 + 3*-1, 4*1 + 5*0 + 6*-1]

