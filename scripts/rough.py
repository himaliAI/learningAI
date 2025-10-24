import numpy as np

M = np.arange(12).reshape(3,4)
col = np.array([1,10,100]).reshape(3,1)
print("M:\n", M)
print("col:\n", col)
print("M + col:\n", M + col)