import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
B = np.array([[1, 2, 3],
              [2, 3, 1],
              [3, 1, 2]])

dot = np.dot(A, B) #
print(dot)