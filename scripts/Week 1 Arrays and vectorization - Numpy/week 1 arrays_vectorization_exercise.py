import numpy as np

#use seed 123 to set reproducibility
rng = np.random.default_rng(seed=123)

#generate an array of 20 random floats from a normal distribution
x = rng.normal(0, 1, 20)

#print all positive values using boolean masks
allPositives = x[x > 0]
print(allPositives)

#replace all negatives in the array with their absolute values
x[x < 0] = -x[x < 0]

#make a (5,4) array using np.arange(20) and reshape it to (5,4)
arr = np.arange(20).reshape(5,4)

#make a vector with 5 random integers shaped (5,1)
arr2 = rng.integers(0, 100, 5).reshape(5,1)

#add arr and arr2
arr3 = arr + arr2
print(arr3)

#show every column of the new array above
print(arr3[::,1::2])

#reverse just the last row and print it
print(arr3[-1,::-1])

#compute the sum of squares for original random float array
arrSumOfSquares = (arr**2).sum()

#replace all entries of arr array, greater than 10 with 10, using boolean indexing
arrLessThanTen = arr.copy()
arrLessThanTen[arrLessThanTen > 10] = 10