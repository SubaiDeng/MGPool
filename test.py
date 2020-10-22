import numpy as np

a = np.array([1, 2, 0, 1])

mask = np.zeros([4,3])

mask[list(range(len(a))), list(a)] = 1

print(mask)