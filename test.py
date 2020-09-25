import numpy as np
from sklearn.model_selection import KFold
X = np.array([[100, 200], [300, 400], [100, 200], [300, 400]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
t = kf.get_n_splits(X)
print(t)
print(np.average(y))
