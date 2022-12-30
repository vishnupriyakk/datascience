

import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=4)
kf.get_n_splits(X)

print(kf)
KFold(n_splits=2, random_state=None, shuffle=False)
for i,(train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    print(f"Train: index={train_index}")
    print(f"Test:  index={test_index}")
