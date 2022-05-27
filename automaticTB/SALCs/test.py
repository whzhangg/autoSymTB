import numpy as np
from automaticTB.utilities import print_matrix
old = np.array([
    [1,1,0,0,1,0,0],
    [1,1,0,0,1,1,1],
    [0,0,0,0,0,1,0]
])

identity = np.matmul(old, old.T)
w, v = np.linalg.eig(identity)
new = v @ old
print_matrix(np.matmul(new, new.T))

new = np.linalg.inv(v) @ old
print_matrix(np.matmul(new, new.T))