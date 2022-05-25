import numpy as np

A = np.array([
    [1,1,1],
    [0, 1, -1]
], dtype=float)

b = np.zeros([0,0], dtype=float)


u,s,vh = np.linalg.svd(A, full_matrices=False)
print(vh.T)
print(s)
print(u @ np.diag(s) @ vh)

from scipy.linalg import lu
U = lu(A)[2]
basis_columns = {np.flatnonzero(U[i, :])[0] for i in range(U.shape[0])}
free_variables = set(range(U.shape[1])) - basis_columns
print(free_variables)

P, L, U = lu(A)
print(P)
print(L)
print(U)
for i in range(U.shape[0]):
    print(np.flatnonzero(U[i, :]))
lin_indep_columns = [np.flatnonzero(U[i, :])[0] for i in range(U.shape[0])]
print(lin_indep_columns)