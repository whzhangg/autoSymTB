from automaticTB.tools import find_free_variable_indices
import numpy as np
import scipy
from automaticTB.printing import print_matrix

def row_echelon(A):
    """ Return Row Echelon Form of matrix A """
    """https://math.stackexchange.com/questions/3073083/how-to-reduce-matrix-into-row-echelon-form-in-numpy"""
    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]

    # we perform REF on matrix from second row, from second column
    B = row_echelon(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

number1 = np.array([
    [7.07, 0, 0, 7.07, 0, 0],
    [0, 7.07, 0, 0, -7.07, 0],
    [0, 0, 7.07, 0, 0, -7.07],
    [0, 7.07, 0, 0, 7.07, 0],
    [0, 0, 7.07, 0, 0, 7.07],
])

number = np.array([[ 7.07106781e-01, -2.17894100e-33,  2.16489014e-17,  7.07106781e-01, -2.17894100e-33,  2.16489014e-17],
 [ 0.00000000e+00,  7.07106781e-01, -4.32978028e-17,  0.00000000e+00, -7.07106781e-01,  4.32978028e-17],
 [ 0.00000000e+00,  4.32978028e-17,  7.07106781e-01,  0.00000000e+00, -4.32978028e-17, -7.07106781e-01],
 [ 0.00000000e+00,  7.07106781e-01, -4.32978028e-17,  0.00000000e+00, 7.07106781e-01, -4.32978028e-17],
 [ 0.00000000e+00,  4.32978028e-17,  7.07106781e-01,  0.00000000e+00, 4.32978028e-17,  7.07106781e-01]], dtype=float)

number2 = np.array(
[[ 7.07106781e-01,  7.07106781e-01, -2.17894100e-33, -2.17894100e-33, 2.16489014e-17,  2.16489014e-17],
 [ 0.00000000e+00,  0.00000000e+00,  7.07106781e-01,  7.07106781e-01, -4.32978028e-17, -4.32978028e-17],
 [ 0.00000000e+00, -0.00000000e+00,  7.07106781e-01, -7.07106781e-01, -4.32978028e-17,  4.32978028e-17],
 [ 0.00000000e+00,  0.00000000e+00,  4.32978028e-17,  4.32978028e-17, 7.07106781e-01,  7.07106781e-01],
 [ 0.00000000e+00, -0.00000000e+00,  4.32978028e-17, -4.32978028e-17, 7.07106781e-01, -7.07106781e-01]]
)

print_matrix(row_echelon(number1))
print_matrix(row_echelon(number2))

print(find_free_variable_indices(number1))
print_matrix(number1)
P, L, U = scipy.linalg.lu(number1)
print_matrix(U)
print(find_free_variable_indices(number2))
print_matrix(number2)
P, L, U = scipy.linalg.lu(number2)
print_matrix(U)