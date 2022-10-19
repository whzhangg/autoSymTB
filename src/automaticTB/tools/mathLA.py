import numpy as np
import typing, scipy
from ..parameters import zero_tolerance


def remove_zero_vector_from_coefficients(coeff_matrix: np.ndarray) -> np.ndarray:
    is_zero = np.isclose(np.linalg.norm(coeff_matrix, axis = 1), 0.0, atol = zero_tolerance)
    not_zero = np.invert(is_zero)
    return coeff_matrix[not_zero, :]


def remove_same_direction(vectors: np.ndarray) -> np.ndarray:
    distinct_direction = [vectors[0]]
    for vector in vectors:
        is_close = False
        for compare in distinct_direction:
            cos = np.dot(vector, compare) / np.linalg.norm(vector) / np.linalg.norm(compare)
            if np.isclose(np.abs(cos), 1.0, atol=zero_tolerance):
                is_close = True
                break
        if not is_close:
            distinct_direction.append(vector)
    return np.array(distinct_direction)


def find_linearly_independent_rows(coeff_matrix: np.ndarray) -> np.ndarray:
    # https://stackoverflow.com/questions/53667174/elimination-the-linear-dependent-columns-of-a-non-square-matrix-in-python
    m,n = coeff_matrix.shape
    assert m <= n, print(f"m={m} n={n}")

    q, r = np.linalg.qr(coeff_matrix.T)
    diagonals = np.diagonal(r)
    indices = np.argwhere(np.abs(diagonals) > zero_tolerance)
    return coeff_matrix[indices.flatten(), :]


def first_non_zero(row: np.ndarray) -> int:
    close_to_zero = np.argwhere(np.abs(row) > zero_tolerance)
    return int(close_to_zero[0])


def wrong_find_free_variable_indices(A:np.ndarray) -> typing.Set[int]:
    """
    I followed this method, which worked for some case but it is not correct
    the problem is that U is an upper triangluar matrix, but not necessarily
    a row echelon form, so that free variable idenfitication is not correct.
    """
    # https://stackoverflow.com/questions/52303550/find-which-variables-are-free-in-underdetermined-linear-system
    P, L, U = scipy.linalg.lu(A)
    basis_columns = {first_non_zero(row) for row in U}
    free_variables = set(range(U.shape[1])) - basis_columns
    return free_variables


def find_free_variable_indices(A:np.ndarray) -> typing.Set[int]:
    echelon = row_echelon(A)
    leading_variables = {first_non_zero(row) for row in echelon}
    free_variables = set(range(echelon.shape[1])) - leading_variables
    return free_variables


def row_echelon(A):
    """ 
    Return Row Echelon Form of matrix A 
    https://math.stackexchange.com/questions/3073083/how-to-reduce-matrix-into-row-echelon-form-in-numpy
    """
    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if np.abs(A[i,0]) > zero_tolerance:
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



if __name__ == "__main__":
    a = np.array([
        [-0.5, 0.5, -0.5],
        [ 0.0, 0.5,  1.0],
        [0.5, 0.5, 0.5]
    ])