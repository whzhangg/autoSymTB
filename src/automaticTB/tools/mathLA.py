import numpy as np
import typing
from ..parameters import zero_tolerance

__all__ = [
    "get_distinct_nonzero_vector_from_coefficients",
    "remove_zero_vector_from_coefficients", 
    "find_linearly_independent_rows",
    "remove_parallel_vectors",
    "find_free_variable_indices_by_row_echelon",
    "_row_echelon"
]


def get_distinct_nonzero_vector_from_coefficients(coeff_matrix: np.ndarray) \
-> np.ndarray:
    """
    This function combines the following functions, since they are used together
    1. remove_zero_vector_from_coefficients()
    2. find_linearly_independent_rows
    It returns an empty array if there are no nonzero coefficients
    """
    non_zero = remove_zero_vector_from_coefficients(coeff_matrix)
    if len(non_zero) > 0:
        return find_linearly_independent_rows(non_zero)
    else:
        return np.array([], dtype = coeff_matrix.dtype)


def remove_zero_vector_from_coefficients(coeff_matrix: np.ndarray) -> np.ndarray:
    """
    This function remove rows of zero from an array
    """
    is_zero = np.isclose(np.linalg.norm(coeff_matrix, axis = 1), 0.0, atol = zero_tolerance)
    not_zero = np.invert(is_zero)
    return coeff_matrix[not_zero, :]


def remove_parallel_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    This function provide method to remove the direction that already appeared before by 
    checking the generalized cos angle (absolute value) between two vectors. 
    """
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
    """
    Find linear independent rows with QR factorization, 
    asserting number of rows are smaller than number of columns.
    ref: https://stackoverflow.com/questions/53667174/elimination-the-linear-dependent-columns-of-a-non-square-matrix-in-python
    """
    m,n = coeff_matrix.shape
    assert m <= n, print(f"m={m} n={n}")

    q, r = np.linalg.qr(coeff_matrix.T)
    diagonals = np.diagonal(r)
    indices = np.argwhere(np.abs(diagonals) > zero_tolerance)
    return coeff_matrix[indices.flatten(), :]


def find_free_variable_indices_by_row_echelon(A:np.ndarray) -> typing.Set[int]:
    """
    When a matrix is in echelon form: 
    A = [[1, a, b, c],
         [0, 0, 1, e],
         [0, 0, 0, 1],
         [0, 0, 0, 0]]
    we find the position of the first non-zero element in a row
    set({0, 2, 3})
    the free variable index is therefore {1}
    The leading variables are the ones that can be solved given the free variables. 
    See Page 13. Nicholson, 2020, Linear Algebra with Applications
    Note that we also removed the rows with all zeros
    """
    echelon = _row_echelon(A)
    reduced_echelon = remove_zero_vector_from_coefficients(echelon)

    leading_variables = {_first_non_zero(row) for row in reduced_echelon}
    free_variables = set(range(reduced_echelon.shape[1])) - leading_variables
    
    return free_variables


def _first_non_zero(row: np.ndarray) -> int:
    close_to_zero = np.argwhere(np.abs(row) > zero_tolerance)
    return int(close_to_zero[0])


def _row_echelon(A):
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
        B = _row_echelon(A[:,1:])
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
    B = _row_echelon(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])


'''
this should be removed in due time

def _wrong_find_free_variable_indices(A:np.ndarray) -> typing.Set[int]:
    """
    I followed this method, which worked for some case but it is not correct
    the problem is that U is an upper triangluar matrix, but not necessarily
    a row echelon form, so that free variable idenfitication is not correct.
    """
    # https://stackoverflow.com/questions/52303550/find-which-variables-are-free-in-underdetermined-linear-system
    P, L, U = scipy.linalg.lu(A)
    basis_columns = {_first_non_zero(row) for row in U}
    free_variables = set(range(U.shape[1])) - basis_columns
    return free_variables
'''
