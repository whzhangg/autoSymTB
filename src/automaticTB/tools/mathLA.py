import typing
import dataclasses
import numpy as np

from automaticTB.parameters import ztol


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
    """This function remove rows of zero from an array"""
    #is_zero = np.isclose(np.linalg.norm(coeff_matrix, axis = 1), 0.0, atol = ztol)
    is_zero = np.all(np.isclose(coeff_matrix, 0.0, atol=ztol), axis=1)
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
            if np.isclose(np.abs(cos), 1.0, atol=ztol):
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
    indices = np.argwhere(np.abs(diagonals) > ztol)
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
    echelon = row_echelon(A)
    reduced_echelon = remove_zero_vector_from_coefficients(echelon)

    leading_variables = {first_non_zero(row) for row in reduced_echelon}
    free_variables = set(range(reduced_echelon.shape[1])) - leading_variables
    
    return free_variables


def first_non_zero(row: np.ndarray) -> int:
    """first non zero element """
    close_to_zero = np.argwhere(np.abs(row) > ztol)
    try:
        return int(close_to_zero[0])
    except IndexError as error:
        print(row)
        print("Zero not found")
        raise error


def row_echelon(A):
    """Return Row Echelon Form of matrix A 
    
    reference: https://math.stackexchange.com/questions/3073083/how-to-reduce-matrix-into-row-echelon-form-in-numpy
    """
    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    #A[np.abs(A) < ztol] = 0.0
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if np.abs(A[i,0]) > ztol: break
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


def row_echelon_inplace(A, tol) -> np.ndarray:
    """Return Row Echelon Form of matrix A in place
    
    This version is faster than row_echelon()
    reference: https://math.stackexchange.com/questions/3073083/how-to-reduce-matrix-into-row-echelon-form-in-numpy
    """
    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    #A[np.abs(A) < ztol] = 0.0
    if r == 0 or c == 0:
        return

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if np.abs(A[i,0]) > tol: break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        row_echelon_inplace(A[:,1:], tol)
        # and then add the first zero-column back
        return

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
    row_echelon_inplace(A[1:,1:], tol)

    # we add first row and first (zero) column, and return
    return


def row_echelon_scaled_partial_pivoting(A: np.ndarray, tol: float) -> np.ndarray:
    scale = np.max(np.abs(A), axis = 1)
    if np.any(np.isclose(scale, 0.0)): 
        print(scale)
        raise RuntimeError
    _row_echelon_spp(A, scale, tol)


def _row_echelon_spp(matrix: np.ndarray, scale: np.ndarray, tol: float) -> np.ndarray:
    
    nr, nc = matrix.shape
    if nr == 0 or nc == 0:
        return

    # choose the pivot position
    scaled_x1 = np.abs(matrix[:,0]) / scale
    pivot_index = -1
    largest = tol
    for i, sx in enumerate(scaled_x1):
        if sx > largest:
            largest = sx
            pivot_index = i

    if pivot_index == -1:
        _row_echelon_spp(matrix[:,1:], scale, tol)
    else:
        # swap 0th row and ith row
        swap_row = matrix[pivot_index].copy()
        matrix[pivot_index] = matrix[0]
        matrix[0] = swap_row
        # for scale also
        swap = scale[pivot_index]
        scale[pivot_index] = scale[0]
        scale[0] = swap

        # use first row to eliminate all other rows
        matrix[0] = matrix[0] / matrix[0,0]
        matrix[1:] -= matrix[0] * matrix[1:,0:1]

        _row_echelon_spp(matrix[1:,1:], scale[1:], tol)


def row_echelon_sympy(A):
    """reduced row echelon form using sympy, very slow"""
    import sympy as sp
    matrix = sp.Matrix(A)
    ref, _ = matrix.rref()
    return np.array(ref, dtype=A.dtype)


def solve_matrix(
    A: np.ndarray, indices: typing.List[int], values: typing.List[float]
) -> np.ndarray:
    nrow, ncol = A.shape
    nfree = len(values)
    new_rows = np.zeros((nfree, ncol), dtype=A.dtype)
    for i, fi in enumerate(indices):
        new_rows[i, fi] = 1.0

    stacked_left = np.vstack([A, new_rows])
    stacked_right= np.hstack(
        [ np.zeros(nrow, dtype=A.dtype), np.array(values, dtype=A.dtype) ]
    )

    return np.dot(np.linalg.inv(stacked_left), stacked_right)


@dataclasses.dataclass
class LinearEquation:
    input_equation: np.ndarray
    row_echelon_form: np.ndarray
    free_variable_indices: typing.List[int]  # could be empty

    @classmethod
    def from_equation(cls, equation: np.ndarray) -> "LinearEquation":
        input_equation = equation.copy()
        #input_equation = remove_zero_vector_from_coefficients(input_equation)
        row_echelon_inplace(input_equation, tol=ztol)
        #row_echelon_scaled_partial_pivoting(input_equation, tol=ztol)
        processed_equation = remove_zero_vector_from_coefficients(input_equation)
        leading_varaibles_indices = {first_non_zero(row) for row in processed_equation}
        free_variable_indices = list(
            set(range(processed_equation.shape[1])) - leading_varaibles_indices
        )

        nrow, ncol = processed_equation.shape
        if nrow + len(free_variable_indices) != ncol:
            print(f"{cls.__name__}: finding free variable indices not successful")
            print(
                "echelon form shape: ", processed_equation.shape, 
                f" found free indices: {len(free_variable_indices)}"
            )
            raise RuntimeError

        return cls(
            equation, processed_equation, sorted(free_variable_indices)
        )

    def solve_with_values(self, input_values: typing.List[float]) -> np.ndarray:
        """this functionality solve the all the interactions from the free ones"""

        nrow, ncol = self.row_echelon_form.shape
        nfree = ncol - nrow
        if len(self.free_variable_indices) != len(input_values):
            raise RuntimeError(
                "the input values do not fit with row_echelon_form with shape {:d},{:d}".format\
                (*self.row_echelon_form.shape)
            )

        return solve_matrix(self.row_echelon_form, self.free_variable_indices, input_values)
        data_type = self.row_echelon_form.dtype
        new_rows = np.zeros((nfree, ncol), dtype=data_type)
        for i, fi in enumerate(self.free_variable_indices):
            new_rows[i, fi] = 1.0

        stacked_left = np.vstack([self.row_echelon_form, new_rows])
        stacked_right= np.hstack(
            [ np.zeros(nrow, dtype=data_type),
              np.array(input_values, dtype=data_type) ]
        )

        return np.dot(np.linalg.inv(stacked_left), stacked_right)

