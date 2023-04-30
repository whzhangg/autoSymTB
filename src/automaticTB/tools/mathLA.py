import typing
import dataclasses

import numpy as np

from automaticTB.parameters import ztol
from .cython.rref_cython import cython_cp_wrapper, cython_spp_wrapper

def get_distinct_nonzero_vector_from_coefficients(
        coeff_matrix: np.ndarray) -> np.ndarray:
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
    """remove additional parallel vector (both direction.)

    It return a list of vectors that does not contain the same 
    direction by checking the generalized cos angle (absolute value) 
    between two vectors. 
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

    _, r = np.linalg.qr(coeff_matrix.T)
    diagonals = np.diagonal(r)
    indices = np.argwhere(np.abs(diagonals) > ztol)
    return coeff_matrix[indices.flatten(), :]


def find_free_indices_by_pivoting(matrix: np.ndarray, tol: float, method: str = 'spp'):
    """perform guassian elimination. copies the input matrix

    supported method: 
    - 'ssp': scaled partial pivoting
    - 'cp': complete pivoting 
    
    perform row elimination and return the row eliminated matrix where 
    coloum containing entry at the main diagonal is completely 
    eliminated. The second return is the free index. 

    For example, the output matrix looks like this
    A = [[1, a, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    we find the position of the first non-zero element in a row
    set({0, 2, 3}). the free variable index is therefore {1}.
    The leading variables are the ones that can be solved given the 
    free variables. Note that we also removed the rows with all zeros

    reference: 
    - Nicholson, 2020, Linear Algebra with Applications, 
    - Burden's Numerical Analysis book
    """
    def _first_non_zero(row: np.ndarray) -> int:
        close_to_zero = np.argwhere(np.abs(row) > ztol)
        return int(close_to_zero[0])

    if method == 'spp':
        solver = cython_spp_wrapper
    elif method == 'cp':
        solver = cython_cp_wrapper
    else:
        raise ValueError("Pivoting method {method} is not supported.")

    matrix_to_solve = matrix.copy()    
    colperm = solver(matrix_to_solve, tol)
    row_is_zero = np.all(np.isclose(matrix_to_solve, 0.0, atol=tol), axis=1)
    row_is_not_zero = np.invert(row_is_zero)
    
    remain = matrix_to_solve[row_is_not_zero][:,colperm]
    leading_variables = {_first_non_zero(row) for row in remain}
    free_variables = set(range(remain.shape[1])) - leading_variables
    free_variables = sorted(list(free_variables))
    true_free_indices = [colperm[f] for f in free_variables]

    return matrix_to_solve[row_is_not_zero,:], true_free_indices
    

def solve_matrix(
        A: np.ndarray, indices: typing.List[int], values: typing.List[float]) -> np.ndarray:
    """solve x of an row-deficit equation providing some values of x"""
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

    def find_solvable(
        self, solved_indices: typing.List[int]
    ) -> typing.Tuple[typing.List[int],typing.List[int]]:
        """find a submatrix that is solvable

        In row_echelon_form, given a set of indices that is a subset 
        of the 'free_variable_indices', return the row and column number
        so that all the columns can be solved knowing the index in 
        solved_indices.

        We have len(nrow) + len(solved_indices) == len(ncol)
        """
        for i in solved_indices:
            if i not in self.free_variable_indices:
                print("!! LinearEquation.find_solvable() require", 
                      " the solvable index to be a sutset of free_variable_indices")
                print("solved_indices: ", solved_indices)
                print("free_variable: ", self.free_variable_indices)
                raise ValueError
        
        other_indices = [li for li in self.free_variable_indices if li not in solved_indices]
        not_zero = np.logical_not(np.isclose(self.row_echelon_form, 0.0, atol=ztol))

        right_part_solved = not_zero[:, np.array(solved_indices)]
        related_row = np.any(right_part_solved, axis = 1)

        if len(other_indices) == 0:
            row_can_be_solved = list(np.where(related_row)[0])
        else:
            right_part_unkownn = not_zero[:, np.array(other_indices)]
            cannot_be_solved_row = np.any(right_part_unkownn, axis = 1)
            row_can_be_solved = list(np.where(
                np.logical_and(related_row, np.logical_not(cannot_be_solved_row)))[0])

        if len(row_can_be_solved) == 0:
            # no symmetry relationship
            return [], solved_indices
        
        solvable_index = np.where(np.any(not_zero[np.array(row_can_be_solved)], axis=0))[0]

        return row_can_be_solved, solvable_index

    @classmethod
    def from_equation(cls, equation: np.ndarray) -> "LinearEquation":
        input_equation = equation.copy()

        processed_equation, free_variable_indices = find_free_indices_by_pivoting(
            input_equation, ztol, method='cp')
        # stable for 1e-6 tolerance

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


    def find_remaining_free_indices(self, provided_free_indices) -> typing.List[int]:
        nprovided = len(provided_free_indices)
        _, ncol = self.row_echelon_form.shape
        additional_rows = np.zeros((nprovided, ncol), dtype=self.row_echelon_form.dtype)
        for i, ifree in enumerate(provided_free_indices):
            additional_rows[i, ifree] = 1.0
        stacked = np.vstack([self.row_echelon_form, additional_rows])
        _, free_variable_indices = find_free_indices_by_pivoting(
            stacked, ztol)
        return free_variable_indices


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

