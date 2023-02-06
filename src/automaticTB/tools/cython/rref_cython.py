import numpy as np

from .cpivoting import row_echelon_cp_cython_double, row_echelon_cp_cython_complex
from .cpivoting import row_echelon_spp_cython_double, row_echelon_spp_cython_complex

# cython 
def cython_cp_wrapper(A: np.ndarray, tol: float, eliminate_above: bool = True) -> np.ndarray:
    if A.dtype == np.cdouble:
        perm = row_echelon_cp_cython_complex(A, tol, eliminate_above)
        return perm
    elif A.dtype == np.double:
        perm = row_echelon_cp_cython_double(A, tol, eliminate_above)
        return perm
    else:
        raise RuntimeError(f"Data type of array {A.dtype} is not supported!")

# cython 
def cython_spp_wrapper(A: np.ndarray, tol: float, eliminate_above: bool = True) -> np.ndarray:
    if A.dtype == np.cdouble:
        perm = row_echelon_spp_cython_complex(A, tol, eliminate_above)
        return perm
    elif A.dtype == np.double:
        perm = row_echelon_spp_cython_double(A, tol, eliminate_above)
        return perm
    else:
        raise RuntimeError(f"Data type of array {A.dtype} is not supported!")


def row_echelon_inplace(A: np.ndarray, tol) -> None:
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