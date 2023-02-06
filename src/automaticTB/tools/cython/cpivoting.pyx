import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def row_echelon_spp_cython_double(double[:,:] A, double tol, bint eliminate_above):
    """perform row echelon reduction with scaled partial pivot"""
    cdef Py_ssize_t nrow = A.shape[0]
    cdef Py_ssize_t ncol = A.shape[1]
    cdef Py_ssize_t rs = 0
    cdef Py_ssize_t cs = 0

    cdef Py_ssize_t[:] rowperm = np.arange(nrow, dtype=np.intp)
    cdef double[:] scale = np.zeros(nrow, dtype = np.double)

    cdef double largest, factor, scaled_value
    cdef Py_ssize_t swap, pivot_index, rowsel, rowstart

    for irow in range(nrow):
        largest = tol
        for icol in range(ncol):
            if abs(A[irow, icol]) > largest:
                largest = abs(A[irow, icol])
        scale[irow] = largest
            
    while rs < nrow and cs < ncol:
        
        pivot_index = -1
        largest = tol
        for ir in range(rs, nrow):
            rowsel = rowperm[ir]
            if abs(A[rowsel, cs]) > tol:
                scaled_value = abs(A[rowsel, cs]) / scale[rowsel]
                # in the worest case, scale and current number is just barely larger than tol 
                #and then round off error can be enlarged if it is selected a pivot 
                if scaled_value > largest:
                    largest = scaled_value
                    pivot_index = ir 

        if pivot_index == -1:
            cs += 1
            continue

        swap = rowperm[pivot_index]
        rowperm[pivot_index] = rowperm[rs]
        rowperm[rs] = swap
        
        rowsel = rowperm[rs]
        factor = A[rowsel, cs]
        for ic in range(cs, ncol):
            A[rowsel, ic] /= factor
        rowstart = rowperm[rs]

        if eliminate_above:
            for ir in range(rs):
                rowsel = rowperm[ir]
                factor = A[rowsel, cs]
                for ic in range(cs, ncol):
                    A[rowsel, ic] -= factor * A[rowstart, ic]

        for ir in range(rs+1, nrow):
            rowsel = rowperm[ir]
            factor = A[rowsel, cs]
            for ic in range(cs, ncol):
                A[rowsel, ic] -= factor * A[rowstart, ic]
        cs += 1
        rs += 1

    return np.arange(ncol, dtype = np.intp)

@cython.boundscheck(False)
@cython.wraparound(False)
def row_echelon_spp_cython_complex(double complex[:,:] A, double tol, bint eliminate_above):
    """perform row echelon reduction with scaled partial pivot"""
    cdef Py_ssize_t nrow = A.shape[0]
    cdef Py_ssize_t ncol = A.shape[1]
    cdef Py_ssize_t rs = 0
    cdef Py_ssize_t cs = 0

    cdef Py_ssize_t[:] rowperm = np.arange(nrow, dtype=np.intp)
    cdef double[:] scale = np.zeros(nrow, dtype = np.double)

    cdef double largest, scaled_value
    cdef double complex factor
    cdef Py_ssize_t swap, pivot_index, rowsel, rowstart

    for irow in range(nrow):
        largest = tol
        for icol in range(ncol):
            if abs(A[irow, icol]) > largest:
                largest = abs(A[irow, icol])
        scale[irow] = largest
            
    while rs < nrow and cs < ncol:
        
        pivot_index = -1
        largest = tol
        for ir in range(rs, nrow):
            rowsel = rowperm[ir]
            if abs(A[rowsel, cs]) > tol:
                scaled_value = abs(A[rowsel, cs]) / scale[rowsel]
                # in the worest case, scale and current number is just barely larger than tol 
                #and then round off error can be enlarged if it is selected a pivot 
                if scaled_value > largest:
                    largest = scaled_value
                    pivot_index = ir 

        if pivot_index == -1:
            cs += 1
            continue

        swap = rowperm[pivot_index]
        rowperm[pivot_index] = rowperm[rs]
        rowperm[rs] = swap
        
        rowsel = rowperm[rs]
        factor = A[rowsel, cs]
        for ic in range(cs, ncol):
            A[rowsel, ic] /= factor
        rowstart = rowperm[rs]

        if eliminate_above:
            for ir in range(rs):
                rowsel = rowperm[ir]
                factor = A[rowsel, cs]
                for ic in range(cs, ncol):
                    A[rowsel, ic] -= factor * A[rowstart, ic]

        for ir in range(rs+1, nrow):
            rowsel = rowperm[ir]
            factor = A[rowsel, cs]
            for ic in range(cs, ncol):
                A[rowsel, ic] -= factor * A[rowstart, ic]
        cs += 1
        rs += 1

    return np.arange(ncol, dtype = np.intp)

@cython.boundscheck(False)
@cython.wraparound(False)
def row_echelon_cp_cython_double(double[:,:] A, double tol, bint eliminate_above):
    """for real array"""
    cdef Py_ssize_t nrow = A.shape[0]
    cdef Py_ssize_t ncol = A.shape[1]
    cdef Py_ssize_t rs = 0
    cdef Py_ssize_t cs = 0

    cdef Py_ssize_t[:] rowperm = np.arange(nrow, dtype=np.intp)
    cdef Py_ssize_t[:] colperm = np.arange(ncol, dtype=np.intp)

    cdef Py_ssize_t max_row, max_col, rowsel, colsel, rowstart, colstart, swap
    cdef double largest, factor

    while rs < nrow and cs < ncol:
        
        max_row = -1
        max_col = -1
        largest = tol
        for ir in range(rs, nrow):
            for ic in range(cs, ncol):
                rowsel = rowperm[ir]
                colsel = colperm[ic]
                if A[rowsel, colsel] ** 2 > largest ** 2:
                    max_row = ir
                    max_col = ic
                    largest = A[rowsel, colsel]
                
        if max_row == -1 or max_col == -1:
            break 
        
        swap = rowperm[max_row]
        rowperm[max_row] = rowperm[rs]
        rowperm[rs] = swap

        swap = colperm[max_col]
        colperm[max_col] = colperm[cs]
        colperm[cs] = swap

        rowsel = rowperm[rs] # pivot row
        colstart = colperm[cs]
        factor = A[rowsel, colstart]
        for ic in range(cs, ncol):
            colsel = colperm[ic]
            A[rowsel, colsel] /= factor
        
        rowstart = rowperm[rs]
        colstart = colperm[cs]
        for ir in range(rs+1, nrow):
            rowsel = rowperm[ir]
            factor = A[rowsel, colstart]
            for ic in range(cs, ncol):
                colsel = colperm[ic]
                A[rowsel, colsel] -= factor * A[rowstart, colsel]
        
        if eliminate_above:
            for ir in range(rs):
                rowsel = rowperm[ir]
                factor = A[rowsel, colstart]
                for ic in range(cs, ncol):
                    colsel = colperm[ic]
                    A[rowsel, colsel] -= factor * A[rowstart, colsel]

        rs += 1
        cs += 1

    return colperm


@cython.boundscheck(False)
@cython.wraparound(False)
def row_echelon_cp_cython_complex(double complex[:,:] A, double tol, bint eliminate_above):
    """for complex array"""
    cdef Py_ssize_t nrow = A.shape[0]
    cdef Py_ssize_t ncol = A.shape[1]
    cdef Py_ssize_t rs = 0
    cdef Py_ssize_t cs = 0

    cdef Py_ssize_t[:] rowperm = np.arange(nrow, dtype=np.intp)
    cdef Py_ssize_t[:] colperm = np.arange(ncol, dtype=np.intp)

    cdef Py_ssize_t max_row, max_col, rowsel, colsel, rowstart, colstart, swap

    cdef double complex largest, factor

    while rs < nrow and cs < ncol:
        
        max_row = -1
        max_col = -1
        largest = tol
        for ir in range(rs, nrow):
            for ic in range(cs, ncol):
                rowsel = rowperm[ir]
                colsel = colperm[ic]
                if abs(A[rowsel, colsel]) > abs(largest):
                    max_row = ir
                    max_col = ic
                    largest = A[rowsel, colsel]
                
        if max_row == -1 or max_col == -1:
            break 
        
        swap = rowperm[max_row]
        rowperm[max_row] = rowperm[rs]
        rowperm[rs] = swap

        swap = colperm[max_col]
        colperm[max_col] = colperm[cs]
        colperm[cs] = swap

        rowsel = rowperm[rs] # pivot row
        colstart = colperm[cs]
        factor = A[rowsel, colstart]
        for ic in range(cs, ncol):
            colsel = colperm[ic]
            A[rowsel, colsel] /= factor
        
        rowstart = rowperm[rs]
        colstart = colperm[cs]
        for ir in range(rs+1, nrow):
            rowsel = rowperm[ir]
            factor = A[rowsel, colstart]
            for ic in range(cs, ncol):
                colsel = colperm[ic]
                A[rowsel, colsel] -= factor * A[rowstart, colsel]

        if eliminate_above:
            for ir in range(rs):
                rowsel = rowperm[ir]
                factor = A[rowsel, colstart]
                for ic in range(cs, ncol):
                    colsel = colperm[ic]
                    A[rowsel, colsel] -= factor * A[rowstart, colsel]
        rs += 1
        cs += 1

    return colperm