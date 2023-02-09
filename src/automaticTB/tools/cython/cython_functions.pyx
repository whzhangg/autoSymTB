import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def create_tetras(
    double[:,:] kpoints,
    Py_ssize_t[:,:] subcell_tetras,
    Py_ssize_t[:,:] mesh_shift,
    Py_ssize_t nk1, 
    Py_ssize_t nk2, 
    Py_ssize_t nk3
):
    cdef Py_ssize_t numk = kpoints.shape[0]
    cdef Py_ssize_t numt = subcell_tetras.shape[0]
    
    cdef Py_ssize_t[:,:] tetra = np.zeros((numk * 6, 4), dtype = np.intp)
    #xc = np.zeros(3)
    cdef Py_ssize_t ik, it, ic, corner, ixc, i, j, k

    for ik in range(numk):
        for it in range(numt):
            for ic in range(4):
                corner = subcell_tetras[it][ic]
                i = round(kpoints[ik,0] * nk1 + mesh_shift[corner,0] + 2 * nk1 ) % nk1
                j = round(kpoints[ik,1] * nk2 + mesh_shift[corner,1] + 2 * nk2 ) % nk2
                k = round(kpoints[ik,2] * nk3 + mesh_shift[corner,2] + 2 * nk3 ) % nk3
                ixc = k + j * nk3 + i * nk2 * nk3
                tetra[ik * 6 + it, ic] = ixc
    
    return tetra


@cython.boundscheck(False)
@cython.wraparound(False)
def c_calc_dos_contribution_multie(double[:] bandenergy, Py_ssize_t[:,:] tetras, double[:] e):
    cdef double[:] sum_dos = np.zeros_like(e, dtype = np.double)
    cdef Py_ssize_t num_tetra = len(tetras)
    cdef Py_ssize_t nenergy = len(e)
    cdef Py_ssize_t itetra, ienergy
    cdef double e1, e2, e3, e4, esingle
    cdef double a, b, c, d
    cdef double[:] sorted_energy = np.zeros(4, dtype = np.double)

    for itetra in range(num_tetra):
        a = bandenergy[tetras[itetra, 0]]
        b = bandenergy[tetras[itetra, 1]]
        c = bandenergy[tetras[itetra, 2]]
        d = bandenergy[tetras[itetra, 3]]
        
        if a < b:
            part1_min = a; part1_max = b
        else:
            part1_min = b; part1_max = a
        if c < d:
            part2_min = c; part2_max = d
        else:
            part2_min = d; part2_max = c

        if part1_min < part2_min:
            e1 = part1_min
            left1 = part2_min
        else:
            e1 = part2_min
            left1 = part1_min
        
        if part1_max > part2_max:
            e4 = part1_max
            left2 = part2_max
        else:
            e4 = part2_max
            left2 = part1_max

        if left1 < left2:
            e2 = left1
            e3 = left2
        else:
            e2 = left2
            e3 = left1
        
        for ienergy in range(nenergy):
            esingle = e[ienergy]
            if esingle <= e1:
                pass

            elif e1 <= esingle <= e2:
                if e1 == e2:
                    pass
                else:
                    sum_dos[ienergy] += 3*(esingle - e1)**2/((e2 - e1)*(e3 - e1)*(e4 - e1)) / num_tetra

            elif e2 <= esingle <= e3:
                if e2 == e3:
                    sum_dos[ienergy] += 3.0 * (e2 - e1) / ((e3 - e1) * (e4 - e1)) / num_tetra
                else:
                    fac = 1.0 / ((e3 - e1)*(e4 - e1))
                    elin = 3*(e2 - e1) + 6*(esingle - e2)
                    esq = -3*(((e3 - e1) + (e4 - e2))/((e3 - e2)*(e4 - e2))) * (esingle - e2)**2
                    sum_dos[ienergy] += fac * (elin + esq) / num_tetra

            elif e3 <= esingle <= e4:
                if e3 == e4:
                    pass
                else:
                    sum_dos[ienergy] += 3 *(e4 - esingle)**2/((e4 - e1)*(e4 - e2)*(e4 - e3)) / num_tetra

            else:
                # E >= E4
                pass

    return sum_dos


# for DOS calculation
@cython.boundscheck(False)
@cython.wraparound(False)
def c_calc_dos_contribution(double[:] bandenergy, Py_ssize_t[:,:] tetras, double e):
    cdef double sum_dos = 0.0
    cdef Py_ssize_t num_tetra = len(tetras)
    cdef Py_ssize_t itetra
    cdef double e1, e2, e3, e4
    cdef double a, b, c, d
    cdef double[:] sorted_energy = np.zeros(4, dtype = np.double)

    for itetra in range(num_tetra):
        a = bandenergy[tetras[itetra, 0]]
        b = bandenergy[tetras[itetra, 1]]
        c = bandenergy[tetras[itetra, 2]]
        d = bandenergy[tetras[itetra, 3]]
        
        if a < b:
            part1_min = a; part1_max = b
        else:
            part1_min = b; part1_max = a
        if c < d:
            part2_min = c; part2_max = d
        else:
            part2_min = d; part2_max = c

        if part1_min < part2_min:
            e1 = part1_min
            left1 = part2_min
        else:
            e1 = part2_min
            left1 = part1_min
        
        if part1_max > part2_max:
            e4 = part1_max
            left2 = part2_max
        else:
            e4 = part2_max
            left2 = part1_max

        if left1 < left2:
            e2 = left1
            e3 = left2
        else:
            e2 = left2
            e3 = left1
        
        if e <= e1:
            pass

        elif e1 <= e <= e2:
            if e1 == e2:
                pass
            else:
                sum_dos += 3*(e - e1)**2/((e2 - e1)*(e3 - e1)*(e4 - e1))

        elif e2 <= e <= e3:
            if e2 == e3:
                sum_dos += 3.0 * (e2 - e1) / ((e3 - e1) * (e4 - e1))
            else:
                fac = 1.0 / ((e3 - e1)*(e4 - e1))
                elin = 3*(e2 - e1) + 6*(e - e2)
                esq = -3*(((e3 - e1) + (e4 - e2))/((e3 - e2)*(e4 - e2))) * (e - e2)**2
                sum_dos += fac * (elin + esq)

        elif e3 <= e <= e4:
            if e3 == e4:
                pass
            else:
                sum_dos += 3 *(e4 - e)**2/((e4 - e1)*(e4 - e2)*(e4 - e3))

        else:
            # E >= E4
            pass

    sum_dos /= num_tetra

    return sum_dos


@cython.boundscheck(False)
@cython.wraparound(False)
def c_calc_dos_nsum(double[:] bandenergy, Py_ssize_t[:,:] tetras, double e):
    cdef double nsum = 0.0
    cdef Py_ssize_t num_tetra = len(tetras)
    cdef Py_ssize_t itetra
    cdef double e1, e2, e3, e4
    cdef double a, b, c, d
    cdef double[:] sorted_energy = np.zeros(4, dtype = np.double)

    for itetra in range(num_tetra):
        a = bandenergy[tetras[itetra, 0]]
        b = bandenergy[tetras[itetra, 1]]
        c = bandenergy[tetras[itetra, 2]]
        d = bandenergy[tetras[itetra, 3]]
        
        if a < b:
            part1_min = a; part1_max = b
        else:
            part1_min = b; part1_max = a
        if c < d:
            part2_min = c; part2_max = d
        else:
            part2_min = d; part2_max = c

        if part1_min < part2_min:
            e1 = part1_min
            left1 = part2_min
        else:
            e1 = part2_min
            left1 = part1_min
        
        if part1_max > part2_max:
            e4 = part1_max
            left2 = part2_max
        else:
            e4 = part2_max
            left2 = part1_max

        if left1 < left2:
            e2 = left1
            e3 = left2
        else:
            e2 = left2
            e3 = left1
        
        if e <= e1:
            pass

        elif e1 <= e <= e2:
            if e1 == e2:
                pass
            else:
                nsum += 1.0 * (e - e1)**3 / ((e2 - e1)*(e3 - e1)*(e4 - e1))

        elif e2 <= e <= e3:
            if e2 == e3:
                nsum += 1.0 * (e2 - e1) * (e2 - e1) / ((e3 - e1) * (e4 - e1))
            else:
                fac = 1.0 / ((e3 - e1)*(e4 - e1))
                esq = (e2 - e1)**2 + 3.0*(e2 - e1)*(e - e2) + 3.0*(e - e2)**2
                ecub = -(((e3 - e1) + (e4 - e2))/((e3 - e2)*(e4 - e2))) * (e - e2)**3
                nsum += fac * (esq + ecub)

        elif e3 <= e <= e4:
            if e3 == e4:
                nsum += 1.0
            else:
                nsum += 1.0 * (1.0 - (e4 - e)**3/((e4 - e1)*(e4 - e2)*(e4 - e3)))

        else:
            # E >= E4
            nsum += 1.0
        
    nsum /= num_tetra
    return nsum

#################################################
# for math
#################################################

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