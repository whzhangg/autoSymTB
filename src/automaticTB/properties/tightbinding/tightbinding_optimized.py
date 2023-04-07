import typing
from collections import namedtuple

import numpy as np
import scipy
import joblib

from automaticTB import tools
from automaticTB.properties import reciprocal
from automaticTB import parameters
from .hij import Pindex_lm, HijR, SijR

HSijR = namedtuple('HSijR', "i j r H S")

class TightBindingModel:
    """faster version of the tightbinding model with some difference"""
    nproc = joblib.cpu_count() // 2
    parallel_threshold = 10000
    def __init__(self, 
        cell: np.ndarray, positions: np.ndarray, types: typing.List[int],
        HijR_list: typing.List[HijR],
        SijR_list: typing.Optional[typing.List[SijR]] = None
    ) -> None:
        """
        we assert HijR and SijR are the same list with the same pairs 
        and we put them in tuples with index
        """
        self._cell = cell
        self._positions = positions
        self._types = types

        self._basis: typing.List[Pindex_lm] = []
        for hijR in HijR_list:
            if hijR.left not in self._basis:
                self._basis.append(hijR.left)
                
        index_ref: typing.Dict[tuple, int] = {
            (basis.pindex, basis.n, basis.l, basis.m):i for i, basis in enumerate(self._basis)
        }

        if SijR_list is None:
            # create SijR list if not supplied
            SijR_list = []
            for hijr in HijR_list:
                if hijr.left == hijr.right:
                    SijR_list.append(
                        SijR(hijr.left, hijr.right, 1.0)
                    )
                else:
                    SijR_list.append(
                        SijR(hijr.left, hijr.right, 0.0)
                    )

        debug_print = False

        self._HSijRs = []
        for hijr in HijR_list:
            index_i = index_ref[
                (hijr.left.pindex, hijr.left.n, hijr.left.l, hijr.left.m)
            ]
            index_j = index_ref[
                (hijr.right.pindex, hijr.right.n, hijr.right.l, hijr.right.m)
            ]

            r = hijr.right.translation + \
                self._positions[hijr.right.pindex] - self._positions[hijr.left.pindex]

            if debug_print:
                print(f"({index_i+1:>2d},{index_j+1:>2d})", end=" ") 
                print(f"H = {hijr.value.real:>10.5f}", end=" ")
                print("t_r = {:>6.3f},{:>6.3f},{:>6.3f}".format(*hijr.right.translation), end=" ")
                print("t_l = {:>6.3f},{:>6.3f},{:>6.3f}".format(*hijr.left.translation), end=" ")
                print("p_r = {:>6.3f},{:>6.3f},{:>6.3f}".format(*self._positions[hijr.right.pindex]), end=" ")
                print("p_l = {:>6.3f},{:>6.3f},{:>6.3f}".format(*self._positions[hijr.left.pindex]), end=" ")
                print(f"r = {r[0]:>6.3f}{r[1]:>6.3f}{r[2]:>6.3f}")

            hvalue = hijr.value
            found = False
            for sijr in SijR_list:
                if sijr.left == hijr.left and sijr.right == hijr.right:
                    svalue = sijr.value
                    found = True
                    break
            
            if not found:
                raise RuntimeError("cannot find correcponding sijr")

            self._HSijRs.append(
                HSijR(
                    index_i, index_j, r, hvalue, svalue
                )
            )
        

    @property
    def reciprocal_cell(self) -> np.ndarray:
        return reciprocal.find_RCL(self.cell)


    @property
    def basis(self) -> typing.List[Pindex_lm]:
        return self._basis


    @property
    def basis_name(self) -> typing.List[str]:
        names = []
        for b in self._basis:
            orbital_symbol = tools.get_orbital_symbol_from_lm(b.l, b.m)
            chem_symbol = tools.chemical_symbols[self.types[b.pindex]]
            names.append(f"{chem_symbol}({b.pindex+1}) {b.n}{orbital_symbol}")
        return names


    @property
    def nbasis(self) -> int:
        return len(self._basis)


    @property
    def cell(self) -> np.ndarray:
        return self._cell


    @property
    def positions(self) -> np.ndarray:
        return self._positions


    @property
    def types(self) -> typing.List[int]:
        return self._types


    def Hijk_Sijk_and_derivatives_at_k(
            self, k: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """return H(k), S(k), dH(k)/dk and dS(k)/dk at a given k"""
        ks = np.array([k])

        hijk, sijk, hijk_derv, sijk_derv = \
            self.Hijk_SijK_and_derivatives(ks, True)
        
        return hijk[0], sijk[0], hijk_derv[0], sijk_derv[0]


    def Hijk_SijK_and_derivatives(
        self, ks: np.ndarray, require_derivative: bool = True
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """return H(k), S(k), dH(k)/dk and dS(k)/dk at a list of k"""
        ks = np.array(ks)
        nk = len(ks)
        nbasis = len(self._basis)
    
        cell_T = self._cell.T
        pi = np.pi

        if require_derivative:
            hijk = np.zeros((nk, nbasis,nbasis), dtype=complex)
            hijk_derv = np.zeros((nk, 3, nbasis, nbasis), dtype=complex)
            sijk = np.zeros((nk, nbasis,nbasis), dtype=complex)
            sijk_derv = np.zeros((nk, 3, nbasis, nbasis), dtype=complex)

            for hsijr in self._HSijRs:
                index_i, index_j, r, hvalue, svalue = hsijr
                
                kR = np.dot(ks, r)
                tmph = hvalue * np.exp(-2j * pi * kR) # use -2 instead of 2
                tmps = svalue * np.exp(-2j * pi * kR)

                hijk[:, index_i, index_j] += tmph
                sijk[:, index_i, index_j] += tmps

                r_cart_x, r_cart_y, r_cart_z = cell_T @ r # in A
                hijk_derv[:, 0, index_i, index_j] += tmph * 1j * r_cart_x
                hijk_derv[:, 1, index_i, index_j] += tmph * 1j * r_cart_y
                hijk_derv[:, 2, index_i, index_j] += tmph * 1j * r_cart_z 
                sijk_derv[:, 0, index_i, index_j] += tmps * 1j * r_cart_x
                sijk_derv[:, 1, index_i, index_j] += tmps * 1j * r_cart_y
                sijk_derv[:, 2, index_i, index_j] += tmps * 1j * r_cart_z 

            return hijk, sijk, hijk_derv, sijk_derv
        else: 
            hijk = np.zeros((nk, nbasis,nbasis), dtype=complex)
            sijk = np.zeros((nk, nbasis,nbasis), dtype=complex)

            for hsijr in self._HSijRs:
                index_i, index_j, r, hvalue, svalue = hsijr
                
                kR = np.dot(ks, r)
                tmph = hvalue * np.exp(-2j * pi * kR)
                tmps = svalue * np.exp(-2j * pi * kR)

                hijk[:, index_i, index_j] += tmph
                sijk[:, index_i, index_j] += tmps

            return hijk, sijk, (), ()


    def solveE_at_ks(self, ks: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """return energy and eigen-vector for a list of k points
        
        -> (eigenvalues, eigenvectors)
        For each k point, the eigen values are sorted
        The outputs have shape:
         - eigenvalue has shape (nk, nbnd)
         - eigenvector has shape (nk, nstate, nbnd)
        """
        ks = np.array(ks)
        hijk, sijk, _, _ = self.Hijk_SijK_and_derivatives(ks, require_derivative=False)
        if len(ks) < self.parallel_threshold:
            return _solve_E(hijk, sijk)
        else:
            #print(f"calculate in parallel for {len(ks)} kpoints on {self.nproc} process")
            divided_index = _divide_jobs(len(ks), self.nproc)
            divided_jobs = [
                (hijk[start:end], sijk[start:end]) for start, end in divided_index
            ]
            #njob_each = len(ks) // self.nproc + 1
            #divided_jobs = []
            #for iproc in range(self.nproc):
            #    start = njob_each * iproc
            #    end = min(njob_each * (iproc + 1), len(ks))
            #    divided_jobs.append((hijk[start:end], sijk[start:end]))
            
            results = joblib.Parallel(n_jobs=self.nproc)(
                joblib.delayed(_solve_E)(hs, ss) for hs, ss in divided_jobs)
            
            energies = np.vstack([e for e, _ in results])
            vector = np.vstack([c for _, c in results])
            return energies, vector
    

    def solveE_V_at_ks(
        self, ks: np.ndarray, average_degenerate = False
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """provide energy and derivative for a list of kpoints
        
        -> (eigenvalues, velocity, eigenvectors)
        For each k point, the eigen values are sorted
        The outputs have shape:
         - eigenvalue has shape (nk, nbnd)
         - bandvelocity has shape (nk, nbnd, 3)
         - eigenvector has shape (nk, nstate, nbnd)
         """
        ks = np.array(ks)
        hijk, sijk, hijk_derv, sijk_derv \
            = self.Hijk_SijK_and_derivatives(ks, require_derivative=True)

        if len(ks) < self.parallel_threshold:
            obtained_energy, obtained_velocity, obtained_coefficient \
                = _solve_E_V(hijk, sijk, hijk_derv, sijk_derv)
        else:
            divided_index = _divide_jobs(len(ks), self.nproc)
            divided_jobs = [
                (hijk[start:end], sijk[start:end], hijk_derv[start:end], sijk_derv[start:end])
                for start, end in divided_index
            ]
            #njob_each = len(ks) // self.nproc + 1
            #divided_jobs = []
            #for iproc in range(self.nproc):
            #    start = njob_each * iproc
            #    end = min(njob_each * (iproc + 1), len(ks))
            #    divided_jobs.append(
            #        (hijk[start:end], sijk[start:end], 
            #         hijk_derv[start:end], sijk_derv[start:end]))
            
            results = joblib.Parallel(n_jobs=self.nproc)(
                joblib.delayed(_solve_E_V)(hs, ss, dhs, dss) for hs, ss, dhs, dss in divided_jobs)
            
            obtained_energy = np.vstack([e for e,_,_ in results])
            obtained_velocity = np.vstack([v for _,v,_ in results])
            obtained_coefficient = np.vstack([c for _,_,c in results])

        if average_degenerate:
            obtained_velocity = _get_averaged_degenerate_velocity(
                obtained_energy, obtained_velocity
            )
        return obtained_energy, obtained_velocity, obtained_coefficient



def _get_averaged_degenerate_velocity(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """average velocity at degenerate points"""
    nk, nbnd = w.shape
    v_new = v.copy()
    for ik in range(nk):
        for ibnd in range(nbnd):
            deg = np.isclose(w[ik, ibnd], w[ik,:], atol = parameters.ztol)
            v_new[ik, ibnd, :] = np.average(v[ik, deg, :], axis=0)
    return v_new


def _divide_jobs(nk, nproc):
    divided_index = []
    njob_each = nk // nproc + 1
    for iproc in range(nproc):
        start = njob_each * iproc
        end = min(njob_each * (iproc + 1), nk)
        divided_index.append((start, end))
    return divided_index


def _solve_E(hijks: np.ndarray, sijks: np.ndarray) -> np.ndarray:
    """worker function, hijks and sijks have shape (nk, nbnd, nbnd)
    
    For each k point, the eigen values are sorted

    The calculated eigenvalue has shape (nk, nbnd)
                 eigen vector has shape (nk, nstate, nbnd)
    Note, the last index of the eigenvector gives the band index.
    """
    nk, nbasis, _ = hijks.shape
    eigenvalues = np.zeros((nk, nbasis), dtype=np.double)
    eigenvectors = np.zeros((nk, nbasis, nbasis), dtype=np.cdouble)

    for ik in range(nk):
        w, c = scipy.linalg.eig(hijks[ik], sijks[ik])
        sort_indices = np.argsort(w.real)
        eigenvalues[ik] = w.real[sort_indices]
        eigenvectors[ik] = c[:,sort_indices]
    
    for ik in range(nk):
        for ibnd in range(nbasis):
            left = hijks[ik] @ eigenvectors[ik,:,ibnd]
            right = eigenvalues[ik, ibnd] * sijks[ik] @ eigenvectors[ik,:,ibnd]
            assert np.allclose(left, right, atol = 1e-4)
    return eigenvalues, eigenvectors


def _solve_E_V(
    hijks: np.ndarray, sijks: np.ndarray, hijk_dervs: np.ndarray, sijk_dervs: np.ndarray
) -> np.ndarray:
    """worker function for mpi, eigenvalue in eV and V is in m/s
    
    the calculated eigenvalue has shape (nk, nbnd)
    """
    nk, nbasis, _ = hijks.shape
    eigenvalues = np.zeros((nk, nbasis), dtype=np.double)
    eigenvectors = np.zeros((nk, nbasis, nbasis), dtype=np.cdouble)
    velocity = np.zeros((nk, nbasis, 3), dtype=np.double)

    for ik in range(nk):
                
        w, cT = scipy.linalg.eig(hijks[ik], sijks[ik]) # it has shape
        cs = cT.T

        derivative = []
        for epsilon, c in zip(w, cs):
            dhds = hijk_dervs[ik] - epsilon * sijk_dervs[ik] # (3, nbnd, nbnd)
            derivative.append(
                np.einsum("i, kij, j -> k", np.conjugate(c), dhds, c))

        #veloc_SI = (np.array(derivative) 
        #            * 1e-10 * scipy.constants.elementary_charge  / scipy.constants.hbar)
        # this velocity is in Ang/s
        # derivative = hbar^2 k / m with unit eV*A, velocity = 1/hbar de/dk
        # hbar in unit J/s. Therefore, velocity = 1/hbar de/dk in unit A/s if eV is converted to J
        veloc = (np.array(derivative) 
                * 1e-10 * scipy.constants.elementary_charge  / scipy.constants.hbar)
        sort_index = np.argsort(w.real)

        eigenvalues[ik] = w.real[sort_index]
        velocity[ik] = veloc.real[sort_index]
        eigenvectors[ik] = cT[:,sort_index]

    return eigenvalues, velocity, eigenvectors
    

"""
Note the following about ordering like c[:,:,order]

In [8]: order = np.array([1,0])

In [11]: a
Out[11]:
array([[[ 0,  1],
        [ 2,  3]],

       [[ 4,  5],
        [ 6,  7]],

       [[ 8,  9],
        [10, 11]],

       [[12, 13],
        [14, 15]],

       [[16, 17],
        [18, 19]]])

In [12]: a[0,:,order]
Out[12]:
array([[1, 3],
       [0, 2]])

In [13]: a[0][:,order]
Out[13]:
array([[1, 0],
       [3, 2]])

"""