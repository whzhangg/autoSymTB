import typing

import numpy as np
import scipy
from collections import namedtuple

from automaticTB.properties import kpoints
from .hij import Pindex_lm, HijR, SijR

HSijR = namedtuple('HSijR', "i j r H S")

class TightBindingModel:
    """faster version of the tightbinding model with some difference"""
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
        return kpoints.find_RCL(self.cell)


    @property
    def basis(self) -> typing.List[Pindex_lm]:
        return self._basis


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


    def solveE_V_at_k(self, k: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """provide energy and band derivative at k"""
        # it now wraps over solveE_at_ks version
        ks = np.array([k])
        es, vs = self.solveE_V_at_ks(ks)
        return es[0], vs[0]

            
    def solveE_c2_at_ks(self, ks: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """provide energies and coefficients**2, for fat band plot"""
        hijk, sijk, _, _ = self.Hijk_SijK_and_derivatives(ks, require_derivative=False)
        result = []
        coefficients = []
        for h, s in zip(hijk, sijk):
            #h = (h + np.conjugate(h.T)) / 2.0
            #s = (s + np.conjugate(s.T)) / 2.0  # should symmetrize hamiltonian
            w, c = scipy.linalg.eig(h, s)
            # c is indexed by c[iorb, ibnd], 
            # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html
            cT = (np.abs(c)**2).T  # [ibnd, iorb]
            coefficients.append(cT[np.newaxis,...])
            result.append(np.sort(w.real))

        return np.array(result), np.vstack(coefficients)
        
    #@profile
    # self.Hijk_SijK_and_derivatives took around 14%
    # scipy.linalg.eig(h, s) toke ~ 85% of the time
    def solveE_at_ks(self, ks: np.ndarray) -> np.ndarray:
        """provide energies for a list of k points"""
        hijk, sijk, _, _ = self.Hijk_SijK_and_derivatives(ks, require_derivative=False)
        if len(ks) < 1000:
            result = []
            
            for h, s in zip(hijk, sijk):
                #h = (h + np.conjugate(h.T)) / 2.0
                #s = (s + np.conjugate(s.T)) / 2.0  # should symmetrize hamiltonian
                w, c = scipy.linalg.eig(h, s)
                result.append(np.sort(w.real))
        else:
            # parallel version, for the moment the same as serial version
            result = []
            for h, s in zip(hijk, sijk):
                #h = (h + np.conjugate(h.T)) / 2.0
                #s = (s + np.conjugate(s.T)) / 2.0  # should symmetrize hamiltonian
                w, c = scipy.linalg.eig(h, s)
                result.append(np.sort(w.real))
        return np.array(result)
    

    def solveE_V_at_ks(self, ks: np.ndarray) -> np.ndarray:
        """provide energy and derivative for a list of kpoints"""
        hijk, sijk, hijk_derv, sijk_derv \
            = self.Hijk_SijK_and_derivatives(ks, require_derivative=True)

        if len(ks) < 1000:
            energy = []
            velocity = []
            for h, s, dh, ds in zip(hijk, sijk, hijk_derv, sijk_derv):
                w, cT = scipy.linalg.eig(h, s) # it has shape
                cs = cT.T

                derivative = []
                for epsilon, c in zip(w, cs):
                    dhds = dh - epsilon * ds # (3, nbnd, nbnd)
                    derivative.append(
                        np.einsum("i, kij, j -> k", np.conjugate(c), dhds, c)
                    )

                veloc = np.array(derivative) * 1e-10 * scipy.constants.elementary_charge  / scipy.constants.hbar
                sort_index = np.argsort(w.real)
                energy.append(w.real[sort_index])
                velocity.append(veloc.real[sort_index])
        else:
            # parallel version
            energy = []
            velocity = []
            for h, s, dh, ds in zip(hijk, sijk, hijk_derv, sijk_derv):
                w, cT = scipy.linalg.eig(h, s) # it has shape
                cs = cT.T

                derivative = []
                for epsilon, c in zip(w, cs):
                    dhds = dh - epsilon * ds # (3, nbnd, nbnd)
                    derivative.append(
                        np.einsum("i, kij, j -> k", np.conjugate(c), dhds, c)
                    )

                veloc = np.array(derivative) * 1e-10 * scipy.constants.elementary_charge  / scipy.constants.hbar
                sort_index = np.argsort(w.real)
                energy.append(w.real[sort_index])
                velocity.append(veloc.real[sort_index])
        return np.array(energy), np.array(velocity)
