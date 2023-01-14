import numpy as np, typing
from scipy import constants, linalg as scipylinalg
from .hij import Pindex_lm, HijR, SijR
from ..kpoints import UnitCell
from collections import namedtuple

__all__ = ["TightBindingModelOptimized"]

HSijR = namedtuple('HSijR', "i j r H S")

class TightBindingModelOptimized():
    def __init__(self, 
        cell: np.ndarray,
        positions: np.ndarray,
        types: typing.List[int],
        HijR_list: typing.List[HijR],
        SijR_list: typing.Optional[typing.List[SijR]] = None
    ) -> None:
        """
        we assert HijR and SijR are the same list with the same pairs and we put them in tuples with index
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
        return UnitCell.find_RCL(self.cell)


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


    def Hijk_Sijk_at_k(
        self, k: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ks = np.array([k])
        hijk, sijk, hijk_derv, sijk_derv = \
            self.Hijk_SijK_and_derivatives(ks, True)
        
        return hijk[0], sijk[0], hijk_derv[0], sijk_derv[0]


    def Hijk_SijK_and_derivatives(
        self, ks: np.ndarray, require_derivative: bool = True
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


    def solveE_at_k(self, k: np.ndarray) -> np.ndarray:
        # it now wraps over solveE_at_ks version
        ks = np.array([k])
        es = self.solveE_at_ks(ks)
        return es[0]


    def solveE_at_ks(self, ks: np.ndarray) -> np.ndarray:
        hijk, sijk, _, _ = self.Hijk_SijK_and_derivatives(ks, require_derivative=False)
        result = []
        for h, s in zip(hijk, sijk):
            #h = (h + np.conjugate(h.T)) / 2.0
            #s = (s + np.conjugate(s.T)) / 2.0  # should symmetrize hamiltonian
            w, c = scipylinalg.eig(h, s)
            result.append(np.sort(w.real))

        return np.array(result)
            

    def solveE_V_at_ks(self, ks: np.ndarray) -> np.ndarray:
        energy = []
        velocity = []
        hijk, sijk, hijk_derv, sijk_derv \
            = self.Hijk_SijK_and_derivatives(ks, require_derivative=True)

        for h, s, dh, ds in zip(hijk, sijk, hijk_derv, sijk_derv):
            w, cT = scipylinalg.eig(h, s) # it has shape
            cs = cT.T

            derivative = []
            for epsilon, c in zip(w, cs):
                dhds = dh - epsilon * ds # (3, nbnd, nbnd)
                derivative.append(
                    np.einsum("i, kij, j -> k", np.conjugate(c), dhds, c)
                )

            veloc = np.array(derivative) * 1e-10 * constants.elementary_charge  / constants.hbar
            sort_index = np.argsort(w.real)
            energy.append(w.real[sort_index])
            velocity.append(veloc.real[sort_index])

        return np.array(energy), np.array(velocity)

