import typing
import numpy as np
from .hij import Pindex_lm, HijR, SijR
from ._tightbinding_base import TightBindingBase

"""
this module defines tight-binding model, the only thing we need is the HijR and SijR, 
which contain the energy and overlap of atomic orbitals.
"""

__all__ = ["TightBindingModel"]


class TightBindingModel(TightBindingBase):
    def __init__(self, 
        cell: np.ndarray,
        positions: np.ndarray,
        types: typing.List[int],
        HijR_list: typing.List[HijR],
        SijR_list: typing.Optional[typing.List[SijR]] = None
    ) -> None:
        self._cell = cell
        self._positions = positions
        self._types = types
        self._HijRs = HijR_list

        """
        if SijR_list is None:
            # create SijR list
            self._SijRs = []
            for hijr in self._HijRs:
                if hijr.left == hijr.right:
                    self._SijRs.append(
                        SijR(hijr.left, hijr.right, 1.0)
                    )
        else:
        """
        self._SijRs = SijR_list

        self._basis: typing.List[Pindex_lm] = []
        for hijR in self._HijRs:
            if hijR.left not in self._basis:
                self._basis.append(hijR.left)
                
        self._index_ref: typing.Dict[tuple, int] = {
            (basis.pindex, basis.n, basis.l, basis.m):i for i, basis in enumerate(self._basis)
        }

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

    def Hijk_and_derivatives(self, k: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        it returns the H matrix and its derivative with respect to k
        H matrix is in unit eV and its derivative has unit eV * angstrom
        """
        k = np.array(k)
        nbasis = len(self._basis)
        ham = np.zeros((nbasis,nbasis), dtype=complex)
        ham_derv = np.zeros((3, nbasis, nbasis), dtype=complex)
        for HijR in self._HijRs:
            index_i = self._index_ref[
                (HijR.left.pindex, HijR.left.n, HijR.left.l, HijR.left.m)
            ]
            index_j = self._index_ref[
                (HijR.right.pindex, HijR.right.n, HijR.right.l, HijR.right.m)
            ]

            r = HijR.right.translation + \
                self._positions[HijR.right.pindex] - self._positions[HijR.left.pindex]
            kR = np.dot(k, r)
            r_cart = np.dot(self._cell.T, r) # in A
            tmp = HijR.value * np.exp(2j * np.pi * kR)
            ham[index_i,index_j] += tmp
            ham_derv[0, index_i, index_j] += tmp * 1j * r_cart[0]
            ham_derv[1, index_i, index_j] += tmp * 1j * r_cart[1]
            ham_derv[2, index_i, index_j] += tmp * 1j * r_cart[2] 

        return ham, ham_derv


    def Sijk_and_derivatives(self, k: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        it returns the overlap matrix S and its derivative with respect to k.
        S matrix is unitless while the derivative has unit angstrom
        """
        k = np.array(k)
        nbasis = len(self._basis)

        if self._SijRs is None:
            return np.eye(nbasis, dtype=complex), np.zeros(nbasis, dtype=complex)

        ham = np.zeros((nbasis,nbasis), dtype=complex)
        ham_derv = np.zeros((3, nbasis, nbasis), dtype=complex)
        for HijR in self._SijRs:
            index_i = self._index_ref[
                (HijR.left.pindex, HijR.left.n, HijR.left.l, HijR.left.m)
            ]
            index_j = self._index_ref[
                (HijR.right.pindex, HijR.right.n, HijR.right.l, HijR.right.m)
            ]

            r = HijR.right.translation + \
                self._positions[HijR.right.pindex] - self._positions[HijR.left.pindex]
            kR = np.dot(k, r)
            r_cart = np.dot(self._cell.T, r) # in A
            tmp = HijR.value * np.exp(2j * np.pi * kR)
            ham[index_i,index_j] += tmp
            ham_derv[0, index_i, index_j] += tmp * 1j * r_cart[0]
            ham_derv[1, index_i, index_j] += tmp * 1j * r_cart[1]
            ham_derv[2, index_i, index_j] += tmp * 1j * r_cart[2] 

        return ham, ham_derv


    def Hijk(self, k: typing.Tuple[float]):
        k = np.array(k)
        nbasis = len(self._basis)
        ham = np.zeros((nbasis,nbasis), dtype=complex)
        for HijR in self._HijRs:
            assert np.allclose(HijR.left.translation, np.zeros(3))
            r = HijR.right.translation
            index_i = self._index_ref[
                (HijR.left.pindex, HijR.left.n, HijR.left.l, HijR.left.m)
            ]
            index_j = self._index_ref[
                (HijR.right.pindex, HijR.right.n, HijR.right.l, HijR.right.m)
            ]
            r = HijR.right.translation + \
                self._positions[HijR.right.pindex] - self._positions[HijR.left.pindex]
            ham[index_i,index_j] += HijR.value * np.exp(2j * np.pi * np.dot(k, r))
        return ham


    def Sijk(self, k: typing.Tuple[float]):
        k = np.array(k)
        nbasis = len(self._basis)

        if self._SijRs is None:
            return np.eye(nbasis, dtype=complex)
        
        ham = np.zeros((nbasis,nbasis), dtype=complex)
        for HijR in self._SijRs:
            assert np.allclose(HijR.left.translation, np.zeros(3))
            r = HijR.right.translation
            index_i = self._index_ref[
                (HijR.left.pindex, HijR.left.n, HijR.left.l, HijR.left.m)
            ]
            index_j = self._index_ref[
                (HijR.right.pindex, HijR.right.n, HijR.right.l, HijR.right.m)
            ]
            kR = np.dot(k, r + self._positions[HijR.right.pindex] - self._positions[HijR.left.pindex])
            ham[index_i,index_j] += HijR.value * np.exp(2j * np.pi * kR)
        return ham