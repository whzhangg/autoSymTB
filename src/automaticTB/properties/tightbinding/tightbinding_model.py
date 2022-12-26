import numpy as np, typing
from scipy import constants
from .hij import Pindex_lm, HijR, SijR
from ..kpoints import UnitCell
import numpy as np
import typing

"""
this module defines tight-binding model, the only thing we need is the HijR and SijR, 
which contain the energy and overlap of atomic orbitals.
"""

__all__ = ["TightBindingModel"]


def solve_secular(H: np.ndarray, S: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    #
    import scipy.linalg as scipylinalg
    #
    if H.shape != S.shape: raise "Shape must the same"
    S_minus_half = scipylinalg.fractional_matrix_power(S, -0.5)
    
    tilde_H = S_minus_half @ H @ S_minus_half
    w, v = np.linalg.eig(tilde_H)
    c = np.einsum("ij, jk -> ik", S_minus_half, v)

    return w, c


def solve_secular_sorted(H: np.ndarray, S: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    w, c = solve_secular(H, S)
    sort_index = np.argsort(w.real)
    return w[sort_index], c[:,sort_index]


class TightBindingModel():
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

        self._SijRs = SijR_list

        self._basis: typing.List[Pindex_lm] = []
        for hijR in self._HijRs:
            if hijR.left not in self._basis:
                self._basis.append(hijR.left)
                
        self._index_ref: typing.Dict[tuple, int] = {
            (basis.pindex, basis.n, basis.l, basis.m):i for i, basis in enumerate(self._basis)
        }

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

    
    def solveE_at_k(self, k: np.ndarray) -> np.ndarray:
        # solve at a single kpoints
        h = self.Hijk(k)
        #w, v = np.linalg.eig(h)
        s = self.Sijk(k)
        w, c = solve_secular_sorted(h, s)
        return w.real, c

    def calculate_e_velocity_lee(
        self, k: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        obtain the band velocity using the equation derived in Lee et al. 2018, without the need
        for finite difference.
        """
        h, dhdk = self.Hijk_and_derivatives(k)
        s, dsdk = self.Sijk_and_derivatives(k)
        w, cT = solve_secular(h, s) # it has shape
        cs = cT.T

        derivative = []
        for epsilon, c in zip(w, cs):
            dhds = dhdk - epsilon * dsdk # (3, nbnd, nbnd)
            derivative.append(
                np.einsum("i, kij, j -> k", np.conjugate(c), dhds, c)
            )

        veloc = np.array(derivative) * 1e-10 * constants.elementary_charge  / constants.hbar
        sort_index = np.argsort(w.real)
        
        return w.real[sort_index], veloc.real[sort_index, :]

    def calculate_e_velocity_finite_diff(
        self, k: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        r"""
        return analytic velocity in SI
        v_n(k) = \frac{1}{\hbar} \nabla_k e_n(k)
        """
        # a set of K
        delta_k = 0.01
        difference_array = np.array([
                                [ 1.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0],
                                [ 0.0, 1.0, 0.0],
                                [ 0.0,-1.0, 0.0],
                                [ 0.0, 0.0, 1.0],
                                [ 0.0, 0.0,-1.0]
                            ]) * delta_k
        
        rcell = self.reciprocal_cell
        fractional_difference = \
            np.einsum("ij, kj -> ki", np.linalg.inv(rcell.T), difference_array) + k

        energies = self.solveE_at_ks(fractional_difference) # nk, nbnd
        _, nbnd = energies.shape
        derivative = np.zeros((nbnd, 3))

        # energy in unit eV and dk in unit of 1/A
        for ibnd in range(nbnd):
            derivative[ibnd, 0] = (energies[0, ibnd] - energies[1, ibnd]) / (delta_k * 2)
            derivative[ibnd, 1] = (energies[2, ibnd] - energies[3, ibnd]) / (delta_k * 2)
            derivative[ibnd, 2] = (energies[4, ibnd] - energies[5, ibnd]) / (delta_k * 2)

        derivative *= 1e-10 * constants.elementary_charge # SI

        energy, _ = self.solveE_at_k(k)
        
        return energy, derivative / constants.hbar


    def solveE_c2_at_ks(self, ks: np.ndarray) -> np.ndarray:
        ek = []
        coeff = []
        for k in ks:
            w, complex_c =self.solveE_at_k(k)
            ek.append(w)
            c_T = (np.abs(complex_c)**2).T # [ibnd, iorb]
            coeff.append(c_T[np.newaxis,...])
        ek = np.vstack(ek)
        coeff = np.vstack(coeff)
        assert coeff.shape[0] == ek.shape[0]
        return ek, coeff

    def solveE_at_ks(self, ks: np.ndarray) -> np.ndarray:
        return np.vstack(
            [ self.solveE_at_k(k)[0] for k in ks ]
        )

    def solveE_V_at_ks(self, ks: np.ndarray) -> np.ndarray:
        energy = []
        velocity = []
        for k in ks:
            e, v = self.calculate_e_velocity_lee(k)
            energy.append(e)
            velocity.append(v)

        return np.array(energy), np.array(velocity)