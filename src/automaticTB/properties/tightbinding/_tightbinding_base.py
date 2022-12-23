import typing, abc
import numpy as np
from ..kpoints import UnitCell
from scipy import constants

from .hij import Pindex_lm
from ._secular_solver import solve_secular_sorted, solve_secular

class TightBindingBase(abc.ABC):

    @property
    @abc.abstractmethod
    def cell(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def positions(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def types(self) -> typing.List[int]:
        pass

    @property
    @abc.abstractmethod
    def basis(self) -> typing.List[Pindex_lm]:
        pass

    @property
    def reciprocal_cell(self) -> np.ndarray:
        return UnitCell.find_RCL(self.cell)

    @abc.abstractmethod 
    def Hijk(self, frac_k: np.ndarray) -> np.ndarray:
        pass 

    @abc.abstractmethod
    def Sijk(self, frac_k: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def Hijk_and_derivatives(self, frac_k: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def Sijk_and_derivatives(self, frac_k: np.ndarray) -> np.ndarray:
        pass

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