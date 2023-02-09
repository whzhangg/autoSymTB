import typing
import dataclasses

import numpy as np
import scipy

from automaticTB.properties import tightbinding
from automaticTB.properties import kpoints


def fit_function(dks, e0, k0x, k0y, k0z, dxx, dxy, dxz, dyy, dyz, dzz):
    # fit effective near maximum point
    kx = dks[:,0] - k0x
    ky = dks[:,1] - k0y
    kz = dks[:,2] - k0z

    return  e0 + 0.5 * dxx * kx * kx + dxy * kx * ky + dxz * kx * kz \
                               + 0.5 * dyy * ky * ky + dyz * ky * kz \
                                               + 0.5 * dzz * kz * kz


def fit_function_taylor(dks, dxx, dxy, dxz, dyy, dyz, dzz):
    # fit taylor expansion around k
    kx = dks[:,0]
    ky = dks[:,1]
    kz = dks[:,2]

    return  0.5 * dxx * kx * kx + dxy * kx * ky + dxz * kx * kz \
                          + 0.5 * dyy * ky * ky + dyz * ky * kz \
                                          + 0.5 * dzz * kz * kz


@dataclasses.dataclass
class TaylorBandFitting:
    """provide a taylor expansion for a band at k"""
    energy: float
    first_order_vector: np.ndarray
    second_order_tensor: np.ndarray

    @property
    def inv_mass(self) -> np.ndarray:
        return (self.second_order_tensor
                * scipy.constants.elementary_charge * (1e-10)**2 / scipy.constants.hbar**2
                * scipy.constants.electron_mass)


@dataclasses.dataclass
class ParabolicBandFitting:
    """assume parabolic band"""
    energy: float
    kmin: np.ndarray
    second_order_tensor: np.ndarray

    @property
    def inv_mass(self) -> np.ndarray:
        return (self.second_order_tensor
                * scipy.constants.elementary_charge * (1e-10)**2 / scipy.constants.hbar**2
                * scipy.constants.electron_mass)


class BandEffectiveMass:
    """calculate effective mass at k point
    
    It will create a small grid nk = ngrid^3 and fit some function. 
    The kpoints will span approximately 5% * 5% * 5% of the BZ
    """
    ngrid: int = 5
    dkmax_frac: float = 0.05

    def __init__(self, tb: tightbinding.TightBindingModel) -> None:
        self.tb = tb
        self.rlv = kpoints.find_RCL(tb.cell)

        vr = np.average(np.linalg.norm(self.rlv * self.dkmax_frac, axis=1))
        kgrid_cart = []
        for x in np.linspace(-1 * vr, vr, self.ngrid):
            for y in np.linspace(-1 * vr, vr, self.ngrid):
                for z in np.linspace(-1 * vr, vr, self.ngrid):
                    kgrid_cart.append([x, y, z])
        self.kgrid_cart = np.array(kgrid_cart)


    def _get_energies_for_kgrid(self, kgrid_cart: np.ndarray, ibnd: int) -> np.ndarray:
        inv_rlv_T = np.linalg.inv(self.rlv.T)
        kg_fra = np.einsum("ij,kj -> ki", inv_rlv_T, kgrid_cart)
        energies = self.tb.solveE_at_ks(kg_fra)[:, ibnd]
        return energies


    def fit_taylor_expansion_at_k(
            self, kfrac_in: np.ndarray, ibnd: int) -> TaylorBandFitting:
        """fits the band curvature up to second order
        
        return e0 in eV, band velocity in eV*A and inv_m tensor
        """
        from functools import partial
        kin_cart = np.dot(self.rlv.T, kfrac_in)
        e0, v0 = self.tb.solveE_V_at_k(kfrac_in)

        energies = self._get_energies_for_kgrid(self.kgrid_cart + kin_cart, ibnd)

        de = energies - e0[ibnd] \
            - v0[ibnd][0] * self.kgrid_cart[:,0] \
            - v0[ibnd][1] * self.kgrid_cart[:,1] \
            - v0[ibnd][2] * self.kgrid_cart[:,2]
        
        guess = [0.0] * 6
        popt, _ = scipy.optimize.curve_fit(
            fit_function_taylor, self.kgrid_cart, de, p0=guess, maxfev=5000)

        # curve_fit does not accept partial function: 
        # https://stackoverflow.com/questions/49813481/how-to-pass-parameter-to-fit-function-when-using-scipy-optimize-curve-fit
        dxx, dxy, dxz, dyy, dyz, dzz = popt
        tensor = np.array([
                [dxx, dxy, dxz], 
                [dxy, dyy, dyz], 
                [dxz, dyz, dzz]
            ]) 
        
        return TaylorBandFitting(e0[ibnd], v0[ibnd], tensor)
    

    def fit_parabolic_at_k(
            self, kfrac_in: np.ndarray, ibnd: int) -> ParabolicBandFitting:
        """fits a parabolic curve assuming near band maximum
        
        returns the obtained minimum energy e0 and its position. 
        It also obtain inverse mass tensor
        """
        kin_cart = np.dot(self.rlv.T, kfrac_in)
        e0, _ = self.tb.solveE_V_at_k(kfrac_in)

        energies = self._get_energies_for_kgrid(self.kgrid_cart + kin_cart, ibnd)
        guess = [e0[ibnd], kin_cart[0], kin_cart[1], kin_cart[2], 0, 0, 0, 0, 0, 0]

        popt, _ = scipy.optimize.curve_fit(
            fit_function, self.kgrid_cart + kin_cart, energies, p0=guess, maxfev=5000)
        
        e0, k0x, k0y, k0z, dxx, dxy, dxz, dyy, dyz, dzz = popt

        kmin_frac = np.dot(np.linalg.inv(self.rlv.T), np.array([k0x, k0y, k0z]))
        tensor = np.array([
                [dxx, dxy, dxz], 
                [dxy, dyy, dyz], 
                [dxz, dyz, dzz]
            ])
        
        return ParabolicBandFitting(e0, kmin_frac, tensor)
