import numpy as np, typing
from ..tightbinding import TightBindingModel
from ..kpoints import UnitCell
from scipy.optimize import curve_fit
from scipy import constants


def fit_function(dks, e0, k0x, k0y, k0z, dxx, dxy, dxz, dyy, dyz, dzz):
    # fit effective mass tensor with linear part
    kx = dks[:,0] - k0x
    ky = dks[:,1] - k0y
    kz = dks[:,2] - k0z

    return  e0 + 0.5 * dxx * kx * kx + dxy * kx * ky + dxz * kx * kz \
                            + 0.5 * dyy * ky * ky + dyz * ky * kz \
                                            + 0.5 * dzz * kz * kz


class BandEffectiveMass:
    ngrid: int = 5
    dkmax_frac: float = 0.03

    def __init__(self, tb: TightBindingModel) -> None:
        self.tb = tb
        self.rlv = UnitCell.find_RCL(tb.cell)
        self.dkg_cart = self._calc_dk_cart()


    def _calc_dk_cart(self) -> np.ndarray:
        vr = np.linalg.norm(np.dot(self.rlv.T, np.ones(3) * self.dkmax_frac)) / np.sqrt(3)
        dkg_cart = []
        for x in np.linspace(-1 * vr, vr, self.ngrid):
            for y in np.linspace(-1 * vr, vr, self.ngrid):
                for z in np.linspace(-1 * vr, vr, self.ngrid):
                    dkg_cart.append([x, y, z])
        return np.array(dkg_cart)

    def calc_effective_mass(self, k_in: np.ndarray, ibnd: int) -> dict:
        """return a dictionary with fitting result"""
        k_in_cart = np.dot(self.rlv.T, k_in)
        kg_cart = self.dkg_cart + k_in_cart
        inv_rlv_T = np.linalg.inv(self.rlv.T)
        kg_fra = np.einsum("ij,kj -> ki", inv_rlv_T, kg_cart)

        energies = self.tb.solveE_at_ks(kg_fra)[:, ibnd]

        result = self.do_fitting(kg_cart, energies)
        return {
            "k0_frac" : np.dot(inv_rlv_T, result["k0_cart"]),
            "e0" : result["e0"],
            "inv_mass_tensor" : result["inv_mass_tensor"],
            "eigen_mass" : result["eigen_mass"]
        }

    @staticmethod
    def do_fitting(kg_cart: np.ndarray, energies: np.ndarray) -> dict:
        popt, _ = curve_fit(fit_function, kg_cart, energies)
        e0, k0x, k0y, k0z, dxx, dxy, dxz, dyy, dyz, dzz = popt

        inv_m = np.array([
                [dxx, dxy, dxz], 
                [dxy, dyy, dyz], 
                [dxz, dyz, dzz]
            ])  * constants.elementary_charge * (1e-10)**2 / constants.hbar**2 \
                * constants.electron_mass

        kmin = np.array([k0x, k0y, k0z])

        eig_mass = 1.0 / np.linalg.eig(inv_m)[0].real

        return {
            "k0_cart" : kmin,
            "e0" : e0,
            "inv_mass_tensor" : inv_m,
            "eigen_mass" : eig_mass
        }