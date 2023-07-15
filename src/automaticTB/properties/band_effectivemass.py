import typing
import dataclasses
import warnings

import numpy as np
import scipy
from scipy import constants
import lmfit

from automaticTB.properties import tightbinding
from automaticTB.properties import reciprocal


def find_band_minimum(
    tb: tightbinding.TightBindingModel, 
    ibnd: int, 
    k_approx:  np.ndarray, 
    is_valence: bool,
    delta_k: float = 0.15,
    verbose: bool = False,
) -> typing.Tuple[np.ndarray, float]:
    """find the band minimum from the calculated grid
    
    Parameters
    ----------
    is_valence: bool
        if True, find the band maximum point
    delta_k: float
        give upper and lower range for k points to search: (k_i - dk, k_i + dk), i=xyz
    
    Output
    ------
    (kfound, e): typing.Tuple[np.ndarray, float]
        k position and corresponding energy 
    """
    params = lmfit.Parameters()
    params.add("kx", k_approx[0], min=k_approx[0] - delta_k, max=k_approx[0] + delta_k)
    params.add("ky", k_approx[1], min=k_approx[1] - delta_k, max=k_approx[1] + delta_k)
    params.add("kz", k_approx[2], min=k_approx[2] - delta_k, max=k_approx[2] + delta_k)

    e0, _ = tb.solveE_at_ks(np.array([k_approx]))
    if verbose:
        print("We start from k = {:.3f}, {:.3f}, {:.3f} ...".format(*k_approx))
        print("              e = {:.3f}".format(e0[0][ibnd]))
    out = lmfit.minimize(
        _find_minimum_energy, params, args = (tb, ibnd, is_valence), 
        method = 'bfgs'
    )

    k_final = np.array([v for v in out.params.valuesdict().values()])
    e2, _ = tb.solveE_at_ks(np.array([k_final]))
    e_final = e2[0][ibnd]

    if verbose:
        print("\nOptimization result:")
        for k, v in out.params.valuesdict().items():
            print(f"  {k:s} = {params[k].value:8.4f} ->{v:8.4f}", end = "")
            if np.isclose(np.abs(params[k].value - v), delta_k):
                print("  < value close to boundary !")
            else:
                print("")
        print(f"band edge energy = {e_final:.3f}")
    
    return k_final, e_final


def calculate_inv_em_weighted(
    tb: tightbinding.TightBindingModel, 
    kpos: np.ndarray, 
    ibnd: int, 
    is_valence: bool,
    temperature: float = 1000, 
    mesh_delta_frac: float = 0.05,
) -> np.ndarray:
    """calculate inverse effective mass metrix
    
    We assume that kpos is the band minimum
    """
    e0, _ = tb.solveE_at_ks([kpos])
    e0 = e0[0, ibnd]
    dk_frac, dk_cart = create_small_delta_kmesh(tb.cell, mesh_delta_frac)
    
    kpoints = kpos + dk_frac

    energies, _ = tb.solveE_at_ks(kpoints)
    de = energies[:, ibnd] - e0

    if is_valence and not np.isclose(e0, np.max(energies[:,ibnd])):
        print("Warning: k=[{:7.3f}{:7.3f}{:7.3f}]".format(*kpos), end="") 
        print("@ibnd={:d}, e={:.3f} is not a local vb maximum".format(ibnd, e0))
    if not is_valence and not np.isclose(e0, np.min(energies[:,ibnd])):
        print("Warning: k=[{:7.3f}{:7.3f}{:7.3f}]".format(*kpos), end="") 
        print("@ibnd={:d}, e={:.3f} is not a local cb miniimum".format(ibnd, e0))

    if is_valence:
        weights = np.array([
            np.exp(
                -e * scipy.constants.electron_volt /(temperature * scipy.constants.Boltzmann)
            ) for e in de])
    else:
        weights = np.array([
            np.exp(
                e * scipy.constants.electron_volt /(temperature * scipy.constants.Boltzmann)
            ) for e in de])
    
    weights /= np.sum(weights)

    params = lmfit.Parameters()
    params.add("dxx", 0)
    params.add("dxy", 0)
    params.add("dxz", 0)
    params.add("dyy", 0)
    params.add("dyz", 0)
    params.add("dzz", 0)

    out = lmfit.minimize(
        _fit_loss, params, method = 'lbfgsb', args = (dk_cart, energies, weights)
    )

    dxx, dxy, dxz, dyy, dyz, dzz = out.params.valuesdict().values()
    tensor = np.array([
                    [dxx, dxy, dxz], 
                    [dxy, dyy, dyz], 
                    [dxz, dyz, dzz]
                ])

    inv_mass = (tensor * scipy.constants.elementary_charge * (1e-10)**2 
                / scipy.constants.hbar**2 * scipy.constants.electron_mass)

    return inv_mass


def _find_minimum_energy(
    params: lmfit.Parameters, 
    tb: tightbinding.TightBindingModel, ibnd: int, is_valence: bool
) -> float:
    values = params.valuesdict()
    x, y, z = values["kx"], values['ky'], values['kz']
    e, _ = tb.solveE_at_ks([(x,y,z)])
    if is_valence:
        return -1 * e[0][ibnd]
    else:
        return e[0][ibnd]


def _fit_loss(
    params: lmfit.Parameters,
    kcart: np.ndarray,
    e_fit: np.ndarray,
    weights: np.ndarray
):
    values = params.valuesdict()
    dxx = values["dxx"]
    dxy = values['dxy']
    dxz = values['dxz']
    dyy = values["dyy"]
    dyz = values['dyz']
    dzz = values['dzz']

    kx = kcart[:,0]
    ky = kcart[:,1]
    kz = kcart[:,2]

    predicted =  0.5 * dxx * kx * kx + dxy * kx * ky + dxz * kx * kz \
                               + 0.5 * dyy * ky * ky + dyz * ky * kz \
                                               + 0.5 * dzz * kz * kz
    return np.sum((predicted - e_fit)**2 * weights)


def create_small_delta_kmesh(
    cell: np.ndarray, 
    delta_frac: float = 0.05,
    ngrid: int = 5
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """return both kmesh_frac and kmesh_cart"""
    rlv = reciprocal.find_RCL(cell)
    vr = np.average(np.linalg.norm(rlv * delta_frac, axis=1))
    kgrid_cart = []
    for x in np.linspace(-1 * vr, vr, ngrid):
        for y in np.linspace(-1 * vr, vr, ngrid):
            for z in np.linspace(-1 * vr, vr, ngrid):
                kgrid_cart.append([x, y, z])
    kgrid_cart = np.array(kgrid_cart)
    inv_rlv_T = np.linalg.inv(rlv.T)
    kgrid_frac = np.einsum("ij,kj -> ki", inv_rlv_T, kgrid_cart)
    return kgrid_frac, kgrid_cart

