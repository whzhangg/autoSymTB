"""
calculate kpoint energy and velocity using the irreducible part of the 
Brillouin zone
"""

import typing

import numpy as np
import spglib

from automaticTB.properties import tightbinding
from automaticTB import parameters
from automaticTB.properties import reciprocal

def _get_k_index(xk: np.ndarray, kgrid: np.ndarray):
    i = (xk[0]*kgrid[0] + 5 * kgrid[0]) % kgrid[0] 
    j = (xk[1]*kgrid[1] + 5 * kgrid[1]) % kgrid[1] 
    k = (xk[2]*kgrid[2] + 5 * kgrid[2]) % kgrid[2] 
    return int( k + j * kgrid[2] + i * kgrid[1] * kgrid[2] )


def calculate_e_v_using_ibz(
    tb: tightbinding.TightBindingModel, 
    kpoints: np.ndarray, 
    nks: typing.Tuple[int,int,int],
    energy_only: bool = False
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """calculates the kpoint energy using irreducible grid"""
    cell = (tb.cell, tb.positions, tb.types)
    sym_data = spglib.get_symmetry_dataset(cell, symprec=parameters.spgtol)

    # point group operation
    unique = []
    for r in sym_data["rotations"]:
        found = False

        for r_u in unique:
            if np.allclose(r, r_u, atol=parameters.stol): 
                found = True
                break
        if found: continue
        unique.append(r)
    
    c = tb.cell
    rc = reciprocal.find_RCL(c)

    nrot = len(unique)
    rot_k_frac = np.zeros((nrot, 3, 3))
    rot_k_cart = np.zeros((nrot, 3, 3))
    for ir, r in enumerate(unique):
        opc = c.T @ r @ np.linalg.inv(c.T)
        rot_frac_k = np.linalg.inv(rc.T) @ opc @ rc.T
        rot_k_cart[ir] = opc
        rot_k_frac[ir] = rot_frac_k

    nk = len(kpoints)
    found_k = set()
    symmetry_eq = np.zeros(nk, dtype=np.intp)
    rotation_index = np.zeros(nk, dtype=np.intp)
    
    for i, k in enumerate(kpoints):
        assert i == _get_k_index(k, nks)
        if i in found_k: continue
        rotated_k = rot_k_frac @ k
        rotated_index = [_get_k_index(rk, nks) for rk in rotated_k]
        for ir, rotated_i in enumerate(rotated_index):
            if rotated_i in found_k: continue
            rotation_index[rotated_i] = ir
            symmetry_eq[rotated_i] = i
            found_k.add(rotated_i)

    unique_kid = np.unique(symmetry_eq)
    unique_kpoints = kpoints[unique_kid]
    rev = {u: iu for iu, u in enumerate(unique_kid)}

    if energy_only:
        e, _ = tb.solveE_at_ks(unique_kpoints)
        _, nbnd = e.shape
        full_e = np.zeros((nk, nbnd))
        for i, (eqk, ri) in enumerate(zip(symmetry_eq, rotation_index)):
            which_k = rev[eqk]
            full_e[i, :] = e[which_k,:]
        full_v = None
    else:
        e, v, _ = tb.solveE_V_at_ks(unique_kpoints)
        _, nbnd = e.shape
        full_e = np.zeros((nk, nbnd))
        full_v = np.zeros((nk, nbnd, 3))

        for i, (eqk, ri) in enumerate(zip(symmetry_eq, rotation_index)):
            which_k = rev[eqk]
            rot = rot_k_cart[ri]
            full_e[i, :] = e[which_k,:]
            full_v[i, :, :] = np.einsum(
                "ij,nj -> ni", rot, v[which_k, :, :]
            )

    return full_e, full_v

