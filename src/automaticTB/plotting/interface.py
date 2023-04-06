import typing

import numpy as np

from automaticTB import tools
from automaticTB.solve import SALCs
from .molecular_wavefunction import MolecularWavefunction, Wavefunction, WavefunctionsOnSite


def get_molecular_wavefunction_from_linear_combination(lc: SALCs.LinearCombination) \
-> MolecularWavefunction:
    """
    convert a linear combination into a molecular wavefunction
    """
    cell = tools.get_cell_from_origin_centered_positions([site.pos for site in lc.sites])
    shift = np.dot(cell.T, np.array([0.5,0.5,0.5]))

    atomic_wavefunctions = []
    for site, orbital_slice in zip(lc.sites, lc.orbital_list.atomic_slice_dict):
        shs = lc.orbital_list.sh_list[orbital_slice]
        coeffs = lc.coefficients[orbital_slice]
        wfs: typing.List[Wavefunction] = []
        for sh, coeff in zip(shs, coeffs):
            wfs.append(
                Wavefunction(sh.n, sh.l, sh.m, coeff)
            )
        atomic_wavefunctions.append(
            WavefunctionsOnSite(
                site.pos + shift, site.atomic_number, wfs
            )
        )

    return MolecularWavefunction(
        cell, 
        atomic_wavefunctions
    )

