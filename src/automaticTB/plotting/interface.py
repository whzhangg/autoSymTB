import typing, dataclasses
import numpy as np
from ..interaction.SALCs import LinearCombination
from ..parameters import zero_tolerance
from ..tools import get_cell_from_origin_centered_positions
from .molecular_wavefunction import MolecularWavefunction, Wavefunction, WavefunctionsOnSite


def get_cell_from_origin_centered_positions(positions: typing.List[np.ndarray]) \
-> np.ndarray:
    """
    This function accept a list of cartesian position and create a cell that enclose all 
    the points, the cell size is determined by the maximum distance of the cartesian position 
    with respect to the origin (0, 0, 0)
    """
    rmax = -1.0
    for pos in positions:
        rmax = max(rmax, np.linalg.norm(pos))
    cell = 4.0 * rmax * np.eye(3)
    return cell


def get_molecular_wavefunction_from_linear_combination(lc: LinearCombination) \
-> MolecularWavefunction:
    """
    convert a linear combination into a molecular wavefunction
    """
    cell = get_cell_from_origin_centered_positions([site.pos for site in lc.sites])
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


def get_crystal_wavefunction_from_linear_combination(
    lc: LinearCombination, 
    cell: np.ndarray, positions: np.ndarray, types: np.ndarray
) -> MolecularWavefunction:
    """
    this function convert a linear combination into a wavefunction in the crystal, there is a 
    slight complication because of the periodicity of the crystal which become problematic with 
    the wavefunction is close to the cell boundary
    """
    raise NotImplementedError