import numpy as np

from .cython_functions import (
    c_calc_dos_contribution, c_calc_dos_nsum, create_tetras, c_calc_dos_contribution_multie, 
    c_calc_projdos_contribution_multie)


def cython_dos_contribution_multiple(
        bandenergy: np.ndarray, tetras: np.ndarray, e: np.ndarray) -> np.ndarray:
    return c_calc_dos_contribution_multie(bandenergy, tetras, e)


def cython_projdos_contribution_multiple(
    bandenergy: np.ndarray, bandc: np.ndarray, 
    tetras: np.ndarray, e: np.ndarray
) -> np.ndarray:
    return c_calc_projdos_contribution_multie(bandenergy, bandc, tetras, e, bandc.shape[1])


def cython_create_tetras(
    kpoints: np.ndarray, subcell_tetras: np.ndarray, mesh_shift: np.ndarray, nk1, nk2, nk3
) -> np.ndarray:
    return create_tetras(kpoints, subcell_tetras, mesh_shift, nk1, nk2, nk3)


def cython_dos_contribution(bandenergy: np.ndarray, tetras: np.ndarray, e: float) -> float:
    """calculate contribution of DOS at energy e"""
    return c_calc_dos_contribution(bandenergy, tetras, e)


def cython_nsum_contribution(bandenergy: np.ndarray, tetras: np.ndarray, e: float) -> float:
    """calculate contribution of DOS at energy e"""
    return c_calc_dos_nsum(bandenergy, tetras, e)

