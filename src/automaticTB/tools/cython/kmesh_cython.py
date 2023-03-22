import numpy as np
from .cython_functions import get_rotations

def find_rotation_operation_indices(
        rotations: np.ndarray, mapping: np.ndarray, kpoints: np.ndarray, zero_prec: float):
    """find the rotation index between reducible and irred. k points"""
    
    return get_rotations(
        rotations.astype(np.double), mapping.astype(np.intp), 
        kpoints.astype(np.double), np.double(zero_prec))