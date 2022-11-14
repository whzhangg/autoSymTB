import numpy as np
import typing
from scipy.special import sph_harm

__all__ = ["wavefunction", "xyz_to_r_theta_phi", "r_theta_phi_to_xyz"]

def sh_functions(l, m, theta, phi) -> np.ndarray:
    """
    returns the real part of the spherical harmonic, mainly taken from here 
    https://scipython.com/blog/visualizing-the-real-forms-of-the-spherical-harmonics/ 
    """
    # theta in 0-2pi; phi in 0-pi
    Y = sph_harm(abs(m), l, theta, phi)
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real
    # real spherical harmonic
    return Y.real


def radial_function(n: int, r: typing.Union[float, np.ndarray]) -> np.ndarray:
    """
    This is simply the radial basis function used in Slater type orbitals, 
    given by Albright's book 'Orbital Interactions in Chemistry' Chapter 1. 
    the radial function here is not normalized, but should be ok for plotting
    """
    if n == 0:
        return np.ones_like(r)
    else:
        #r *= 2
        return r**(n-1) * np.exp(-1.0 * r)


def wavefunction(n, l, m, r, theta, phi) -> np.ndarray:
    """
    This should not be used because the coefficients apple on spherical harmonics only, not on 
    radial wavefunction
    """
    # theta in 0-2pi; phi in 0-pi
    return radial_function(n, r) * sh_functions(l, m, theta, phi)


def xyz_to_r_theta_phi(xyz: np.ndarray) -> np.ndarray:
    """cartesian coordinate to polar coordinate"""
    r_theta_phi = np.zeros_like(xyz)
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]
    r_theta_phi[...,0] = np.linalg.norm(xyz, axis=-1) # r
    r_theta_phi[...,1] = np.arctan2(y, x) # theta
    xy = np.hypot(x, y)
    r_theta_phi[...,2] = np.arctan2(xy, z) # phi
    return r_theta_phi


def r_theta_phi_to_xyz(r_theta_phi: np.ndarray) -> np.ndarray:
    """ploar coordinate to cartesian coordinate"""
    xyz = np.zeros_like(r_theta_phi)
    r = r_theta_phi[...,0]
    theta = r_theta_phi[...,1]
    phi = r_theta_phi[...,2]
    xyz[...,0] = r * np.sin(phi) * np.cos(theta)
    xyz[...,1] = r * np.sin(phi) * np.sin(theta)
    xyz[...,2] = r * np.cos(phi) 
    return xyz

