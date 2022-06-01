import numpy as np
import typing
from scipy.special import sph_harm

def sh_functions(l, m, theta, phi) -> np.ndarray:
    # theta in 0-2pi; phi in 0-pi
    Y = sph_harm(abs(m), l, theta, phi)
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real
    # real spherical harmonic
    return Y.real


def radial_function(n: int, r: typing.Union[float, np.ndarray]) -> np.ndarray:
    assert n > 0
    return r**(n-1) * np.exp(-1.0 * r)


def wavefunction(n, l, m, r, theta, phi) -> np.ndarray:
    # theta in 0-2pi; phi in 0-pi
    return radial_function(n, r) * sh_functions(l, m, theta, phi)
