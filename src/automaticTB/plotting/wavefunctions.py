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
    r *= 2
    return r**(n-1) * np.exp(-1.0 * r)


def wavefunction(n, l, m, r, theta, phi) -> np.ndarray:
    # theta in 0-2pi; phi in 0-pi
    return radial_function(n, r) * sh_functions(l, m, theta, phi)


def xyz_to_r_theta_phi(xyz: np.ndarray) -> np.ndarray:
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
    xyz = np.zeros_like(r_theta_phi)
    r = r_theta_phi[...,0]
    theta = r_theta_phi[...,1]
    phi = r_theta_phi[...,2]
    xyz[...,0] = r * np.sin(phi) * np.cos(theta)
    xyz[...,1] = r * np.sin(phi) * np.sin(theta)
    xyz[...,2] = r * np.cos(phi) 

    return xyz


def plot_radial_functions(ns: typing.List[int]):
    filename = "radial_part.pdf"
    import matplotlib.pyplot as plt
    from mplconfig import get_acsparameter

    x = np.linspace(0.0, 10.0, 100)
    with plt.rc_context(get_acsparameter(width = "single", n = 1, color = "line")):
        fig = plt.figure()
        axes = fig.subplots()
                
        axes.set_xlabel("r")
        axes.set_ylabel("Radial Functions")
        
        axes.set_xlim(0,10.0)
        for n in ns:
            axes.plot(x, radial_function(n, x), label = f"n = {n}")
        axes.legend()
        fig.savefig(filename)

