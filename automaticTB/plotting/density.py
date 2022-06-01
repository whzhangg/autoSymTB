import numpy as np
#from .plot_LC import sh_functions
import typing
from scipy.special import sph_harm
from ..SALCs import NamedLC, LinearCombination

def sh_functions(l, m, theta, phi):
    # theta in 0-2pi; phi in 0-pi
    Y = sph_harm(abs(m), l, theta, phi)
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real
    # real spherical harmonic
    return Y.real

def _radial_part(n: int, r: typing.Union[float, np.ndarray]):
    assert n > 0
    return r**(n-1) * np.exp ** (-1.0 * r)

def wavefunction(n, l, m, r, theta, phi):
    # theta in 0-2pi; phi in 0-pi
    return _radial_part(n, r) * sh_functions(l, m, theta, phi)

 
from automaticTB.examples.AH3.structure import get_nncluster_AH3
from automaticTB.structure import NearestNeighborCluster

nncluster = get_nncluster_AH3()

def get_ccpt_from_nncluster(nncluster: NearestNeighborCluster) -> tuple:
    # get cell size: 
    rmax = -1.0
    for site in nncluster.baresites:
        rmax = max(rmax, np.linalg.norm(site.pos))

    cell = 4.0 * rmax * np.eye(3)
    cartesian_position = np.vstack([
        np.array([0.5, 0.5, 0.5]) * 4.0 * rmax + site.pos for site in nncluster.baresites
    ])
    types = [
        site.atomic_number for site in nncluster.baresites
    ]

    return cell, cartesian_position, types

def get_ccpt_from_named_lc(lc: LinearCombination) -> tuple:
    # get cell size: 
    rmax = -1.0
    for site in lc.sites:
        rmax = max(rmax, np.linalg.norm(site.pos))

    cell = 4.0 * rmax * np.eye(3)
    cartesian_position = np.vstack([
        np.array([0.5, 0.5, 0.5]) * 4.0 * rmax + site.pos for site in nncluster.baresites
    ])
    types = [
        site.atomic_number for site in lc.sites
    ]

    return cell, cartesian_position, types

def write_cube_file(cc, p, t, density: np.ndarray, filename: str):
    # assume density is accessed by density[i,j,k]
    nx, ny, nz = density.shape
    result  = "CPMD CUBE FILE.\n"
    result += "OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n"
    result += "{:>6d} {:>12.6f}{:>12.6f}{:>12.6f}\n".format(len(t), 0, 0, 0)
    for i, row in enumerate(cc):
        n = density.shape[i]
        vector = row / n
        result += "{:>6d} {:>12.6f}{:>12.6f}{:>12.6f}\n".format(n, *vector)
    
    for type, position in zip(t, p):
        result += "{:>6d} {:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}\n".format(type, 0.0, *position)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                result += "{:>12.6e}".format(density[ix, iy, iz])
                if iz % 6 == 5:
                    result += "\n"
            result += "\n"

    with open(filename, 'w') as f:
        f.write(result)

cc, p, t = get_ccpt_from_nncluster(nncluster)
density = np.zeros((10,10,10))
write_cube_file(cc, p, t, density, "example.cube")

def plot_MOs(lcs: typing.List[LinearCombination], gridsize: typing.List[int]):
    cc, p, t = get_ccpt_from_named_lc(lcs[0])
    density = np.zeros(gridsize)
    for lc in lcs:
        for pos, orb in zip(lc.sites, lc.orbital_list.orbital_list):
            for sh in orb.sh_list:
                
    
