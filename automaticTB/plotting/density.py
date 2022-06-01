import numpy as np
import typing
from ..SALCs import LinearCombination
from .wavefunctions import wavefunction

Bohr = 1.8897259886

def write_cube_file(cc, p, t, density: np.ndarray, filename: str):
    # assume density is accessed by density[i,j,k]
    # note that unit is in Bohr
    nx, ny, nz = density.shape
    result  = "CPMD CUBE FILE.\n"
    result += "OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n"
    result += "{:>6d} {:>12.6f}{:>12.6f}{:>12.6f}\n".format(len(t), 0, 0, 0)
    for i, row in enumerate(cc):
        n = density.shape[i]
        vector = row / n * Bohr
        result += "{:>6d} {:>12.6f}{:>12.6f}{:>12.6f}\n".format(n, *vector)
    
    for type, position in zip(t, p):
        p = position * Bohr
        result += "{:>6d} {:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}\n".format(type, 0.0, *p)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                result += "{:>20.6e}".format(density[ix, iy, iz])
                if iz % 6 == 5:
                    result += "\n"
            result += "\n"

    with open(filename, 'w') as f:
        f.write(result)


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


class DensityPlot:
    n = 1
    gridsize = {
        "low":      [25,25,25],
        "standard": [50,50,50],
        "high":     [75,75,75],
    }

    def __init__(self, lcs: typing.List[LinearCombination], quality: str = "standard") -> None:
        rmax = -1.0
        for site in lcs[0].sites:
            rmax = max(rmax, np.linalg.norm(site.pos))

        self._cell = 4.0 * rmax * np.eye(3)
        self._cart_positions = np.vstack([
            np.array([0.5, 0.5, 0.5]) * 4.0 * rmax + site.pos for site in lcs[0].sites
        ])
        self._types = [
            site.atomic_number for site in lcs[0].sites
        ]

        self._orbital_slice = lcs[0].orbital_list.atomic_slice_dict
        self._lcs = lcs

        self._xyz = self._make_grid(self.gridsize[quality])


    def make_cube_file(self, filename: str):
        density = self._get_density_at_x(self._xyz)
        write_cube_file(self._cell, self._cart_positions, self._types, density, filename)


    def _make_grid(self, grid: typing.List[int]) -> np.ndarray:
        nx = np.linspace(0, 1, grid[0]+1) # to include 1.0 as the end point
        ny = np.linspace(0, 1, grid[1]+1) # to include 1.0 as the end point
        nz = np.linspace(0, 1, grid[2]+1) # to include 1.0 as the end point
        
        x,y,z = np.meshgrid(nx, ny, nz, indexing="ij")
        positions = np.concatenate((x[...,np.newaxis],y[...,np.newaxis],z[...,np.newaxis]),axis=3)
        return np.einsum("ij,abcj -> abci", self._cell.T, positions)


    def _get_density_at_x(self, x: np.ndarray) -> float:
        result = np.zeros_like(x[...,0])
        for pos, o_slice in zip(self._cart_positions, self._orbital_slice):
            r_theta_phi = xyz_to_r_theta_phi(x - pos)
            r = r_theta_phi[..., 0]
            theta = r_theta_phi[...,1]
            phi = r_theta_phi[...,2]

            for lc in self._lcs:
                related_coefficients = lc.coefficients[o_slice]
                shs = lc.orbital_list.sh_list[o_slice]
                for coeff, sh in zip(related_coefficients, shs):
                    result += coeff.real * wavefunction(self.n, sh.l, sh.m, r, theta, phi)
        return result

