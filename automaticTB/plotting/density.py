import numpy as np
import typing
from .wavefunctions import wavefunction, xyz_to_r_theta_phi
from .adaptor import MolecularWavefunction
from ..SALCs import LinearCombination


Bohr = 1.8897259886


class DensityCubePlot:
    n = 1
    gridsize = {
        "low":      [25,25,25],
        "standard": [50,50,50],
        "high":     [75,75,75],
    }


    def __init__(self, mos: typing.List[MolecularWavefunction], quality: str = "standard") -> None:
        # check MOs are good
        first = mos[0]
        for mo in mos:
            assert mo == first
            
        self._types = mos[0].types
        self._cell = mos[0].cell
        self._positions = mos[0].cartesian_positions

        self._mos = mos
        self._xyz = self._make_grid(self.gridsize[quality])


    @classmethod
    def from_linearcombinations(cls, lcs: typing.List[LinearCombination], quality: str = "standard") -> "DensityCubePlot":
        mos = [
            MolecularWavefunction.from_linear_combination(lc) for lc in lcs
        ]
        return cls(mos, quality)


    def plot_to_file(self, filename: str) -> None:
        density = self._get_density_at_x(self._xyz)
        nx, ny, nz = density.shape
        result  = "CPMD CUBE FILE.\n"
        result += "OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n"
        result += "{:>6d} {:>12.6f}{:>12.6f}{:>12.6f}\n".format(len(self._types), 0, 0, 0)
        for i, row in enumerate(self._cell):
            n = density.shape[i]
            vector = row / n * Bohr
            result += "{:>6d} {:>12.6f}{:>12.6f}{:>12.6f}\n".format(n, *vector)
        
        for type, position in zip(self._types, self._positions):
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


    def _make_grid(self, grid: typing.List[int]) -> np.ndarray:
        nx = np.linspace(0, 1, grid[0]+1) # to include 1.0 as the end point
        ny = np.linspace(0, 1, grid[1]+1) # to include 1.0 as the end point
        nz = np.linspace(0, 1, grid[2]+1) # to include 1.0 as the end point
        
        x,y,z = np.meshgrid(nx, ny, nz, indexing="ij")
        positions = np.concatenate((x[...,np.newaxis],y[...,np.newaxis],z[...,np.newaxis]),axis=3)
        return np.einsum("ij,abcj -> abci", self._cell.T, positions)


    def _get_density_at_x(self, x: np.ndarray) -> float:
        result = np.zeros_like(x[...,0])
        for mo in self._mos:
            for aw in mo.atomic_wavefunctions:
                r_theta_phi = xyz_to_r_theta_phi(x - aw.cartesian_pos)
                r = r_theta_phi[..., 0]
                theta = r_theta_phi[..., 1]
                phi = r_theta_phi[..., 2]
                for wf in aw.wfs:
                    result += wf.coeff.real * wavefunction(self.n, wf.l, wf.m, r, theta, phi)
        return result

