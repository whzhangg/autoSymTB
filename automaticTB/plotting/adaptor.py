import typing, dataclasses
import numpy as np
from ..SALCs import LinearCombination


@dataclasses.dataclass
class Wavefunction:
    l: int
    m: int
    coeff: float
    # n: int  -> we don't have support for n in any case


@dataclasses.dataclass
class WavefunctionsOnSite:
    cartesian_pos: np.ndarray
    atomic_number: int
    chemical_symbol: str
    wfs: typing.List[Wavefunction]


def get_cell_from_origin_centered_positions(positions: typing.List[np.ndarray]) -> np.ndarray:
    rmax = -1.0
    for pos in positions:
        rmax = max(rmax, np.linalg.norm(pos))
    cell = 4.0 * rmax * np.eye(3)
    return cell


@dataclasses.dataclass
class MolecularWavefunction:
    cell: np.ndarray
    atomic_wavefunctions: typing.List[WavefunctionsOnSite]


    @classmethod
    def from_linear_combination(cls, lc: LinearCombination) -> "MolecularWavefunction":
        # confining box
        cell = get_cell_from_origin_centered_positions([site.pos for site in lc.sites])
        shift = np.dot(cell.T, np.array([0.5,0.5,0.5]))

        atomic_wavefunctions = []
        for site, orbital_slice in zip(lc.sites, lc.orbital_list.atomic_slice_dict):
            shs = lc.orbital_list.sh_list[orbital_slice]
            coeffs = lc.coefficients[orbital_slice]
            wfs: typing.List[Wavefunction] = []
            for sh, coeff in zip(shs, coeffs):
                wfs.append(
                    Wavefunction(sh.l, sh.m, coeff)
                )
            atomic_wavefunctions.append(
                WavefunctionsOnSite(
                    site.pos + shift, site.atomic_number, site.chemical_symbol, wfs
                )
            )

        return cls(cell, atomic_wavefunctions)


    @property
    def types(self) -> typing.List[int]:
        return [aw.atomic_number for aw in self.atomic_wavefunctions]


    @property
    def cartesian_positions(self) -> np.ndarray:
        # we put the origin of the positions to the center, but we probably need to set the sites
        return np.vstack(
            [aw.cartesian_pos for aw in self.atomic_wavefunctions]
        )


    # the same geometry
    def __eq__(self, other) -> bool:
        return np.allclose(self.cell, other.cell) \
            and self.types == other.types \
            and np.allclose(self.cartesian_positions, other.cartesian_positions)
