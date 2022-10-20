import typing, dataclasses
import numpy as np
from ..parameters import zero_tolerance
from ..tools import chemical_symbols

__all__ = ["Wavefunction", "WavefunctionsOnSite", "MolecularWavefunction"]

@dataclasses.dataclass
class Wavefunction:
    n: int
    l: int
    m: int
    coeff: float


@dataclasses.dataclass
class WavefunctionsOnSite:
    """
    defines an atom in space (cartesian coordinate) with some combination of wavefunction on it.
    it is necessary to use a list of wavefunction because, for example, we can have a combination 
    of px + py orbital which give a rotated p orbital. 
    """
    cartesian_pos: np.ndarray
    atomic_number: int
    wfs: typing.List[Wavefunction]

    @property
    def chemical_symbol(self) -> str:
        return chemical_symbols[self.atomic_number]

    def __str__(self) -> str:
        main = "{:s} @ {:>6.2f}{:>6.2f}{:>6.2f}\n".format(
                    self.chemical_symbol, *self.cartesian_pos
                )
        
        for wf in self.wfs:
            if np.abs(wf.coeff) <= zero_tolerance: continue
            main += "n={:d} l={:d} m={:d} coeff = {:>6.2f}".format(
                    wf.n, wf.l,wf.m,wf.coeff.real
                )
    
        return main


@dataclasses.dataclass
class MolecularWavefunction:
    """
    it defines a set of wavefunction on site with a bonding cell.
    """
    cell: np.ndarray
    atomic_wavefunctions: typing.List[WavefunctionsOnSite]


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
