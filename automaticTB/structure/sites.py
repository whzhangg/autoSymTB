import dataclasses
import numpy as np
from ase.data import chemical_symbols


@dataclasses.dataclass
class Site:
    atomic_number: int
    pos: np.ndarray

    @property
    def chemical_symbol(self) -> str:
        return chemical_symbols[self.atomic_number]

    def __eq__(self, other) -> bool:
        return self.atomic_number == other.atomic_number and np.allclose(self.pos, other.pos)

    def rotate(self, matrix:np.ndarray):
        newpos = matrix.dot(self.pos)
        return Site(self.atomic_number, newpos)

    def __repr__(self) -> str:
        atom = self.chemical_symbol
        return "{:>2s} @ ({:> 6.2f},{:> 6.2f},{:> 6.2f})".format(atom, *self.pos)


@dataclasses.dataclass
class CrystalSite:
    site: Site # absolute cartesian coordinates
    index_pcell: int
    translation: np.ndarray

    @classmethod
    def from_data(
        cls, atomic_number: int, pos: np.ndarray, index_pcell: int, translation: np.ndarray
    ):
        return cls(
            Site(atomic_number, pos), index_pcell, translation
        )

    def __repr__(self) -> str:
        return str(self.site) + "{:>3d} t = {:>+4.1f}{:>+4.1f}{:>+4.1f}\n".format(self.index_pcell, *self.translation)

    def __eq__(self, other) -> bool:
        return self.site == other.site and \
               self.index_pcell == other.index_pcell and \
               np.allclose(self.translation, other.translation)


