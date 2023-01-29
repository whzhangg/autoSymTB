import dataclasses
import numpy as np

from automaticTB import tools

'''
@dataclasses.dataclass
class LocalSite:
    atomic_number: int
    pos: np.ndarray

    @property
    def chemical_symbol(self) -> str:
        return tools.chemical_symbols[self.atomic_number]

    def __eq__(self, other) -> bool:
        return self.atomic_number == other.atomic_number and np.allclose(self.pos, other.pos)

    def rotate(self, matrix:np.ndarray):
        newpos = matrix.dot(self.pos)
        return LocalSite(self.atomic_number, newpos)

    def __repr__(self) -> str:
        atom = self.chemical_symbol
        return "{:>2s} @ ({:> 6.2f},{:> 6.2f},{:> 6.2f})".format(atom, *self.pos)
'''

@dataclasses.dataclass
class CrystalSite:
    atomic_number: int
    absolute_position: np.ndarray
    index_pcell: int
    equivalent_index: int
    translation: np.ndarray
    orbitals: str   # eg: 3s3p4s

    def __str__(self) -> str:
        return f"{self.site.chemical_symbol} (i={self.index_pcell:>2d}) @ " + \
                "({:>5.2f},{:>5.2f},{:>5.2f})".format(*self.site.pos)

    def __repr__(self) -> str:
        return str(self.site) + \
            "{:>3d} t = {:>+4.1f}{:>+4.1f}{:>+4.1f}".format(self.index_pcell, *self.translation)

    def __eq__(self, other) -> bool:
        return self.site == other.site and \
               self.index_pcell == other.index_pcell and \
               np.allclose(self.translation, other.translation)

    @property
    def chemical_symbol(self) -> str:
        return tools.chemical_symbols[self.atomic_number]
