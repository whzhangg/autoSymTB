import dataclasses, typing
import numpy as np
from ase.data import chemical_symbols


@dataclasses.dataclass
class BareSite:
    atomic_number: int
    pos: np.ndarray

    def __eq__(self, other) -> bool:
        return self.atomic_number == other.atomic_number and np.allclose(self.pos, other.pos)

    def rotate(self, matrix:np.ndarray):
        newpos = matrix.dot(self.pos)
        return BareSite(self.atomic_number, newpos)

    def __repr__(self) -> str:
        atom = chemical_symbols[self.atomic_number]
        return "{:>2s} @ ({:> 6.2f},{:> 6.2f},{:> 6.2f})".format(atom, *self.pos)


@dataclasses.dataclass
class CrystalSite:
    site: BareSite # absolute cartesian coordinates
    index_pcell: int
    supercell_shift: np.ndarray

    @classmethod
    def from_data(
        cls, atomic_number: int, pos: np.ndarray, index_pcell: int, supercell_shift: np.ndarray
    ):
        return cls(
            BareSite(atomic_number, pos), index_pcell, supercell_shift
        )

    def __eq__(self, other) -> bool:
        return self.site == other.site and \
               self.index_pcell == other.index_pcell and \
               np.allclose(self.supercell_shift, other.supercell_shift)


class NearestNeighborSites:
    def __init__(self, listofsites: typing.List[CrystalSite], origin: np.ndarray) -> None:
        self._origin = origin

        self._pindexs = [ site.index_pcell for site in listofsites ]
        self._translations = np.vstack([ site.supercell_shift for site in listofsites ])
        self._baresites = [ site.site for site in listofsites ]

    @property
    def num_sites(self):
        return len(self._pindexs)

    def __str__(self):
        result = "cluster center: {:> 6.2f},{:> 6.2f},{:> 6.2f}\n".format(*self._origin)
        for bs, pi, translation in zip(self._baresites, self._pindexs, self._translations):
            result += str(bs) + "{:>3d} t = {:>+4.1f}{:>+4.1f}{:>+4.1f}\n".format(pi, *translation)
        return result

    def site_permutation_by_symmetry_rotation(self, cartesian_rotation: np.ndarray) -> typing.List[int]:
        index_of_rotated = []
        to_compare = range(self.num_sites)
        for baresite in self._baresites:
            rotated_site = baresite.rotate(cartesian_rotation)
            for i in to_compare:
                if rotated_site == self._baresites[i]:
                    index_of_rotated.append(i)
                    to_compare.remove(i)
                    break
                raise ValueError("rotated site cannot be found")
        return index_of_rotated

    def find_site_symmetry_operation(self, operations: typing.List[np.ndarray]) -> typing.List[np.ndarray]:
        result: list = []
        for isym, sym in enumerate(operations):
            is_symmetry = True
            for site in self._baresites:
                newsite = site.rotate(sym)
                if not newsite in self._baresites: 
                    is_symmetry = False
                    break
            if is_symmetry:
                result.append(sym)
        return result