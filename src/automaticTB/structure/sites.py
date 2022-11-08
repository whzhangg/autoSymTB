import dataclasses, typing, numpy as np
from ..tools import chemical_symbols, get_cell_from_origin_centered_positions
from ._utilities import write_cif_file, atom_from_cpt_cartesian
from ..sitesymmetry import SiteSymmetryGroup
from ..parameters import zero_tolerance

__all__ = ["LocalSite", "CrystalSite", "NeighborCluster"]

@dataclasses.dataclass
class LocalSite:
    atomic_number: int
    pos: np.ndarray

    @property
    def chemical_symbol(self) -> str:
        return chemical_symbols[self.atomic_number]

    def __eq__(self, other) -> bool:
        return self.atomic_number == other.atomic_number and np.allclose(self.pos, other.pos)

    def rotate(self, matrix:np.ndarray):
        newpos = matrix.dot(self.pos)
        return LocalSite(self.atomic_number, newpos)

    def __repr__(self) -> str:
        atom = self.chemical_symbol
        return "{:>2s} @ ({:> 6.2f},{:> 6.2f},{:> 6.2f})".format(atom, *self.pos)


@dataclasses.dataclass
class CrystalSite:
    site: LocalSite # absolute cartesian coordinates
    absolute_position: np.ndarray
    index_pcell: int
    translation: np.ndarray

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


@dataclasses.dataclass
class NeighborCluster:
    center_site: CrystalSite
    neighbor_sites: typing.List[CrystalSite]
    sitesymmetrygroup: SiteSymmetryGroup

    @property
    def distance(self) -> float:
        vectors = np.vstack([ ns.site.pos for ns in self.neighbor_sites ])
        distances = np.linalg.norm(vectors, axis=1)
        assert np.allclose(distances, np.min(distances), zero_tolerance)
        return np.min(distances)


    def __str__(self) -> str:
        results = [f"cluster ({self.sitesymmetrygroup.groupname:>5s}) "]
        results.append(f"centered on {self.center_site.site.chemical_symbol} " + \
                   f"(i={self.center_site.index_pcell:>2d})")
        results.append(f"         -> Distance : {self.distance:>6f}")
        results.append("-"*31)
        for csite in self.neighbor_sites:
            results.append(str(csite))
        results.append("-"*31)
        return "\n".join(results)


    def write_to_cif(self, filename: str) -> None:
        positions = np.array(
            [s.pos for s in self.baresites]
        )
        cell = get_cell_from_origin_centered_positions(positions)
        shift = np.dot(cell.T, np.array([0.5,0.5,0.5]))
        positions += shift
        types = [s.atomic_number for s in self.baresites]
        atom = atom_from_cpt_cartesian(cell, positions, types)
        write_cif_file(filename, atom)