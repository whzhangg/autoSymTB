import numpy as np
import typing, spglib, dataclasses
import pymatgen.core.structure as matgenStructure
from DFTtools.SiteSymmetry.utilities import rotation_fraction_to_cartesian

from .nncluster import NNCluster
from .sites import CrystalSite

symprec = 1e-6

def get_home_position_and_cell_translation(fractional_positions: np.ndarray):
    translation = np.floor(fractional_positions)
    home = fractional_positions - translation
    return home, translation


def position_string(pos: np.ndarray):
    return "{:>+12.6f}{:>+12.6f}{:>+12.6f}".format(pos[0], pos[1], pos[2])


@dataclasses.dataclass
class Structure:
    cell: np.ndarray
    positions: typing.List[np.ndarray]
    types: typing.List[int]
    nnclusters: typing.List[NNCluster]
    cartesian_rotations: typing.List[np.ndarray]

    @property
    def cpt(self) -> typing.Tuple[np.ndarray, typing.List[np.ndarray], typing.List[int]]:
        return (self.cell, self.positions, self.types)

    @classmethod
    def from_cpt_rcut(
        cls, cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int], rcut: float
    ): # -> Structure
        pcell = spglib.find_primitive((cell, positions, types))
        rotations = spglib.get_symmetry(pcell, symprec = symprec)["rotations"]
        c, p, t = pcell
        p, _ = get_home_position_and_cell_translation(p)  # so that p between [0,1)
        fraction_position_reference = {
            position_string(pos):i for i, pos in enumerate(p)
        }
        cartesian_rotations = rotation_fraction_to_cartesian(rotations, c)
        cartesian_p = np.einsum("ji, kj -> ki", c, p) # note we use ji, so no transpose necessary

        pymatgen_structure = matgenStructure.Structure(lattice = c, species = t, coords = p)
        nnsites = []
        for i, cpos in enumerate(cartesian_p):
            crystalsites = []
            sites = pymatgen_structure.get_sites_in_sphere(cpos, rcut)
            fraction_pos = np.vstack([
                np.array([site.a, site.b, site.c]) for site in sites
            ])
            home_pos, translation = get_home_position_and_cell_translation(fraction_pos)
            for hp, tr, site in zip(home_pos, translation, sites):
                index = fraction_position_reference[position_string(hp)]
                relative_cartesian = np.array([site.x, site.y, site.z]) - cpos
                atomic_type = t[index]
                crystalsites.append(
                    CrystalSite.from_data(atomic_type, relative_cartesian, index, tr)
                )
            
            nnsites.append(
                NNCluster(crystalsites, cpos)
            )

        return cls(c, p, t, nnsites, cartesian_rotations)
        
        
    @classmethod
    def from_cpt_Voronoi(
        cls, cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int]
    ): # structure
        pass
