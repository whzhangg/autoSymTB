import numpy as np
import typing, spglib, dataclasses
import pymatgen.core.structure as matgenStructure
from DFTtools.SiteSymmetry.utilities import rotation_fraction_to_cartesian
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup

from .nncluster import NearestNeighborCluster
from .sites import CrystalSite
from ..parameters import symprec
from ..atomic_orbitals import Orbitals, OrbitalsList


def _get_home_position_and_cell_translation(fractional_positions: np.ndarray):
    translation = np.floor(fractional_positions)
    home = fractional_positions - translation
    return home, translation


def _position_string(pos: np.ndarray):
    return "{:>+12.6f}{:>+12.6f}{:>+12.6f}".format(pos[0], pos[1], pos[2])


def _get_SiteSymmetryGroup_from_crystalsites_and_rotations(
    crystalsites: typing.List[CrystalSite], operations: typing.List[np.ndarray]
) -> SiteSymmetryGroup:
    baresites = [ csite.site for csite in crystalsites ]
    symmetries: list = []
    for sym in operations:
        is_symmetry = True
        for site in baresites:
            newsite = site.rotate(sym)
            if not newsite in baresites: 
                is_symmetry = False
                break
        if is_symmetry:
            symmetries.append(sym)
        
    return SiteSymmetryGroup.from_cartesian_matrices(symmetries)


@dataclasses.dataclass
class Structure:
    cell: np.ndarray
    positions: typing.List[np.ndarray]
    types: typing.List[int]
    nnclusters: typing.List[NearestNeighborCluster]

    @classmethod
    def from_cpt_rcut(
        cls, cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int], 
        orbital_dict: typing.Dict[str, int], rcut: float
    ): # -> Structure
        pcell = spglib.find_primitive((cell, positions, types))
        rotations = spglib.get_symmetry(pcell, symprec = symprec)["rotations"]
        c, p, t = pcell
        p, _ = _get_home_position_and_cell_translation(p)  # so that p between [0,1)
        fraction_position_reference = {
            _position_string(pos):i for i, pos in enumerate(p)
        }
        cartesian_rotations = rotation_fraction_to_cartesian(rotations, c)
        cartesian_p = np.einsum("ji, kj -> ki", c, p) # note we use ji, so no transpose necessary

        pymatgen_structure = matgenStructure.Structure(lattice = c, species = t, coords = p)
        nnsites = []
        for i, cpos in enumerate(cartesian_p):
            crystalsites = []
            orbitals = []
            sites = pymatgen_structure.get_sites_in_sphere(cpos, rcut)
            fraction_pos = np.vstack([
                np.array([site.a, site.b, site.c]) for site in sites
            ])
            home_pos, translation = _get_home_position_and_cell_translation(fraction_pos)
            for hp, tr, site in zip(home_pos, translation, sites):
                index = fraction_position_reference[_position_string(hp)]
                relative_cartesian = np.array([site.x, site.y, site.z]) - cpos
                atomic_type = t[index]
                csite = CrystalSite.from_data(atomic_type, relative_cartesian, index, tr)
                crystalsites.append(csite)
                orbitals.append(
                    Orbitals(orbital_dict[csite.site.chemical_symbol])
                )
            
            group = _get_SiteSymmetryGroup_from_crystalsites_and_rotations(
                crystalsites, cartesian_rotations
            )
            nnsites.append(
                NearestNeighborCluster(crystalsites, OrbitalsList(orbitals), group)
            )

        return cls(c, p, t, nnsites)
        
        
    @classmethod
    def from_cpt_Voronoi(
        cls, cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int]
    ): # structure
        pass
