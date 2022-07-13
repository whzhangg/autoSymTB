import numpy as np
import typing, spglib, dataclasses
import pymatgen.core.structure as matgenStructure
from pymatgen.analysis.local_env import VoronoiNN
from automaticTB.sitesymmetry import rotation_fraction_to_cartesian, SiteSymmetryGroup
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
        orbital_dict: typing.Dict[str, typing.List[int]], 
        rcut: float
    ) -> "Structure":
        """
        Parameters:
        - cell: 3 times 3 lattice vector
        - positions: atomic positions in fractional coordinate
        - types: list of atomic number for each atoms
        - orbital_dict: a dictionary specifying atomic orbitals on each atoms
        - rcut: float specifying the radius cutoff
        """
        c, p, t = cls.get_standarized_cpt(cell, positions, types)
        cartesian_p = np.einsum("ji, kj -> ki", c, p) # note we use ji, so no transpose necessary
        fractional_position_index_reference = {
            _position_string(pos):i for i, pos in enumerate(p)
        }

        rotations = spglib.get_symmetry((c,p,t), symprec = symprec)["rotations"]
        cartesian_rotations = rotation_fraction_to_cartesian(rotations, c)

        pymatgen_structure = matgenStructure.Structure(lattice = c, species = t, coords = p)
        nnsites: typing.List[NearestNeighborCluster] = []
        for i, cpos in enumerate(cartesian_p):
            crystalsites = []
            orbitals = []
            sites = pymatgen_structure.get_sites_in_sphere(cpos, rcut)
            fraction_pos = np.vstack([
                np.array([site.a, site.b, site.c]) for site in sites
            ])
            home_pos, translation = _get_home_position_and_cell_translation(fraction_pos)
            for hp, tr, site in zip(home_pos, translation, sites):
                index = fractional_position_index_reference[_position_string(hp)]
                relative_cartesian = np.array([site.x, site.y, site.z]) - cpos
                atomic_type = t[index]
                crystal_site = CrystalSite.from_data(atomic_type, relative_cartesian, index, tr)
                crystalsites.append(crystal_site)
                orbitals.append(
                    Orbitals(orbital_dict[crystal_site.site.chemical_symbol])
                )
            
            group = _get_SiteSymmetryGroup_from_crystalsites_and_rotations(
                crystalsites, cartesian_rotations
            )
            nnsites.append(
                NearestNeighborCluster(crystalsites, OrbitalsList(orbitals), group)
            )

        return cls(c, p, t, nnsites)


    @classmethod 
    def from_cpt_voronoi(
        cls, cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int], 
        orbital_dict: typing.Dict[str, typing.List[int]], 
        rcut: float = 4.0, 
        weight_cutoff: float = 1e-1
    ) -> "Structure":
        """
        Paramters:
        - weight_cutoff: the cutoff in solid angle, discarding the neighbors with very small solid angle
        """
        c, p, t = cls.get_standarized_cpt(cell, positions, types)
        cartesian_p = np.einsum("ji, kj -> ki", c, p) # note we use ji, so no transpose necessary

        rotations = spglib.get_symmetry((c,p,t), symprec = symprec)["rotations"]
        cartesian_rotations = rotation_fraction_to_cartesian(rotations, c)

        pymatgen_structure = matgenStructure.Structure(lattice = c, species = t, coords = p)
        neighborNN = VoronoiNN(cutoff = rcut, compute_adj_neighbors = False)

        nnsites: typing.List[NearestNeighborCluster] = []
        for i, cpos in enumerate(cartesian_p):
            crystalsites = []
            orbitals = []
            nn_info = neighborNN.get_nn_info(pymatgen_structure, i)
            # since voronoi nn_info does not contain the atom itself, we add it
            crystal_site = CrystalSite.from_data(t[i], np.zeros_like(cpos), i, np.zeros_like(cpos))
            crystalsites.append(crystal_site)
            orbitals.append(
                Orbitals(orbital_dict[crystal_site.site.chemical_symbol])
            )
            # other nnsite
            for nn in [nn for nn in nn_info if nn["weight"] > weight_cutoff]:
                index = nn["site_index"]
                atomic_type = t[index]
                tr = nn["image"]
                cartesian = cartesian_p[index]
                relative_cartesian = cartesian + np.dot(c.T, tr) - cpos
                crystal_site = CrystalSite.from_data(atomic_type, relative_cartesian, index, tr)
                crystalsites.append(crystal_site)
                orbitals.append(
                    Orbitals(orbital_dict[crystal_site.site.chemical_symbol])
                )
            
            group = _get_SiteSymmetryGroup_from_crystalsites_and_rotations(
                crystalsites, cartesian_rotations
            )
            nnsites.append(
                NearestNeighborCluster(crystalsites, OrbitalsList(orbitals), group)
            )

        return cls(c, p, t, nnsites)


    @staticmethod
    def get_standarized_cpt(cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int]) -> tuple:
        pcell = spglib.find_primitive((cell, positions, types))
        c, p, t = pcell
        p, _ = _get_home_position_and_cell_translation(p)  # so that p between [0,1)
        return c, p, t
