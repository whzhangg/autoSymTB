import numpy as np
import typing, spglib, dataclasses, collections
import pymatgen.core.structure as matgenStructure
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup
from automaticTB.structure.sites import CrystalSite, NearestNeighborSites
from automaticTB.linear_combination import Orbitals
from DFTtools.SiteSymmetry.utilities import rotation_fraction_to_cartesian
from ase.data import chemical_symbols


def get_home_position_and_cell_translation(fractional_positions: np.ndarray):
    translation = np.floor(fractional_positions)
    home = fractional_positions - translation
    return home, translation


def position_string(pos: np.ndarray):
    return "{:>+12.6f}{:>+12.6f}{:>+12.6f}".format(pos[0], pos[1], pos[2])


Atomlm = collections.namedtuple("Atomlm", "i l m")

class Structure:
    symprec: float = 1e-4
    def __init__(self,
        cell: np.ndarray,
        positions: typing.List[np.ndarray],
        types: typing.List[int],
        clusters: typing.List[NearestNeighborSites],
        symmetry_groups: typing.List[SiteSymmetryGroup]
    ):
        self._cell = cell
        self._positions = positions
        self._types = types
        self._clusters = clusters
        self._symmetry_groups = symmetry_groups


    def get_basis_for_vectorspace(self, orbital_dict: typing.Dict[str, Orbitals]):
        all_orbitals = []
        for it, t in enumerate(self._types):
            all_orbitals += [
                Atomlm(it, sp.l, sp.m) for sp in orbital_dict[chemical_symbols[t]].aolist
            ]
        
        for 


    @classmethod
    def from_cpt_rcut(
        cls, cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int], rcut: float
    ): # -> Structure
        pcell = spglib.find_primitive((cell, positions, types))
        rotations = spglib.get_symmetry(pcell, symprec = cls.symprec)["rotations"]
        c, p, t = pcell
        p, _ = get_home_position_and_cell_translation(p)  # so that p between [0,1)
        fraction_position_reference = {
            position_string(pos):i for i, pos in enumerate(p)
        }
        cartesian_rotations = rotation_fraction_to_cartesian(rotations, c)
        cartesian_p = np.einsum("ji, kj -> ki", c, p) # note we use ji, so no transpose necessary

        pymatgen_structure = matgenStructure.Structure(lattice = c, species = t, coords = p)
        nnsites = []
        ssgroups = []
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
            nnsite = NearestNeighborSites(crystalsites, cpos)
            site_symmetry = nnsite.find_site_symmetry_operation(cartesian_rotations)
            group = SiteSymmetryGroup.from_cartesian_matrices(site_symmetry)
            nnsites.append(nnsite)
            ssgroups.append(group)

        return cls(c, p, t, nnsites, ssgroups)
        
        
    @classmethod
    def from_cpt_Voronoi(
        cls, cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int]
    ): # structure
        pass


if __name__ == "__main__":
    from DFTtools.tools import read_generic_to_cpt
    c,p,t = read_generic_to_cpt("/Users/wenhao/work/code/python/DFTtools/tests/structures/Si_ICSD51688.cif")
    structure: Structure = Structure.from_cpt_rcut(c, p, t, 3.0)
    orbital = {
        "Si": Orbitals("s p")
    }
    structure.get_basis_for_vectorspace(orbital)

