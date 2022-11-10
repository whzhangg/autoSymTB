import typing, spglib, dataclasses, numpy as np, pymatgen.core.structure as matgenStructure
from pymatgen.analysis.local_env import VoronoiNN
from automaticTB.sitesymmetry import rotation_fraction_to_cartesian, SiteSymmetryGroup
from .sites import LocalSite, CrystalSite, NeighborCluster
from ..parameters import symprec, zero_tolerance
from ..tools import read_cif_to_cpt
from ._utilities import (
    get_absolute_pos_from_cell_fractional_pos_translation, 
    get_home_position_and_cell_translation,
    position_string
)

__all__ = ["Structure", "get_structure_from_cif_file"]


def get_structure_from_cif_file(
    filename: str, rcut: typing.Optional[float] = None
) -> "Structure":
    """
    high level function to get a structure object from cif file
    """
    c, p, t = read_cif_to_cpt(filename)
    if rcut is None:
        return Structure.from_cpt_voronoi(c, p, t)
    else:
        return Structure.from_cpt_rcut(c, p, t, rcut)


@dataclasses.dataclass
class Structure:
    cell: np.ndarray
    positions: typing.List[np.ndarray]
    types: typing.List[int]
    nnclusters: typing.List[NeighborCluster]

    @property
    def eqivalent_indices(self) -> typing.List:
        """
        a list containing the equivalent atoms
        """
        sym_data = spglib.get_symmetry(
            (self.cell, self.positions, self.types), 
            symprec = symprec
        )

        return sym_data["equivalent_atoms"]


    @classmethod
    def from_cpt_rcut(
        cls, cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int],
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
            position_string(pos):i for i, pos in enumerate(p)
        }

        sym_data = spglib.get_symmetry((c,p,t), symprec = symprec)
        rotations = sym_data["rotations"]
        eq_indices = sym_data["equivalent_atoms"]
        cartesian_rotations = rotation_fraction_to_cartesian(rotations, c)

        pymatgen_structure = matgenStructure.Structure(lattice = c, species = t, coords = p)
        nnsites: typing.List[NeighborCluster] = []
        for _, cpos in enumerate(cartesian_p):
            sites = pymatgen_structure.get_sites_in_sphere(cpos, rcut)
            fraction_pos = np.vstack([
                np.array([site.a, site.b, site.c]) for site in sites
            ])
            home_pos, translation = get_home_position_and_cell_translation(fraction_pos)

            crystalsites: typing.List[CrystalSite] = []
            for hp, tr, site in zip(home_pos, translation, sites):
                index = fractional_position_index_reference[position_string(hp)]
                relative_cartesian = np.array([site.x, site.y, site.z]) - cpos
                localsite = LocalSite(t[index], relative_cartesian)
                crystalsites.append(
                    CrystalSite(
                        localsite,
                        get_absolute_pos_from_cell_fractional_pos_translation(c, p[index], tr),
                        index, 
                        eq_indices[index],
                        tr
                    )
                )
                
            nnsites += get_neighborcluster_from_crystalsites_rotations(
                crystalsites, cartesian_rotations
            )

        return cls(c, p, t, nnsites)


    @classmethod 
    def from_cpt_voronoi(
        cls, cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int],
        rcut: float = 5.0, 
        weight_cutoff: float = 1e-1
    ) -> "Structure":
        """
        Paramters:
        - weight_cutoff: the cutoff in solid angle, discarding very small solid angle
        """
        c, p, t = cls.get_standarized_cpt(cell, positions, types)
        cartesian_p = np.einsum("ji, kj -> ki", c, p) # note we use ji, so no transpose necessary

        sym_data = spglib.get_symmetry((c,p,t), symprec = symprec)
        rotations = sym_data["rotations"]
        eq_indices = sym_data["equivalent_atoms"]
        cartesian_rotations = rotation_fraction_to_cartesian(rotations, c)

        pymatgen_structure = matgenStructure.Structure(lattice = c, species = t, coords = p)
        neighborNN = VoronoiNN(cutoff = rcut, compute_adj_neighbors = False)

        nnsites: typing.List[NeighborCluster] = []
        for i, cpos in enumerate(cartesian_p):
            crystalsites: typing.List[CrystalSite] = []
            nn_info = neighborNN.get_nn_info(pymatgen_structure, i)
            # since voronoi nn_info does not contain the atom itself, we add it

            crystalsites.append(
                CrystalSite(
                    LocalSite(t[i], np.zeros_like(cpos)),
                    cpos, i, eq_indices[i], np.zeros_like(cpos)
                )
            )
            
            # other nnsite
            for nn in [nn for nn in nn_info if nn["weight"] > weight_cutoff]:
                index = nn["site_index"]
                tr = nn["image"]
                cartesian = cartesian_p[index]
                relative_cartesian = cartesian + np.dot(c.T, tr) - cpos
                localsite = LocalSite(t[index], relative_cartesian)
                crystalsites.append(
                    CrystalSite(
                        localsite, 
                        get_absolute_pos_from_cell_fractional_pos_translation(c, p[index], tr),
                        index, eq_indices[index], tr
                    )
                )
                
            nnsites += get_neighborcluster_from_crystalsites_rotations(
                crystalsites, cartesian_rotations
            )
            
        return cls(c, p, t, nnsites)


    @staticmethod
    def get_standarized_cpt(cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int]) -> tuple:
        """
        to find and use the primitive cell only
        """
        pcell = spglib.find_primitive((cell, positions, types))
        c, p, t = pcell
        p, _ = get_home_position_and_cell_translation(p)  # so that p between [0,1)
        return c, p, t


def get_neighborcluster_from_crystalsites_rotations(
    crystalsites: typing.List[CrystalSite], cartesian_rotations: typing.List[np.ndarray]
) -> typing.List[NeighborCluster]:
    nnsites = []
    bare_sites = [ csite.site for csite in crystalsites ]
    group, equivalentset = _get_siteSym_eqGroup_from_sites_and_rots(
                                bare_sites, cartesian_rotations
                            )
            
    center_index: int = None
    for key, value in equivalentset.items():
        if  len(value) == 1 and \
            value[0] == key and \
            np.linalg.norm(bare_sites[key].pos) < zero_tolerance:
            center_index = key
            break

    for indices in equivalentset.values():
        nnsites.append(
            NeighborCluster(
                crystalsites[center_index], 
                [crystalsites[i] for i in indices], 
                group
            )
        )

    return sorted(nnsites, key=lambda nn: len(nn.neighbor_sites))


def _get_siteSym_eqGroup_from_sites_and_rots(
    baresites: typing.List[LocalSite], operations: typing.List[np.ndarray]
) -> typing.Tuple[SiteSymmetryGroup, typing.Dict[int, typing.List[int]]]:
    """
    return the site symmetry group, as well as the set of atomic index that are equivalent
    """
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
    
    group = SiteSymmetryGroup.from_cartesian_matrices(symmetries)

    equivalent_result = {}
    found = []
    for i, site in enumerate(baresites):
        if i in found: continue
        for rot in group.operations:
            newsite = site.rotate(rot)
            for j,othersite in enumerate(baresites):
                if newsite == othersite:
                    equivalent_result.setdefault(i, set()).add(j)
                    found.append(j)
    
    for k, v in equivalent_result.items():
        equivalent_result[k] = list(v)

    return group, equivalent_result
