import typing, spglib, dataclasses, numpy as np, pymatgen.core.structure as matgenStructure
from pymatgen.analysis.local_env import VoronoiNN
from .sitesymmetry import (
    rotation_fraction_to_cartesian, SiteSymmetryGroup, GroupsList_sch
)
from .sites import LocalSite, CrystalSite
from ..parameters import symprec, zero_tolerance
from ..tools import read_cif_to_cpt, get_cell_from_origin_centered_positions
from ._utilities import (
    get_absolute_pos_from_cell_fractional_pos_translation, 
    get_home_position_and_cell_translation,
    position_string,
    write_cif_file, 
    atom_from_cpt_cartesian, atom_from_cpt_fractional
)
from .pymatgen_sym import symOperations_from_pos_types


__all__ = ["Structure", "CenteredCluster", "CenteredEquivalentCluster"]


class Structure:
    """
    [Structure] -> CenteredCluster -> CenteredEquivalentCluster
    """
    voronoi_rcut: float = 5.0
    voronoi_weight_cutoff: float = 1e-1

    @staticmethod
    def get_standarized_cpt(
        cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int]
    ) -> tuple:
        """
        to find and use the primitive cell only
        """
        pcell = spglib.find_primitive((cell, positions, types))
        c, p, t = pcell
        p, _ = get_home_position_and_cell_translation(p)  # so that p between [0,1)
        return c, p, t


    def __init__(self, 
        cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int],
        atomic_orbital_dict: typing.Dict[str, str], 
        rcut: typing.Optional[float] = None
    ) -> None:
        # main properties
        self.cell, self.positions, self.types = self.get_standarized_cpt(cell, positions, types)

        self._cartesian_pos = np.einsum("ji, kj -> ki", self.cell, self.positions)
        self._atomic_orbital_dict = atomic_orbital_dict
        self._rcut = rcut

        sym_data = spglib.get_symmetry(
            (self.cell, self.positions, self.types), symprec = symprec
        )
        self._cartesian_rotations = rotation_fraction_to_cartesian(
            sym_data["rotations"], self.cell
        )
        self._eq_indices = sym_data["equivalent_atoms"]


    @property
    def centered_clusters(self) -> typing.List["CenteredCluster"]:
        if self._rcut is None:
            return self._get_centered_cluster_rcut()
        else:
            return self._get_centered_cluster_voronoi()


    def _get_centered_cluster_rcut(self) -> typing.List["CenteredCluster"]:
        """
        get CenteredCluster using rcut method
        """
        c, p, t = self.cell, self.positions, self.types
        fractional_position_index_reference = {
            position_string(pos):i for i, pos in enumerate(p)
        }

        pymatgen_structure = matgenStructure.Structure(lattice = c, species = t, coords = p)
        clusters = []
        for _, cpos in enumerate(self._cartesian_pos):
            sites = pymatgen_structure.get_sites_in_sphere(cpos, self._rcut)
            fraction_pos = np.vstack([
                np.array([site.a, site.b, site.c]) for site in sites
            ])
            home_pos, translation = get_home_position_and_cell_translation(fraction_pos)

            neighbor_csites: typing.List[CrystalSite] = []
            for hp, tr, site in zip(home_pos, translation, sites):
                index = fractional_position_index_reference[position_string(hp)]
                relative_cartesian = np.array([site.x, site.y, site.z]) - cpos
                _localsite = LocalSite(t[index], relative_cartesian)
                _pos = get_absolute_pos_from_cell_fractional_pos_translation(c, p[index], tr)
                _crystalsite = \
                    CrystalSite(
                        site = _localsite,
                        absolute_position = _pos,
                        index_pcell = index, 
                        equivalent_index = self._eq_indices[index],
                        translation = tr,
                        orbitals = self._atomic_orbital_dict[_localsite.chemical_symbol]
                    )

                if np.linalg.norm(relative_cartesian) < zero_tolerance:
                    center_csite = _crystalsite
                else:
                    neighbor_csites.append(_crystalsite)
            
            clusters.append(
                CenteredCluster(
                    center_site = center_csite,
                    neighbor_sites = neighbor_csites,
                    cart_rotations = self._cartesian_pos
                )
            )
        
        return clusters


    def _get_centered_cluster_voronoi(self) -> typing.List["CenteredCluster"]:
        """
        get CenteredCluster using voronoi method
        """
        c, p, t = self.cell, self.positions, self.types
        pymatgen_structure = matgenStructure.Structure(lattice = c, species = t, coords = p)
        neighborNN = VoronoiNN(cutoff = self.voronoi_rcut, compute_adj_neighbors = False)
        
        clusters = []
        for i, cpos in enumerate(self._cartesian_pos):
            nn_info = neighborNN.get_nn_info(pymatgen_structure, i)
            # since voronoi nn_info does not contain the atom itself, we add it

            _localsite = LocalSite(t[i], np.zeros_like(cpos))
            center_csite = \
                CrystalSite(
                    site = _localsite,
                    absolute_position = cpos,
                    index_pcell = i,
                    equivalent_index = self._eq_indices[i],
                    translation = np.zeros_like(cpos),
                    orbitals = self._atomic_orbital_dict[_localsite.chemical_symbol]
                )
            
            # other nnsite
            neighbor_csites: typing.List[CrystalSite] = []
            for nn in [nn for nn in nn_info if nn["weight"] > self.voronoi_weight_cutoff]:
                index = nn["site_index"]
                tr = nn["image"]
                cartesian = self._cartesian_pos[index]
                relative_cartesian = cartesian + np.dot(c.T, tr) - cpos
                _localsite = LocalSite(t[index], relative_cartesian)
                _pos = get_absolute_pos_from_cell_fractional_pos_translation(c, p[index], tr)
                neighbor_csites.append(
                    CrystalSite(
                        site = _localsite, 
                        absolute_position = _pos,
                        index_pcell = index,
                        equivalent_index = self._eq_indices[index],
                        translation = tr, 
                        orbitals = self._atomic_orbital_dict[_localsite.chemical_symbol]
                    )
                )
            
            clusters.append(
                CenteredCluster(
                    center_site = center_csite,
                    neighbor_sites = neighbor_csites,
                    cart_rotations = self._cartesian_pos
                )
            )

        return clusters

    
    def write_to_cif(self, filename: str) -> None:
        atom = atom_from_cpt_fractional(self.cell, self.positions, self.types)
        write_cif_file(filename, atom)



@dataclasses.dataclass
class CenteredCluster:
    """
    Structure -> [CenteredCluster] -> CenteredEquivalentCluster
    store all neighbor sites around an atom at the center, as well as 
    """
    center_site: CrystalSite
    neighbor_sites: typing.List[CrystalSite]
    cart_rotations: typing.List[np.ndarray]


    @property
    def centered_equivalent_clusters(self) -> typing.List["CenteredEquivalentCluster"]:
        """
        get centered equivalent clusters in order of increasing number of neighbors
        """
        equivalent_clusters = []
        bare_sites = [ csite.site for csite in self.neighbor_sites ]

        group, equivalentset = self._get_siteSym_eqGroup_from_sites_and_rots(
                                    bare_sites, self.cart_rotations
                                )
                
        for indices in equivalentset.values():
            equivalent_clusters.append(
                CenteredEquivalentCluster(
                    center_site = self.center_site, 
                    neighbor_sites = [self.neighbor_sites[i] for i in indices],
                    sitesymmetrygroup = group
                )
            )

        return sorted(
            equivalent_clusters,
            key = lambda nn: len(nn.neighbor_sites)
        )


    def __str__(self) -> str:
        results = [f"CenteredCluster "]
        results.append(f"centered on {self.center_site.site.chemical_symbol} " + \
                   f"(i={self.center_site.index_pcell:>2d})")
        results.append("-"*31)
        for csite in self.neighbor_sites:
            results.append(str(csite))
        results.append("-"*31)
        return "\n".join(results)


    def write_to_cif(self, filename: str) -> None:
        all_csites: typing.List[CrystalSite] = [self.center_site] + self.neighbor_sites
        positions = np.array(
            [s.site.pos for s in all_csites]
        )
        types = [s.site.atomic_number for s in all_csites]

        cell = get_cell_from_origin_centered_positions(positions)
        shift = np.dot(cell.T, np.array([0.5,0.5,0.5]))
        positions += shift
        atom = atom_from_cpt_cartesian(cell, positions, types)
        write_cif_file(filename, atom)


    @staticmethod
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



@dataclasses.dataclass
class CenteredEquivalentCluster:
    """
    Structure -> CenteredCluster -> [CenteredEquivalentCluster]
    """
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
        all_csites: typing.List[CrystalSite] = [self.center_site] + self.neighbor_sites
        positions = np.array(
            [s.site.pos for s in all_csites]
        )
        types = [s.site.atomic_number for s in all_csites]

        cell = get_cell_from_origin_centered_positions(positions)
        shift = np.dot(cell.T, np.array([0.5,0.5,0.5]))
        positions += shift
        atom = atom_from_cpt_cartesian(cell, positions, types)
        write_cif_file(filename, atom)


    def set_symmetry(self, find_additional_symmetry: bool = False) -> None:
        """
        first it checks if the atom is an isolated atom with full spherical symmetry
        if we require to find additional symmetry, it will call the pymatgen method
        """
        vectors = np.vstack([ns.site.pos for ns in self.neighbor_sites])
        if len(vectors) == 1 and np.allclose(
            vectors[0], np.zeros_like(vectors[0]), atol=zero_tolerance
        ):
            self.sitesymmetrygroup = SiteSymmetryGroup.get_spherical_symmetry_group()
            return
        
        if find_additional_symmetry:
            types = [ns.site.chemical_symbol for ns in self.neighbor_sites]
            sym_sch, operations = symOperations_from_pos_types(
                vectors, types
            )
            if sym_sch == "Kh":
                self.sitesymmetrygroup = SiteSymmetryGroup.get_spherical_symmetry_group()
            if sym_sch in GroupsList_sch:
                self.sitesymmetrygroup = SiteSymmetryGroup.from_cartesian_matrices(operations)
