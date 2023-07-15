import typing
import dataclasses

import numpy as np
import spglib
from pymatgen.core import structure as matgenStructure
from pymatgen.analysis import local_env

from automaticTB import tools
from automaticTB import parameters as params
from automaticTB.solve import sitesymmetry as ssym
from .sites import LocalSite, CrystalSite


def _get_absolute_pos_from_cell_fractional_pos_translation(
    cell: np.ndarray, x: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """return the absolute position using fractional position and translation vector"""
    return cell.T.dot(x + t)


class Structure:
    """
    [Structure] -> CenteredCluster -> CenteredEquivalentCluster
    """
    voronoi_rcut: float = 5.0
    voronoi_weight_cutoff: float = 1e-1

    @staticmethod
    def get_standarized_cpt(
        cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int], 
        standardize: bool = True
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        get structure, with position rounded and floored so that they are strictly
        within [0, 1), although note that 0.9999999999999 can happen due to numerical precision
        """
        if standardize:
            pcell = spglib.find_primitive((cell, positions, types), symprec=params.spgtol)
            c, p, t = pcell
        else:
            c, p, t = cell, positions, types
        round_decimal = int(-1 * np.log10(params.stol*1e-2))
        rounded = np.around(p, decimals=round_decimal)
        shifted = rounded - np.floor(rounded)
        return c, shifted, np.array(t, dtype = int)


    @staticmethod
    def get_unique_rotations(
        rotations: typing.List[np.ndarray]
    ) -> np.ndarray:
        """
        this is necessary because for unit cell, althought the symmetry is the same, there are 
        additional symmetery operation because of the change in the lattice basis.
        """
        unique = []
        for r in rotations:
            found = False

            for r_u in unique:
                if np.allclose(r, r_u, atol=params.stol): 
                    found = True
                    break
            if found: continue
            unique.append(r)
        
        return np.array(unique)


    def __init__(self, 
        cell: np.ndarray, positions: typing.List[np.ndarray], types: typing.List[int],
        atomic_orbital_dict: typing.Dict[str, str], 
        rcut: typing.Optional[float] = None,
        standardize: bool = True
    ) -> None:
        """
        cell, positions, types: (c, p, t) that describes the entire crystal structure
        atomic_orbital_dict: a dictionary specifying the orbitals on each atoms
        rcut: radius cutoff for identifying neighbors, if not specified, use voronoi method
        standardize: standard to primitive cell if specified, otherwise will use the cell as given
        """
        self.cell, self.positions, self.types = \
            self.get_standarized_cpt(cell, positions, types, standardize)


        self._cartesian_pos = np.einsum("ji, kj -> ki", self.cell, self.positions)
        self._atomic_orbital_dict = atomic_orbital_dict
        self._rcut = rcut

        self.sym_data = spglib.get_symmetry_dataset(
            (self.cell, self.positions, self.types), symprec = params.spgtol
        )
        self._cartesian_rotations = tools.rotation_fraction_to_cartesian(
            self.get_unique_rotations(self.sym_data["rotations"]), 
            self.cell
        )
        self._eq_indices = self.sym_data["equivalent_atoms"]

    @property
    def cartesian_rotations(self) -> np.ndarray:
        return self._cartesian_rotations

    @property
    def centered_clusters(self) -> typing.List["CenteredCluster"]:
        if self._rcut is not None:
            return self._get_centered_cluster_rcut()
        else:
            return self._get_centered_cluster_voronoi()

    @property
    def equivalent_clusters(self) -> typing.Dict[int, typing.List["CenteredCluster"]]:
        centered_clusters = self.centered_clusters

        equivalent_clusters = {}
        for clusters, eqi in zip(centered_clusters, self._eq_indices):
            equivalent_clusters.setdefault(eqi, []).append(clusters)

        return equivalent_clusters

    def _get_centered_cluster_rcut(self) -> typing.List["CenteredCluster"]:
        """
        get CenteredCluster using rcut method
        """
        c, p, t = self.cell, self.positions, self.types

        pymatgen_structure = matgenStructure.Structure(lattice = c, species = t, coords = p)
        clusters = []
        for _, cpos in enumerate(self._cartesian_pos):
            sites = pymatgen_structure.get_sites_in_sphere(cpos, self._rcut)
            
            neighbor_csites: typing.List[CrystalSite] = []
            for site in sites:
                relative_cartesian = np.array([site.x, site.y, site.z]) - cpos
                _localsite = LocalSite(t[site.index], relative_cartesian)
                tr = np.array(site.image)
                _pos = _get_absolute_pos_from_cell_fractional_pos_translation(
                    c, p[site.index], tr
                )
                _crystalsite = \
                    CrystalSite(
                        site = _localsite,
                        absolute_position = _pos,
                        index_pcell = site.index, 
                        equivalent_index = self._eq_indices[site.index],
                        translation = tr,
                        orbitals = self._atomic_orbital_dict[_localsite.chemical_symbol]
                    )

                if np.linalg.norm(relative_cartesian) < params.ztol:
                    center_csite = _crystalsite
                else:
                    neighbor_csites.append(_crystalsite)
            
            clusters.append(
                CenteredCluster(
                    center_site = center_csite,
                    neighbor_sites = neighbor_csites,
                    cart_rotations = self._cartesian_rotations
                )
            )
        
        return clusters


    def _get_centered_cluster_voronoi(self) -> typing.List["CenteredCluster"]:
        """
        get CenteredCluster using voronoi method
        """
        c, p, t = self.cell, self.positions, self.types
        pymatgen_structure = matgenStructure.Structure(lattice = c, species = t, coords = p)
        neighborNN = local_env.VoronoiNN(cutoff = self.voronoi_rcut, compute_adj_neighbors = False)
        
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
                _pos = _get_absolute_pos_from_cell_fractional_pos_translation(c, p[index], tr)
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
                    cart_rotations = self._cartesian_rotations
                )
            )

        return clusters

    
    def write_to_cif(self, filename: str) -> None:
        atom = tools.atom_from_cpt_fractional(self.cell, self.positions, self.types)
        tools.write_cif_file(filename, atom)


    def print_log(self) -> None:
        print(f"## Crystal structure used in the calculation")
        print("-"*75)
        print(f"### SPGlib symmetry info (tolerance = {params.stol:>.2e}):")
        print( "  space group: {:s} {:s} ({:0>d})".format(
                self.sym_data['international'],
                self.sym_data['choice'], 
                self.sym_data['number']
            ))
        print(f"  point group: {self.sym_data['pointgroup']}")
        print(f"  number of rotations: {len(self._cartesian_rotations)}")
        print("")
        print(f"### Cell parameters:")
        print(f"  a = {self.cell[0,0]:>12.6f},{self.cell[0,1]:>12.6f},{self.cell[0,2]:>12.6f}")
        print(f"  b = {self.cell[1,0]:>12.6f},{self.cell[1,1]:>12.6f},{self.cell[1,2]:>12.6f}")
        print(f"  c = {self.cell[2,0]:>12.6f},{self.cell[2,1]:>12.6f},{self.cell[2,2]:>12.6f}")
        print("")
        print(f"### Atoms in the cell:")
        natom = len(self.positions)
        for i, pos, type in zip(range(natom), self.positions, self.types):
            symbol = tools.chemical_symbols[type]
            print("  {:0>d} {:>2s} : ({:>10.6f},{:>10.6f},{:>10.6f}) eq_index = {:d}, orb = {:s}".format(
                i+1, symbol, *pos, self._eq_indices[i] + 1, self._atomic_orbital_dict[symbol]
            ))
        print("-"*75)


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
        bare_sites = [ csite.site for csite in self.neighbor_sites ]

        group, equivalentset = self._get_siteSym_eqGroup_from_sites_and_rots(
                                    bare_sites, self.cart_rotations
                                )
        
        equivalent_clusters = [
            CenteredEquivalentCluster(
                    center_site = self.center_site, 
                    neighbor_sites = [self.center_site],
                    sitesymmetrygroup = group
                )
        ] # add the self-interaction for the center atom
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
        results = [f"Centered Cluster "]
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

        cell = tools.get_cell_from_origin_centered_positions(positions)
        shift = np.dot(cell.T, np.array([0.5,0.5,0.5]))
        positions += shift
        atom = tools.atom_from_cpt_cartesian(cell, positions, types)
        tools.write_cif_file(filename, atom)


    def print_log(self) -> None:
        print("## Cluster centered on {:>2s} ({:>d})".format(
            self.center_site.site.chemical_symbol, self.center_site.index_pcell+1
        ))

        distances = {}
        for neisite in self.neighbor_sites:
            dist = f"{np.linalg.norm(neisite.site.pos):>.2f}"
            distances.setdefault(dist, []).append(neisite.site.chemical_symbol)
        sorted_dist = sorted(distances.keys(), key=float)
        distance_string = ""
        for sd in sorted_dist:
            distance_string += f"[{sd}A]-> (" + ",".join(distances[sd]) + ") "
        print("  "+distance_string)

    @property
    def sitesymgroup(self) -> ssym.SiteSymmetryGroup:
        baresites = [ csite.site for csite in self.neighbor_sites ]
        symmetries: list = []
        for sym in self.cart_rotations:
            is_symmetry = True
            for site in baresites:
                newsite = site.rotate(sym)
                if not newsite in baresites: 
                    is_symmetry = False
                    break
            if is_symmetry:
                symmetries.append(sym)
        return ssym.SiteSymmetryGroup.from_cartesian_matrices(symmetries)

    @staticmethod
    def _get_siteSym_eqGroup_from_sites_and_rots(
        baresites: typing.List[LocalSite], operations: typing.List[np.ndarray]
    ) -> typing.Tuple[ssym.SiteSymmetryGroup, typing.Dict[int, typing.List[int]]]:
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

        group = ssym.SiteSymmetryGroup.from_cartesian_matrices(symmetries)

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
    sitesymmetrygroup: ssym.SiteSymmetryGroup

    @property
    def distance(self) -> float:
        vectors = np.vstack([ ns.site.pos for ns in self.neighbor_sites ])
        distances = np.linalg.norm(vectors, axis=1)
        assert np.allclose(distances, np.min(distances), params.ztol)
        return np.min(distances)


    def __str__(self) -> str:
        results = [f"Centered Equiv. Cluster ({self.sitesymmetrygroup.groupname:>5s}) "]
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

        cell = tools.get_cell_from_origin_centered_positions(positions)
        shift = np.dot(cell.T, np.array([0.5,0.5,0.5]))
        positions += shift
        atom = tools.atom_from_cpt_cartesian(cell, positions, types)
        tools.write_cif_file(filename, atom)


    def set_symmetry(self, 
        find_additional_symmetry: bool = False,
        degenerate_atomic_orbital: bool = True
    ) -> None:
        """
        first it checks if the atom is an isolated atom with full spherical symmetry
        if we require to find additional symmetry, it will call the pymatgen method
        """
        from .pymatgen_sym import symOperations_from_pos_types
        vectors = np.vstack([ns.site.pos for ns in self.neighbor_sites])
        if (degenerate_atomic_orbital and 
            len(vectors) == 1 and 
            np.allclose(vectors[0], np.zeros_like(vectors[0]), atol=params.ztol)
        ):
            self.sitesymmetrygroup = ssym.SiteSymmetryGroup.get_spherical_symmetry_group()
            return
        
        if find_additional_symmetry:
            types = [ns.site.chemical_symbol for ns in self.neighbor_sites]
            sym_sch, operations = symOperations_from_pos_types(
                vectors, types
            )
            if sym_sch == "Kh":
                self.sitesymmetrygroup = ssym.SiteSymmetryGroup.get_spherical_symmetry_group()
            if sym_sch in ssym.GroupsList_sch:
                self.sitesymmetrygroup = ssym.SiteSymmetryGroup.from_cartesian_matrices(operations)

    def print_log(self) -> None:
        if self.distance < 1e-5:
            print(
                f"## Centered Equiv. Cluster ({self.sitesymmetrygroup.groupname:>5s})" +
                f" on {self.center_site.site.chemical_symbol}"
                f" ({self.center_site.index_pcell+1:>2d}) with itself"
            )
        else:
            print(
                f"## Centered Equiv. Cluster ({self.sitesymmetrygroup.groupname:>5s})" +
                f" on {self.center_site.site.chemical_symbol}" + 
                f" ({self.center_site.index_pcell+1:>2d})" +
                f" -> Distance : {self.distance:>.4f}A to" + 
                f" {self.neighbor_sites[0].site.chemical_symbol}"
            )
            print("---")
            for csite in self.neighbor_sites:
                print("  " + str(csite))
            print("---")
            