import dataclasses, typing, collections
import numpy as np
from .sites import Site, CrystalSite
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup


@dataclasses.dataclass
class CrystalSitesWithSymmetry:
    crystalsites: typing.List[CrystalSite]
    symmetrygroup: SiteSymmetryGroup
    equivalent_atoms: typing.Dict[int, typing.Set[int]]


@dataclasses.dataclass
class NNCluster:
    crystalsites: typing.List[CrystalSite]
    origin: np.ndarray

    @property
    def num_sites(self) -> int:
        return len(self.crystalsites)

    @property
    def baresites(self) -> typing.List[Site]:
        return [ csite.site for csite in self.crystalsites ]

    def __str__(self):
        result = "cluster center: {:> 6.2f},{:> 6.2f},{:> 6.2f}\n".format(*self.origin)
        for csite in self.crystalsites:
            result += str(csite) + "\n"
        return result

    def get_SiteSymmetryGroup_from_possible_rotations(self, operations: typing.List[np.ndarray]) -> SiteSymmetryGroup:
        symmetries: list = []
        for sym in operations:
            is_symmetry = True
            for site in self.baresites:
                newsite = site.rotate(sym)
                if not newsite in self.baresites: 
                    is_symmetry = False
                    break
            if is_symmetry:
                symmetries.append(sym)
        
        return SiteSymmetryGroup.from_cartesian_matrices(symmetries)

    def get_equivalent_atoms_from_SiteSymmetryGroup(self, group: SiteSymmetryGroup) -> typing.Dict[int, typing.Set[int]]:
        result = {}
        found = []
        for i, site in enumerate(self.baresites):
            if i in found: continue
            for rot in group.operations:
                newsite = site.rotate(rot)
                for j,othersite in enumerate(self.baresites):
                    if newsite == othersite:
                        result.setdefault(i, set()).add(j)
                        found.append(j)
        return result

    def get_CrystalSitesWithSymmetry_from_group(self, group: SiteSymmetryGroup) -> CrystalSitesWithSymmetry:
        equivalent_dict = self.get_equivalent_atoms_from_SiteSymmetryGroup(group)
        return CrystalSitesWithSymmetry(
            self.crystalsites, group, equivalent_dict
        )

'''
@dataclasses.dataclass
class SubspaceName:
    equivalent_index: int
    orbital: str
    irreps: str
    
class NamedVector(typing.NamedTuple):
    name: SubspaceName
    vector: np.ndarray


class NearestNeighborSites:
    def __init__(self, 
        listofsites: typing.List[CrystalSite], 
        origin: np.ndarray, 
        possible_rotations: typing.List[np.ndarray]
    ) -> None:
        self._origin = origin

        self._pindexs = [ site.index_pcell for site in listofsites ]
        self._translations = np.vstack([ site.supercell_shift for site in listofsites ])

        self._baresites = [ site.site for site in listofsites ]
        self._sitesymmetrygroup = SiteSymmetryGroup.from_cartesian_matrices(
            self._find_site_symmetry_operation(possible_rotations)
        )
        self._equivalent = self._find_equivalent_atoms()

    def get_vector_basises(self, orbital_dict: typing.Dict[int, str]) -> typing.List[NamedVector]:
        aolists = []
        for isite, site in enumerate(self._baresites):
            for orb in Orbitals(orbital_dict[site.type]).aolist:
                aolists.append(Atomlm(isite, orb.l, orb.m))
        num_basis = len(aolists)
        reference = {ao:i for i, ao in enumerate(aolists)}

        basises: typing.Dict[str, typing.Dict[str, VectorSpace]] = {}
        for key, atoms in self._equivalent.items():
            atomic_orbit = orbital_dict[self._baresites[key].type]
            for ob in atomic_orbit.split():
                vectorspace = VectorSpace.from_sites(
                    [self._baresites[i] for i in atoms], 
                    ob
                )
                decomposed = decompose_vectorspace(vectorspace, self._sitesymmetrygroup)
                basises[(key,ob)] = decomposed

        named_vectors = []
        for key, value_dict in basises.items():
            equivalent_i = key[0]
            orbital_str = key[1]
            for irrep, vs in value_dict.items():
                name = SubspaceName(equivalent_i, orbital_str, irrep)
                for lc in vs.get_nonzero_linear_combinations():
                    vector = np.zeros(num_basis, dtype=float)
                    for ieq, eq in enumerate(self._equivalent[equivalent_i]):
                        for iao, ao in enumerate(lc.orbitals.aolist):
                            where = reference[Atomlm(eq, ao.l, ao.m)]
                            vector[where] = lc.coefficients[ieq, iao]
                    named_vectors.append(
                        NamedVector(name, vector)
                    ) 
        return named_vectors

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
        found = []
        for baresite in self._baresites:
            rotated_site = baresite.rotate(cartesian_rotation)
            for i, site in enumerate(self._baresites):
                if i in found: continue
                if rotated_site == site:
                    found.append(i)
                    index_of_rotated.append(i)
                    break
                raise ValueError("rotated site cannot be found")
        return index_of_rotated

    def _find_equivalent_atoms(self) -> typing.Dict[int, typing.Set[int]]:
        result = {}
        found = []
        for i, site in enumerate(self._baresites):
            if i in found: continue
            for rot in self._sitesymmetrygroup.operations:
                newsite = site.rotate(rot)
                for j,othersite in enumerate(self._baresites):
                    if newsite == othersite:
                        result.setdefault(i, set()).add(j)
                        found.append(j)
        return result
            

    def _find_site_symmetry_operation(self, operations: typing.List[np.ndarray]) -> typing.List[np.ndarray]:
        result: list = []
        for _, sym in enumerate(operations):
            is_symmetry = True
            for site in self._baresites:
                newsite = site.rotate(sym)
                if not newsite in self._baresites: 
                    is_symmetry = False
                    break
            if is_symmetry:
                result.append(sym)
        return result
'''