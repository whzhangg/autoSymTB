import dataclasses, typing, collections
import numpy as np
from .sites import Site, CrystalSite
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup


@dataclasses.dataclass
class CrystalSites_and_Equivalence:
    crystalsites: typing.List[CrystalSite]
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

    @property
    def origin_index(self) -> int:
        for i,csite in enumerate(self.crystalsites):
            if np.linalg.norm(csite.site.pos) < 1e-6:
                return i
        raise

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

    def get_CrystalSites_and_Equivalence_from_group(self, group: SiteSymmetryGroup) -> CrystalSites_and_Equivalence:
        equivalent_dict = self.get_equivalent_atoms_from_SiteSymmetryGroup(group)
        return CrystalSites_and_Equivalence(
            self.crystalsites, equivalent_dict
        )
