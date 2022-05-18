import dataclasses, typing
import numpy as np
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup
from .sites import Site, CrystalSite
from ..parameters import zero_tolerance
from ..orbitals import Orbitals

@dataclasses.dataclass
class NearestNeighborCluster:
    crystalsites: typing.List[CrystalSite]
    orbitalslist: typing.List[Orbitals]
    sitesymmetrygroup: SiteSymmetryGroup

    def __str__(self):
        result = "cluster\n"
        for csite in self.crystalsites:
            result += str(csite) + "\n"
        return result

    @property
    def num_sites(self) -> int:
        return len(self.crystalsites)

    @property
    def origin_index(self) -> int:
        for i,csite in enumerate(self.crystalsites):
            if np.linalg.norm(csite.site.pos) < zero_tolerance:
                return i
        raise
    
    @property
    def baresites(self) -> typing.List[Site]:
        return [ csite.site for csite in self.crystalsites ]

    @property
    def equivalent_atoms_dict(self) -> typing.Dict[int, typing.Set[int]]:
        result = {}
        found = []
        for i, site in enumerate(self.baresites):
            if i in found: continue
            for rot in self.sitesymmetrygroup.operations:
                newsite = site.rotate(rot)
                for j,othersite in enumerate(self.baresites):
                    if newsite == othersite:
                        result.setdefault(i, set()).add(j)
                        found.append(j)
        return result

    @property
    def equivalent_atoms_reference(self) -> typing.List[int]:
        # eg: [0, 0, 0, 0, 4], the first four sites are equivalent to 0
        equivalent_dict = self.equivalent_atoms_dict
        result = [-1] * self.num_sites
        for equivalent_to_which, indices in equivalent_dict.items():
            for i in indices:
                result[i] = equivalent_to_which
        return result

