import dataclasses, typing
import numpy as np
from automaticTB.sitesymmetry import SiteSymmetryGroup
from .sites import Site, CrystalSite
from ..parameters import zero_tolerance
from ..atomic_orbitals import OrbitalsList
from ..utilities import Pair, tensor_dot
from ..atomic_orbitals import AO


class ClusterSubSpace(typing.NamedTuple):
    equivalence_type: int
    l: int
    indices: typing.List[int]



class NearestNeighborCluster:

    def __init__(self, 
        crystalsites: typing.List[CrystalSite], 
        orbitalslist: OrbitalsList, 
        sitesymmetrygroup: SiteSymmetryGroup
    ) -> None:
        self.crystalsites = crystalsites
        self.orbitalslist = orbitalslist
        self.sitesymmetrygroup = sitesymmetrygroup
        self._meta = {}

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
        if "equivalent_atoms_dict" in self._meta:
            return self._meta["equivalent_atoms_dict"]
        
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
        self._meta["equivalent_atoms_dict"] = result
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


    @property
    def subspace_pairs(self) -> typing.List[Pair]:
        """
        it return all pairs of subspace, where the left part is always 
        subspace on the center atoms
        """
        if "subspace_pairs" in self._meta:
            return self._meta["subspace_pairs"]

        center_index = self.origin_index
        orbital_slices = self.orbitalslist.subspace_slice_dict
        center_subspace = []
        all_subspaces = []
        for eq_type, each_atom_set in self.equivalent_atoms_dict.items():
            for orb in self.orbitalslist.orbital_list[eq_type].l_list:
                indices = []
                for iatom in each_atom_set:
                    indices += list(range(orbital_slices[iatom][orb].start, orbital_slices[iatom][orb].stop))
                all_subspaces.append(ClusterSubSpace(eq_type, orb, indices))
                if eq_type == center_index:
                    center_subspace.append(ClusterSubSpace(eq_type, orb, indices))

        result = []
        for center in center_subspace:
            for other in all_subspaces:
                result.append(
                    Pair(center, other)
                )
        self._meta["subspace_pairs"] = result
        return result


    @property
    def AOlist(self) -> typing.List[AO]:
        if "AOlist" in self._meta:
            return self._meta["AOlist"]

        result = []
        for icite, csite, orb in zip(range(self.num_sites), self.crystalsites, self.orbitalslist.orbital_list):
            for ao in orb.sh_list:
                result.append(
                    AO(
                        cluster_index = icite,
                        primitive_index = csite.index_pcell, 
                        translation = csite.translation, 
                        chemical_symbol = csite.site.chemical_symbol,
                        l = ao.l,
                        m = ao.m
                    )
                )
        self._meta["AOlist"] = result
        return result

    def get_subspace_AO_Pairs(self, subspace_pair: Pair) -> typing.List[Pair]:
        left: ClusterSubSpace = subspace_pair.left
        right: ClusterSubSpace = subspace_pair.right

        left_aos = [ self.AOlist[i] for i in left.indices ]
        right_aos = [ self.AOlist[i] for i in right.indices ]
        return tensor_dot(left_aos, right_aos)
