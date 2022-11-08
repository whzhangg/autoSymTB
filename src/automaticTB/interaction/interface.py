import typing, dataclasses, re, numpy as np
from .interaction_pairs import AO
from ..structure import NeighborCluster, CrystalSite
from ..SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from ..atomic_orbitals import Orbitals, OrbitalsList
from .subspaces import AOSubspace, InteractingAOSubspace

__all__ = ["get_InteractingAOSubspaces_from_cluster"]

def get_InteractingAOSubspaces_from_cluster(
    cluster: NeighborCluster,
    orbital_reference: typing.Dict[str, str]
) -> typing.List[InteractingAOSubspace]:
    
    center_ele = cluster.center_site.site.chemical_symbol
    neighb_ele = cluster.neighbor_sites[0].site.chemical_symbol
    assert center_ele in orbital_reference and neighb_ele in orbital_reference

    center_vectorspace: typing.List[VectorSpace] = []
    center_nls = _get_orbital_ln_from_string(orbital_reference[center_ele])
    for cnl in center_nls:
        center_vectorspace.append(
            VectorSpace.from_sites_and_orbitals(
                [cluster.center_site.site], 
                OrbitalsList(
                    [Orbitals([cnl])]
                )
            )
        )

    neighbor_vectorspace: typing.List[VectorSpace] = []
    neighbor_nls = _get_orbital_ln_from_string(orbital_reference[neighb_ele])
    neighborsites = [csite.site for csite in cluster.neighbor_sites]
    for nnl in neighbor_nls:
        neighbor_vectorspace.append(
            VectorSpace.from_sites_and_orbitals(
                neighborsites, 
                OrbitalsList(
                    [Orbitals([nnl])] * len(neighborsites)
                )
            )
        )
    
    subspaces_pairs = []
    for left in center_vectorspace:
        left_subspace = \
            AOSubspace(
                _get_AO_from_CrystalSites_OrbitalList(
                    [cluster.center_site], left.orbital_list
                ),
                decompose_vectorspace_to_namedLC(left, cluster.sitesymmetrygroup)
            )
            
        for right in neighbor_vectorspace:
            right_subspace = \
                AOSubspace(
                    _get_AO_from_CrystalSites_OrbitalList(
                        cluster.neighbor_sites, right.orbital_list
                    ),
                    decompose_vectorspace_to_namedLC(right, cluster.sitesymmetrygroup)
                )
            subspaces_pairs.append(
                InteractingAOSubspace(left_subspace, right_subspace)
            )

    return subspaces_pairs


def _get_AO_from_CrystalSites_OrbitalList(
    crystalsites: typing.List[CrystalSite],
    orbitallist: OrbitalsList
) -> typing.List[AO]:
    aos = []
    all_sh = orbitallist.sh_list
    i_cluster = 0
    for csite, sh_slice in zip(crystalsites, orbitallist.atomic_slice_dict):
        for sh in all_sh[sh_slice]:
            aos.append(
                AO(
                    cluster_index = i_cluster,
                    primitive_index = csite.index_pcell,
                    absolute_position = csite.absolute_position,
                    translation = csite.translation,
                    chemical_symbol = csite.site.chemical_symbol,
                    n = sh.n,
                    l = sh.l,
                    m = sh.m
                )
            )
        i_cluster += 1

    return aos


def _get_orbital_ln_from_string(orbital: str) -> typing.List[typing.Tuple[int, int]]:
    """
    example "3s3p4s" -> [(3,0),(3,1),(4,0)]
    """
    l_dict = {
        "s": 0, "p": 1, "d": 2, "f": 3
    }
    numbers = re.findall(r"\d", orbital)
    symbols = re.findall(r"[spdf]", orbital)
    if len(numbers) != len(symbols):
        print(f"orbital formula wrong: {orbital}")
    return [
        (int(n),l_dict[lsym]) for n,lsym in zip(numbers, symbols)
    ]