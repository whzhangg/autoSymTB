import typing, re

from automaticTB.solve.structure import CrystalSite, CenteredEquivalentCluster, CenteredCluster
from automaticTB.solve.SALCs import VectorSpace, decompose_vectorspace_to_namedLC, NamedLC
from automaticTB.solve.atomic_orbitals import Orbitals, OrbitalsList
from .interaction_pairs import AO, AOPair, AOSubspace
from .interaction_base import InteractingAOSubspace


def get_InteractingAOSubspaces_from_cluster(
    cluster: CenteredEquivalentCluster
) -> typing.List[InteractingAOSubspace]:
    """solved a centered equivalent cluster into a set of interacting subspaces"""

    center_vectorspace: typing.List[VectorSpace] = []
    center_namedLCs: typing.List[typing.List[NamedLC]] = []
    center_nls = get_orbital_ln_from_string(cluster.center_site.orbitals)
    for cnl in center_nls:
        vs = VectorSpace.from_sites_and_orbitals(
                [cluster.center_site.site], 
                OrbitalsList(
                    [Orbitals([cnl])]
                )
            )
        center_vectorspace.append(vs)
        center_namedLCs.append(
            decompose_vectorspace_to_namedLC(vs, cluster.sitesymmetrygroup)
        )

    neighbor_vectorspace: typing.List[VectorSpace] = []
    neighbor_namedLCs: typing.List[typing.List[NamedLC]] = []
    neighbor_nls = get_orbital_ln_from_string(cluster.neighbor_sites[0].orbitals)
    neighborsites = [csite.site for csite in cluster.neighbor_sites]
    for nnl in neighbor_nls:
        vs = VectorSpace.from_sites_and_orbitals(
                neighborsites, 
                OrbitalsList(
                    [Orbitals([nnl])] * len(neighborsites)
                )
            )
        neighbor_vectorspace.append(vs)
        neighbor_namedLCs.append(
            decompose_vectorspace_to_namedLC(vs, cluster.sitesymmetrygroup)
        )
    
    subspaces_pairs = []
    for left_vs, left_nlc in zip(center_vectorspace, center_namedLCs):
        left_subspace = \
            AOSubspace(
                get_AO_from_CrystalSites_OrbitalList(
                    [cluster.center_site], left_vs.orbital_list
                ),
                left_nlc
            )
            
        for right_vs, right_nlc in zip(neighbor_vectorspace, neighbor_namedLCs):
            right_subspace = \
                AOSubspace(
                    get_AO_from_CrystalSites_OrbitalList(
                        cluster.neighbor_sites, right_vs.orbital_list
                    ),
                    right_nlc
                )
            subspaces_pairs.append(
                InteractingAOSubspace(left_subspace, right_subspace)
            )

    return subspaces_pairs


def get_AO_from_CrystalSites_OrbitalList(
    crystalsites: typing.List[CrystalSite],orbitallist: OrbitalsList
) -> typing.List[AO]:
    """given a list of atomic position and orbitals, convernt them to a list of AO"""

    aos = []
    all_sh = orbitallist.sh_list
    i_cluster = 0
    for csite, sh_slice in zip(crystalsites, orbitallist.atomic_slice_dict):
        for sh in all_sh[sh_slice]:
            aos.append(
                AO(
                    equivalent_index = csite.equivalent_index,
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


def get_orbital_ln_from_string(orbital: str) -> typing.List[typing.Tuple[int, int]]:
    """convert string to dict: example "3s3p4s" -> [(3,0),(3,1),(4,0)]"""
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


def generate_aopair_from_cluster(cluster: CenteredCluster) -> typing.List[AOPair]:
    """generate all AOPair on the given centeredCluster"""
    all_aopairs_on_cluster = []
    center_nls = get_orbital_ln_from_string(cluster.center_site.orbitals)
    center_vs = VectorSpace.from_sites_and_orbitals(
                [cluster.center_site.site], OrbitalsList([Orbitals(center_nls)])
            )

    neighbor_nls = get_orbital_ln_from_string(cluster.neighbor_sites[0].orbitals)
    neighbor_sites = \
                [cluster.center_site.site] + [csite.site for csite in cluster.neighbor_sites]
    neighbor_vs = VectorSpace.from_sites_and_orbitals(
                    neighbor_sites, 
                    OrbitalsList([Orbitals(center_nls)] + [Orbitals(neighbor_nls)]*len(neighbor_sites))
                )

    center_aos = get_AO_from_CrystalSites_OrbitalList(
                    [cluster.center_site], center_vs.orbital_list
                )
    neighbor_aos = get_AO_from_CrystalSites_OrbitalList(
                    [cluster.center_site] + cluster.neighbor_sites, neighbor_vs.orbital_list
                ) 

    for cao in center_aos:
        for nao in neighbor_aos:
            pair = AOPair(cao, nao)
            all_aopairs_on_cluster.append(pair)

    return all_aopairs_on_cluster 
