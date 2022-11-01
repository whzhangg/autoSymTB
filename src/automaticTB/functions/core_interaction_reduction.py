import typing
from ..interaction import InteractionEquation, CombinedInteractionEquation
from ..SALCs import NamedLC
from ..structure import NearestNeighborCluster
from ..tools import Pair
from ..structure import Structure
from .core_SALC import get_namedLCs_from_nncluster

__all__ = [
    "get_free_AOpairs_from_nncluster_and_namedLCs", 
    "get_overall_free_AOpairs_from_nnclusters_and_namedLCs",
    "get_combined_equation_from_structure"
]

def get_free_AOpairs_from_nncluster_and_namedLCs(
    cluster: NearestNeighborCluster, 
    named_lcs: typing.List[NamedLC]
) -> typing.List[Pair]:
    """
    A high level function that return the free AO pairs
    """
    system = InteractionEquation.from_nncluster_namedLC(cluster, named_lcs)
    return system.free_AOpairs


def get_overall_free_AOpairs_from_nnclusters_and_namedLCs(
    list_of_clusters: typing.List[NearestNeighborCluster],
    list_of_named_lcs: typing.List[typing.List[NamedLC]]
) -> typing.List[Pair]:
    """
    A high level function that return the global free AO pairs
    """
    equations = [
        InteractionEquation.from_nncluster_namedLC(ncluster, nlcs)
        for ncluster, nlcs in zip(list_of_clusters, list_of_named_lcs)
    ]
    combined_systems = CombinedInteractionEquation(equations)
    return combined_systems.free_AOpairs

def get_combined_equation_from_structure(structure: Structure) -> CombinedInteractionEquation:
    """
    this function provide high level utility to get a combined equation from input structure
    """
    equations = []
    for nncluster in structure.nnclusters:
        named_lcs = get_namedLCs_from_nncluster(nncluster)
        equations.append(
            InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs)
        )
    return CombinedInteractionEquation(equations)

