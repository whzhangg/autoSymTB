import typing
from ..interaction import (
    get_InteractingAOSubspaces_from_cluster, CombinedAOSubspaceInteraction
)
from ..structure import NeighborCluster
from ..structure import Structure

__all__ = [
    "get_equationsystem_from_nncluster", 
    "get_free_AOpairs_from_structure",
]


def get_equationsystem_from_nncluster(cluster: NeighborCluster) \
-> CombinedAOSubspaceInteraction:
    """
    A high level function that return the free AO pairs
    """
    iaosubspaces = get_InteractingAOSubspaces_from_cluster(cluster)
    return CombinedAOSubspaceInteraction(iaosubspaces)


def get_free_AOpairs_from_structure(structure: Structure) \
-> CombinedAOSubspaceInteraction:
    all_iao = []
    for cluster in structure.nnclusters:
        all_iao += get_InteractingAOSubspaces_from_cluster(cluster)
    return CombinedAOSubspaceInteraction(all_iao)
