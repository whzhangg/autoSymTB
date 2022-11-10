import typing
from ..interaction import (
    get_InteractingAOSubspaces_from_cluster, CombinedAOSubspaceInteraction
)
from ..structure import NeighborCluster
from ..structure import Structure

def get_equationsystem_from_nncluster_orbital_dict(
    cluster: NeighborCluster, orbital_dict: typing.Dict[str, str]
) -> CombinedAOSubspaceInteraction:
    """
    A high level function that return the free AO pairs
    """
    iaosubspaces = get_InteractingAOSubspaces_from_cluster(cluster, orbital_dict)
    return CombinedAOSubspaceInteraction(iaosubspaces)


def get_equationsystem_from_structure_orbital_dict(
    structure: Structure, orbital_dict: typing.Dict[str, str]
) -> CombinedAOSubspaceInteraction:
    all_iao = []
    for cluster in structure.nnclusters:
        cluster.find_and_set_additional_symmetry()
        all_iao += get_InteractingAOSubspaces_from_cluster(cluster, orbital_dict)
    return CombinedAOSubspaceInteraction(all_iao)
