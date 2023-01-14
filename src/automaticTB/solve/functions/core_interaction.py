from ..interaction import (
    get_InteractingAOSubspaces_from_cluster, CombinedAOSubspaceInteraction
)
from ..structure import Structure, CenteredCluster, CenteredEquivalentCluster

def get_EquationSystem_from_CenteredEquivalentCluster(
    centered_equivalent_cluster: CenteredEquivalentCluster,
    find_additional_symmetry: bool
) -> CombinedAOSubspaceInteraction:
    """get equation system for centered equation"""
    centered_equivalent_cluster.set_symmetry(find_additional_symmetry=find_additional_symmetry)
    iaosubspaces = get_InteractingAOSubspaces_from_cluster(centered_equivalent_cluster)
    return CombinedAOSubspaceInteraction(iaosubspaces.subspaceslist)

def get_EquationSystem_from_CenteredCluster(
    centered_cluster: CenteredCluster,
    find_additional_symmetry: bool
) -> CombinedAOSubspaceInteraction:
    """high level function that obtain the combined equation for centered cluster"""
    all_iao = []
    for ceq_cluster in centered_cluster.centered_equivalent_clusters:
        ceq_cluster.set_symmetry(find_additional_symmetry = find_additional_symmetry)
        all_iao += get_InteractingAOSubspaces_from_cluster(ceq_cluster).subspaceslist
    return CombinedAOSubspaceInteraction(all_iao)

def get_EquationSystem_from_Structure(
    structure: Structure,
    find_additional_symmetry: bool
) -> CombinedAOSubspaceInteraction:
    """for crystal structure"""
    all_iao = []
    for centered_cluster in structure.centered_clusters:
        for ceq_cluster in centered_cluster.centered_equivalent_clusters:
            ceq_cluster.set_symmetry(find_additional_symmetry = find_additional_symmetry)
            all_iao += get_InteractingAOSubspaces_from_cluster(ceq_cluster).subspaceslist
    return CombinedAOSubspaceInteraction(all_iao)
