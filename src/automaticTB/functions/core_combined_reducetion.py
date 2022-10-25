from ..structure import Structure
from ..interaction import CombinedInteractionEquation, InteractionEquation
from .core_SALC import get_namedLCs_from_nncluster


__all__ = ["get_combined_equation_from_structure"]


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
