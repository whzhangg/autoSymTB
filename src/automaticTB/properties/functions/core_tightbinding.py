import typing
from ..tightbinding import (
    TightBindingModel, 
    gather_InteractionPairs_into_HijRs, 
    gather_InteractionPairs_into_SijRs,
)
from ...solve.interaction import AOPairWithValue
from ...solve.structure import Structure

__all__ = ["get_tbModel_from_structure_interactions_overlaps"]

def get_tbModel_from_structure_interactions_overlaps(
    structure: Structure, 
    interactions: typing.List[AOPairWithValue], 
    overlaps: typing.Optional[typing.List[AOPairWithValue]] = None
) -> TightBindingModel:
    hijRs = gather_InteractionPairs_into_HijRs(interactions)
    c = structure.cell
    p = structure.positions
    t = structure.types
    if overlaps is None:
        return TightBindingModel(c, p, t, hijRs)
    else:
        sijRs = gather_InteractionPairs_into_SijRs(overlaps)
        return TightBindingModel(c, p, t, hijRs, sijRs)
    