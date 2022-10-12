import typing
from ..interaction import InteractionEquation
from ..SALCs import NamedLC
from ..structure import NearestNeighborCluster
from ..tools import Pair

__all__ = [
    "get_free_AOpairs_from_nncluster_and_namedLCs", 
    "get_reduced_pairs"
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


def get_reduced_pairs(ao_pairs: typing.List[Pair]) -> typing.List[Pair]:
    """
    Provide some minor clean-ups for the final list of free pair of AOs. Notice however that 
    the equivalence of these two pairs are not guaranteed by the results from irreducible
    representations, therefore all pairs provided by freeAOpairs are necessary if we want to 
    recover the final Hamiltonian. 

    For example: the pairs that are removed here is:
        Si 1s -> Si 2s @ (  0.00,  0.00,  0.00)
        Si 2s -> Si 1s @ (  0.00,  0.00,  0.00)
    """
    known_pairs = []
    for current_pair in ao_pairs:
        if len(known_pairs) == 0:
            known_pairs.append(current_pair)
        else:
            already_known = False
            for known_pair in known_pairs:
                if ( current_pair.left == known_pair.left ) \
                and ( current_pair.right == known_pair.right ):
                    already_known = True
                    break
                elif ( current_pair.left == known_pair.right ) \
                and ( current_pair.right == known_pair.left ):
                    already_known = True
                    break
            
            if not already_known:
                known_pairs.append(current_pair)

    return known_pairs    

