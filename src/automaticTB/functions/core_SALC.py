import typing
from ..SALCs import VectorSpace, decompose_vectorspace_to_namedLC, NamedLC
from ..structure import NearestNeighborCluster


__all__ = ["get_namedLCs_from_nncluster"]


def get_namedLCs_from_nncluster(cluster: NearestNeighborCluster) -> typing.List[NamedLC]:
    """
    This is a high level function that take an nearest neighbor cluster as inputs and output 
    a list of all the symmetry adopted linear combination as named LCs. 
    The NamedLC class contains a string for the representation name and a LinearCombination object
    containing all the information of the orbital component. 
    """
    _vectorspace = VectorSpace.from_NNCluster(cluster)
    return decompose_vectorspace_to_namedLC(_vectorspace, cluster.sitesymmetrygroup)
