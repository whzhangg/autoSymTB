"""Module that symmetrize atomic orbital according to symmetry

`VectorSpace` can be created from a set of `CrystalSite`, it can be
decomposed into a set of LinearCombinations each according to a given
symmetry. LinearCombinations can also be initialized by a set of 
`CrystalSite` objects. Finally, the `NamedLC` provide the main output
of this module.

This module take in a set of CrystalSite and their corresponding
orbitals and produce the symmetry adopted linear combinations. It 
includes the decomposition of the 

"""
from .stable_decompose import decompose_vectorspace_to_namedLC
from .vectorspace import VectorSpace
from .linear_combination import *

"""
This module deals with vector and vector space, as well as the projection operation. 
"""