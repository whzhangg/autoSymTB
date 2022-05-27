from vectorspace import VectorSpace
import typing
from linear_combination import LinearCombination
from ..sitesymmetry import SiteSymmetryGroup
from ..parameters import zero_tolerance
import numpy as np

# I think this is not correct
def organize_decomposed_lcs(decomposedLCs: typing.List[LinearCombination], group: SiteSymmetryGroup, rep_dimension: int) \
-> typing.List[typing.List[LinearCombination]]:
    
    identified_index = []
    result = []
    for i, lc in enumerate(decomposedLCs):
        if i in identified_index:
            continue
        found = []
        finished = False
        for rot in group.operations:
            rotated = lc.symmetry_rotate(rot)
            for j, jlc in enumerate(decomposedLCs):
                if j in identified_index: continue
                if np.abs(np.dot(rotated.coefficients, jlc.coefficients)) > zero_tolerance:
                    found.append(jlc)
                    identified_index.append(j)
        result.append(found)
    return result