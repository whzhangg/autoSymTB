from automaticTB.utilities import load_yaml, print_matrix
from automaticTB.sitesymmetry import subduction_data, SiteSymmetryGroup
#from automaticTB.SALCs import decompose_vectorspace_to_namedLC
from automaticTB.SALCs import LinearCombination
from automaticTB.SALCs.vectorspace import get_nonzero_independent_linear_combinations
from automaticTB.SALCs.decompose_new import decompose_vectorspace_to_namedLC
from automaticTB.SALCs import VectorSpace
import numpy as np
import typing

vs: VectorSpace = load_yaml("T2_vectorspace.yml")
group: SiteSymmetryGroup = load_yaml("-43m.yml")
subduction = subduction_data[group.groupname]
subgroup = group.get_subgroup_from_subgroup_and_seitz_map(
                subduction["sub"], subduction["seitz_map"]
            )



decomposed = decompose_vectorspace_to_namedLC(vs, group)
for nlc in decomposed:
    print(nlc.name)

raise
t2s: typing.Dict[str, typing.List[LinearCombination]] = {}
for nlc in decomposed:
    if nlc.name.main_irrep == "T_2":
        t2s.setdefault(nlc.name.symmetry_symbol,[]).append(nlc.lc)

for lc in t2s["T_2->A_1"]:
    rotated_lcs = []
    for rot in group.operations:
        rotated_lcs.append(lc.symmetry_rotate(rot))
    
    vectorset = get_nonzero_independent_linear_combinations(rotated_lcs)
    T_2_vectorspace = VectorSpace.from_list_of_linear_combinations(vectorset)
    furtherdecompose = decompose_vectorspace_to_namedLC(T_2_vectorspace, subgroup)
    for nlc in furtherdecompose:
        print(nlc.name)
        print(nlc.lc)
    
    print("-----------------------------")

    

for key, lcs in t2s.items():
    print(key)

    nbasis = len(lcs)
    rotation_matrix = np.zeros((nbasis, nbasis))
    for j, jlc in enumerate(lcs):
        for i, ilc in enumerate(lcs):
            dot = np.dot(ilc.coefficients, jlc.coefficients)
            rotation_matrix[i,j] = dot.real
    print_matrix(rotation_matrix)
