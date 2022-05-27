from automaticTB.utilities import load_yaml, print_matrix
from automaticTB.sitesymmetry import subduction_data, SiteSymmetryGroup
from automaticTB.SALCs.decompose import decompose_vectorspace_to_dict, organize_decomposed_lcs
from automaticTB.SALCs import VectorSpace
import numpy as np

vs: VectorSpace = load_yaml("T2_vectorspace.yml")
group: SiteSymmetryGroup = load_yaml("-43m.yml")

subduction = subduction_data[group.groupname]
subgroup = group.get_subgroup_from_subgroup_and_seitz_map(
                subduction["sub"], subduction["seitz_map"]
            )

lcs = vs.get_nonzero_linear_combinations()

new = decompose_vectorspace_to_dict(vs, group, recursive=False)
for s, vs in new.items():
    print(s)
    for lc in vs.get_nonzero_linear_combinations():
        print(lc)

print("")
lcs = vs.get_nonzero_linear_combinations()
nbasis = len(lcs)
rotation_matrix = np.zeros((nbasis, nbasis))
for j, jlc in enumerate(lcs):
    for i, ilc in enumerate(lcs):
        dot = np.dot(ilc.coefficients, jlc.coefficients)
        rotation_matrix[i,j] = dot.real
print_matrix(rotation_matrix)
