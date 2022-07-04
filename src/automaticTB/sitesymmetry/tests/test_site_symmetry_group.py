from automaticTB.sitesymmetry.utilities import random_rotation_matrix, rotation_fraction_to_cartesian
import numpy as np
from automaticTB.sitesymmetry.group_list import GroupsList
from automaticTB.sitesymmetry.Bilbao.interface import PointGroupData
from automaticTB.sitesymmetry.site_symmetry_group import SiteSymmetryGroup

ptgdata = PointGroupData(complex_character=True)

    
def random_rotated_group_symmetry_operations(groupname) -> np.ndarray:
    groupdata = ptgdata.get_BilbaoPointGroup(groupname)
    matrices = np.array([ op.matrix for op in groupdata.operations ])
    rot = random_rotation_matrix()
    return rotation_fraction_to_cartesian(matrices, rot.T)


def test_generating_sitesymmetrygroup_and_subgroups():
    for group in GroupsList:
        matrices_in_rotated_system = random_rotated_group_symmetry_operations(group)
        sitesymmetry = SiteSymmetryGroup.from_cartesian_matrices(matrices_in_rotated_system)
        for subgroup in sitesymmetry.subgroups:
            subgroup_site_symmetry = sitesymmetry.get_subgroup(subgroup)
