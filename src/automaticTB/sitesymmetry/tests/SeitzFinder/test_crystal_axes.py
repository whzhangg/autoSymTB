from automaticTB.sitesymmetry.SeitzFinder.dress_operation import (
    organize_operation_by_symmetry_direction,
    organize_symmetry_into_equivalent_directions
)
from automaticTB.sitesymmetry.SeitzFinder.symmetry_elements import SymOp_on_Direction, Symmetries_on_Direction_Set
from automaticTB.sitesymmetry.utilities import random_rotation_matrix, rotation_fraction_to_cartesian
import typing
import numpy as np

from automaticTB.sitesymmetry.SeitzFinder.crystal_axes import CrystalAxes
from automaticTB.sitesymmetry.group_list import GroupsList
from automaticTB.sitesymmetry.Bilbao.interface import PointGroupData
ptgdata = PointGroupData(complex_character=True)

def get_main_axes(matrices: typing.List[np.ndarray])\
-> typing.List[Symmetries_on_Direction_Set]:
    operations = [ SymOp_on_Direction.from_cartesian_rotation_matrix(m) for m in matrices ]

    listSymOnDirection = organize_operation_by_symmetry_direction(operations)
    return organize_symmetry_into_equivalent_directions(matrices, listSymOnDirection)

def random_rotated_group_symmetry_operations(groupname: str) -> np.ndarray:
    groupdata = ptgdata.get_BilbaoPointGroup(groupname)
    matrices = np.array([ op.matrix for op in groupdata.operations ])
    rot = random_rotation_matrix()
    return rotation_fraction_to_cartesian(matrices, rot.T)


def test_calculated_symmetry_direction_equal_tabulated():
    for group in GroupsList:
        matrices_in_rotated_system = random_rotated_group_symmetry_operations(group)
        direction_sets = get_main_axes(matrices_in_rotated_system)

        calculated_ops = [ direction.op_set for direction in direction_sets ]
        tabulated_ops = list(CrystalAxes[group].values())
        for c_ops in calculated_ops:
            for i,t_ops in enumerate(tabulated_ops):
                if c_ops == t_ops:
                    tabulated_ops.pop(i)
                    break

        assert len(tabulated_ops) == 0
        assert len(direction_sets) - 1 == len(CrystalAxes[group].keys())


