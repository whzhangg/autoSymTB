# this is not a test but to generate the calculated symmetry

from automaticTB.sitesymmetry.SeitzFinder.dress_operation import (
    organize_operation_by_symmetry_direction,
    organize_symmetry_into_equivalent_directions,
    identify_primiary_secondary_tertiary_axes
)
from automaticTB.sitesymmetry.SeitzFinder.symmetry_elements import SymOp_on_Direction, Symmetries_on_Direction_Set
import typing
import numpy as np

from automaticTB.sitesymmetry.group_list import GroupsList
from automaticTB.sitesymmetry.Bilbao.interface import PointGroupData
ptgdata = PointGroupData(complex_character=True)

def get_main_axes(matrices: typing.List[np.ndarray])\
-> typing.List[Symmetries_on_Direction_Set]:
    operations = [ SymOp_on_Direction.from_cartesian_rotation_matrix(m) for m in matrices ]

    listSymOnDirection = organize_operation_by_symmetry_direction(operations)
    result = organize_symmetry_into_equivalent_directions(matrices, listSymOnDirection)
    return result

def print_organized_symmetry_result():
    for group in GroupsList:

        groupdata = ptgdata.get_BilbaoPointGroup(group)
        matrices = np.array([ op.matrix for op in groupdata.operations ])
        
        direction_sets = get_main_axes(matrices)
        organized_axis = identify_primiary_secondary_tertiary_axes(group, direction_sets)

        print("Group{:>6s} ".format(group) + "-" * 59)
        for key, value in organized_axis.items():
            print("{:>12s} : ".format(key) + str(value))
        print("-" * 71)
        print("")

if __name__ == "__main__":
    print_organized_symmetry_result()
        


