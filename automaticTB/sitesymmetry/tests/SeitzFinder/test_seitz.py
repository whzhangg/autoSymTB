from automaticTB.sitesymmetry.SeitzFinder.dress_operation import (
    dress_symmetry_operation
)
from automaticTB.sitesymmetry.utilities import random_rotation_matrix, rotation_fraction_to_cartesian
import numpy as np

from automaticTB.sitesymmetry.group_list import GroupsList
from automaticTB.sitesymmetry.Bilbao.interface import PointGroupData

ptgdata = PointGroupData(complex_character=True)

    
def random_rotated_group_symmetry_operations(groupname) -> np.ndarray:
    groupdata = ptgdata.get_BilbaoPointGroup(groupname)
    matrices = np.array([ op.matrix for op in groupdata.operations ])
    rot = random_rotation_matrix()
    return rotation_fraction_to_cartesian(matrices, rot.T)


def test_calculated_symmetry_direction_equal_tabulated():
    for group in GroupsList:
        matrices_in_rotated_system = random_rotated_group_symmetry_operations(group)
        dressed_operation = dress_symmetry_operation(matrices_in_rotated_system)
        seitz_calculated = set(dressed_operation.keys())
        seitz_bilbao = set(ptgdata.get_BilbaoPointGroup(group).seitz_operation_dict.keys())

        assert seitz_calculated == seitz_bilbao

def test_sense_of_rotoinversion():
    # this test shows that the sense of rotoinversion operation, such as -3+, is equal to 
    # the sense of its rotation only part
    group = "-3"
    seitz_bilbao = ptgdata.get_BilbaoPointGroup(group).seitz_operation_dict

    rotoinversion = np.matmul(
        seitz_bilbao["3 001 +"].matrix, seitz_bilbao["-1 000"].matrix
    )

    rotoinversion_symbol = ""
    for seitz, op in seitz_bilbao.items():
        if np.allclose(rotoinversion, op.matrix):
            rotoinversion_symbol = seitz
    
    assert "+" in rotoinversion_symbol

def test_multiplication_table():
    # this test shows that the seitz symbol assignment is correct.
    from automaticTB.sitesymmetry.SeitzFinder.mulitplication_table import get_multiplication_table
    for group in GroupsList:
        if group != "-42m": continue
        matrices_in_rotated_system = random_rotated_group_symmetry_operations(group)
        dressed_operation = dress_symmetry_operation(matrices_in_rotated_system)
        table_calculated = get_multiplication_table(dressed_operation)

        seitz_bilbao = {
            key:value.matrix for key, value in ptgdata.get_BilbaoPointGroup(group).seitz_operation_dict.items()
        }
        
        table_bilbao = get_multiplication_table(seitz_bilbao)
        assert table_bilbao == table_calculated

