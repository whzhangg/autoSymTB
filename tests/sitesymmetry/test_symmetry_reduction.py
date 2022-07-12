from automaticTB.sitesymmetry.symmetry_reduction import symmetry_reduction
from automaticTB.sitesymmetry.Bilbao.interface import PointGroupData

def test_symmetry_reduction_operations_are_valid():
    pointgroup_data = PointGroupData(complex_character=True)
    for main_group_name, subgroup_dict in symmetry_reduction.items():
        supergroup = pointgroup_data.get_BilbaoPointGroup(main_group_name)
        for subgroup_name, mapper in subgroup_dict.items():
            subgroup = pointgroup_data.get_BilbaoPointGroup(subgroup_name)
            for seitz_in_sub, seitz_in_main in mapper.items():
                assert seitz_in_sub in [ op.seitz for op in subgroup.operations ], " ".join([main_group_name, subgroup_name])
                assert seitz_in_main in [ op.seitz for op in supergroup.operations ], " ".join([main_group_name, subgroup_name])
