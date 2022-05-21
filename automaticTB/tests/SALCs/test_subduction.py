from automaticTB.SALCs.subduction import subduction_data
from automaticTB.sitesymmetry.group_list import GroupsList

def test_subduction_operation_is_probably_correct():
    from automaticTB.sitesymmetry.Bilbao.interface import PointGroupData
    pgd = PointGroupData(complex_character=False)

    for group in GroupsList:
        subduction_information = subduction_data[group]
        subgroup = subduction_information["sub"]
        
        group_data = pgd.get_BilbaoPointGroup(group)
        subgroup_data = pgd.get_BilbaoPointGroup(subgroup)

        map = subduction_information["seitz_map"]
        for sub_seitz, super_seitz in map.items():
            assert sub_seitz in subgroup_data.seitz_operation_dict, f"{group} -> {subgroup}"
            assert super_seitz in group_data.seitz_operation_dict, f"{group} -> {subgroup}"

def test_generate_subduction():
    from automaticTB.sitesymmetry.site_symmetry_group import get_point_group_as_SiteSymmetryGroup
    for group in GroupsList:
        sitesymmetrygroup = get_point_group_as_SiteSymmetryGroup(group)
        subgroup_info = subduction_data[group]
        subgroup_sitesymmetry = sitesymmetrygroup.get_subgroup_from_subgroup_and_seitz_map(
            subgroup_info["sub"],
            subgroup_info["seitz_map"]
        )
        assert subgroup_sitesymmetry.groupname == subgroup_info["sub"]