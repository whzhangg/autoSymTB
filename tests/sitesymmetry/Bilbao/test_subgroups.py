from automaticTB.sitesymmetry.Bilbao.extract_group_characters import get_GroupCharacterInfo_from_bilbao
from automaticTB.sitesymmetry.symmetry_reduction import symmetry_reduction
from automaticTB.sitesymmetry.group_list import GroupsList

# test the subgroups stored in file SiteSymmetry/symmetry_reduction.py is indeed
# the ones crawled from Bilbao group page
def test_subgroups_same_as_in_symmetry_reduction():
    for group in GroupsList:
        group_info = get_GroupCharacterInfo_from_bilbao(group, use_complex=False)
        bilbao_subgroups = set(group_info.subgroups)
        stored_subgroups = set(symmetry_reduction[group].keys())
        assert bilbao_subgroups == stored_subgroups