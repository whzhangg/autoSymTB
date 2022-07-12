from automaticTB.sitesymmetry.Bilbao.crawler import Bilbao_PointGroups
from automaticTB.sitesymmetry.group_list import GroupsList

# show that group_list is the same as listed at Bilbao: https://www.cryst.ehu.es/rep/point.html
def test_group_list_symbol_same_as_bilbao():
    bilbao_data = Bilbao_PointGroups()
    groupnames_on_bilbao = bilbao_data.group_list
    assert set(groupnames_on_bilbao) == set(GroupsList)