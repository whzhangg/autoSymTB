import automaticTB.sitesymmetry.Bilbao.crawler as crawler
from automaticTB.sitesymmetry.group_list import GroupsList

def test_getsoup_for_characterTable():
    local_pointgroup_data = crawler.Bilbao_PointGroups()
    for group in GroupsList:
        local_pointgroup_data.get_group_main_page_as_soup(group)
    
def test_getsoup_for_group_operations():
    local_pointgroup_data = crawler.Bilbao_PointGroups()
    for group in GroupsList:
        local_pointgroup_data.get_group_operation_page_as_soup(group)


