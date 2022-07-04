from automaticTB.sitesymmetry.Bilbao.interface import PointGroupData
from automaticTB.sitesymmetry.group_list import GroupsList

# interface package just organize the results together, if the 
# method it depend on is OK, this should be OK
def test_PointGroupData_complex():
    pointgroups = PointGroupData(complex_character=True)
    for group in GroupsList:
        pointgroups.get_BilbaoPointGroup(group)
        
def test_PointGroupData_real():
    pointgroups = PointGroupData(complex_character=False)
    for group in GroupsList:
        pointgroups.get_BilbaoPointGroup(group)