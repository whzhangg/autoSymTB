from automaticTB.sitesymmetry.Bilbao.group_operations import get_group_operation
from automaticTB.sitesymmetry.group_list import GroupsList

# source https://en.wikipedia.org/wiki/Crystallographic_point_group
group_order = {
    "1": 1, 
    "-1": 2, 
    "2": 2, 
    "m": 2, 
    "2/m": 4, 
    "222": 4, 
    "mm2": 4, 
    "mmm": 8, 
    "4": 4, 
    "-4": 4, 
    "4/m": 8, 
    "422": 8, 
    "4mm": 8, 
    "-42m": 8, 
    "4/mmm": 16, 
    "3": 3, 
    "-3": 6, 
    "32": 6, 
    "3m": 6, 
    "-3m": 12, 
    "6": 6, 
    "-6": 6, 
    "6/m": 12, 
    "622": 12,
    "6mm": 12, 
    "-6m2": 12, 
    "6/mmm": 24, 
    "23": 12, 
    "m-3": 24, 
    "432": 24, 
    "-43m": 24, 
    "m-3m": 48
}

def test_group_order():
    for group in GroupsList:
        seitz_matrix = get_group_operation(group)
        unique_seitz_operation = set(seitz_matrix.keys())
        assert len(unique_seitz_operation) == group_order[group]