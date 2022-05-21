from automaticTB.sitesymmetry.Bilbao.extract_group_characters import get_GroupCharacterInfo_from_bilbao
from automaticTB.sitesymmetry.group_list import GroupsList

nrep_standard = {
    "1": {
        "nrep_real": 1,
        "nrep_comp": 1
    },
    "-1": {
        "nrep_real": 2,
        "nrep_comp": 2
    },
    "2": {
        "nrep_real": 2,
        "nrep_comp": 2
    },
    "m": {
        "nrep_real": 2,
        "nrep_comp": 2
    },
    "2/m": {
        "nrep_real": 4,
        "nrep_comp": 4
    },
    "222": {
        "nrep_real": 4,
        "nrep_comp": 4
    },
    "mm2": {
        "nrep_real": 4,
        "nrep_comp": 4
    },
    "mmm": {
        "nrep_real": 8,
        "nrep_comp": 8
    },
    "4": {
        "nrep_real": 3,
        "nrep_comp": 4
    },
    "-4": {
        "nrep_real": 3,
        "nrep_comp": 4
    },
    "4/m": {
        "nrep_real": 6,
        "nrep_comp": 8
    },
    "422": {
        "nrep_real": 5,
        "nrep_comp": 5
    },
    "4mm": {
        "nrep_real": 5,
        "nrep_comp": 5
    },
    "-42m": {
        "nrep_real": 5,
        "nrep_comp": 5
    },
    "4/mmm": {
        "nrep_real": 10,
        "nrep_comp": 10
    },
    "3": {
        "nrep_real": 2,
        "nrep_comp": 3
    },
    "-3": {
        "nrep_real": 4,
        "nrep_comp": 6
    },
    "32": {
        "nrep_real": 3,
        "nrep_comp": 3
    },
    "3m": {
        "nrep_real": 3,
        "nrep_comp": 3
    },
    "-3m": {
        "nrep_real": 6,
        "nrep_comp": 6
    },
    "6": {
        "nrep_real": 4,
        "nrep_comp": 6
    },
    "-6": {
        "nrep_real": 4,
        "nrep_comp": 6
    },
    "6/m": {
        "nrep_real": 8,
        "nrep_comp": 12
    },
    "622": {
        "nrep_real": 6,
        "nrep_comp": 6
    },
    "6mm": {
        "nrep_real": 6,
        "nrep_comp": 6
    },
    "-6m2": {
        "nrep_real": 6,
        "nrep_comp": 6
    },
    "6/mmm": {
        "nrep_real": 12,
        "nrep_comp": 12
    },
    "23": {
        "nrep_real": 3,
        "nrep_comp": 4
    },
    "m-3": {
        "nrep_real": 6,
        "nrep_comp": 8
    },
    "432": {
        "nrep_real": 5,
        "nrep_comp": 5
    },
    "-43m": {
        "nrep_real": 5,
        "nrep_comp": 5
    },
    "m-3m": {
        "nrep_real": 10,
        "nrep_comp": 10
    }
}


def test_number_of_representation_complex():
    for group in GroupsList:
        group_character_info = get_GroupCharacterInfo_from_bilbao(group, use_complex=True)
        num_rep = [ len(reps.keys()) for reps in group_character_info.SymWithChi.values() ]
        
        assert num_rep[0] == nrep_standard[group]["nrep_comp"]
        assert len(set(num_rep)) == 1   # uniform representations
        assert group_character_info.name == group
        assert group_character_info.character_is_complex == True


def test_number_of_representation_real():
    for group in GroupsList:
        group_character_info = get_GroupCharacterInfo_from_bilbao(group, use_complex=False)
        num_rep = [ len(reps.keys()) for reps in group_character_info.SymWithChi.values() ]
        
        assert num_rep[0] == nrep_standard[group]["nrep_real"]
        assert len(set(num_rep)) == 1   # uniform representations
        assert group_character_info.name == group
        assert group_character_info.character_is_complex == False