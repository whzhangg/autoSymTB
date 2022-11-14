subduction_data = {
    "1" : { # C1 OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "-1" : { # Ci OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "2" : { # C2 OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "m" : { # Cs OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "2/m" : { # C2h OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "222" : { # D2 OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "mm2" : { # C2v OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "mmm" : { # D2h OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "4" : { # C4 OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "-4" : { # S4 OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "4/m" : { # C4h OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "422" : { # D4 OK
        "sub"  : "222", # D2
        "splittable_rep" : ["E"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 001",
            "2 010" : "2 010",
            "2 100" : "2 100",
        }
    # supergroup: "1 000", "2 001", "2 010", "2 1-10", "2 100", "2 110", "4 001 +", "4 001 -"
    },
    "4mm" : { # C4v OK
        "sub"  : "mm2", # C2v
        "splittable_rep" : ["E"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 001",
            "m 010" : "m 010",
            "m 100" : "m 100",
        }
    # supergroup: "1 000", "2 001", "4 001 +", "4 001 -", "m 010", "m 1-10", "m 100", "m 110"
    },
    "-42m" : { # D2d OK
        "sub"  : "2", # C2
        "splittable_rep" : ["E"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 100",
        }
    # supergroup: "-4 001 +", "-4 001 -", "1 000", "2 001", "2 010", "2 100", "m 1-10", "m 110"
    },
    "4/mmm" : { # D4h OK
        "sub"  : "mm2", # C2v
        "splittable_rep" : ["E_g", "E_u"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 001",
            "m 010" : "m 010",
            "m 100" : "m 100",
        }
    # supergroup: "-1 000", "-4 001 +", "-4 001 -", "1 000", "2 001", "2 010", "2 1-10", "2 100", "2 110", "4 001 +", "4 001 -", "m 001", "m 010", "m 1-10", "m 100", "m 110"
    },
    "3" : { # C3 OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "-3" : { # S6 OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "32" : { # D3 OK
        "sub"  : "2", # C2
        "splittable_rep" : ["E"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 100",
        }
    # supergroup: "1 000", "2 010", "2 100", "2 110", "3 001 +", "3 001 -"
    },
    "3m" : { # C3v OK
        "sub"  : "m", # Cs
        "splittable_rep" : ["E"],
        "seitz_map" : {
            "1 000" : "1 000",
            "m 001" : "m 100",
        }
    # supergroup: "1 000", "3 001 +", "3 001 -", "m 010", "m 100", "m 110"
    },
    "-3m" : { # D3d OK
        "sub"  : "2", # C2
        "splittable_rep" : ["E_g", "E_u"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 100",
        }
    # supergroup: "-1 000", "-3 001 +", "-3 001 -", "1 000", "2 010", "2 100", "2 110", "3 001 +", "3 001 -", "m 010", "m 100", "m 110"
    },
    "6" : { # C6 OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "-6" : { # C3h OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "6/m" : { # C6h OK
        "sub"  : "1", # C1
        "splittable_rep" : [],
        "seitz_map" : {
            "1 000" : "1 000",
        }
    },
    "622" : { # D6 OK
        "sub"  : "2", # C2
        "splittable_rep" : ["E_1", "E_2"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 100",
        }
    # supergroup: "1 000", "2 001", "2 010", "2 1-10", "2 100", "2 110", "2 120", "2 210", "3 001 +", "3 001 -", "6 001 +", "6 001 -"
    },
    "6mm" : { # C6v OK
        "sub"  : "mm2", # C2v
        "splittable_rep" : ["E_1", "E_2"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 001",
            "m 010" : "m 120",
            "m 100" : "m 100",
        }
    # supergroup: "1 000", "2 001", "3 001 +", "3 001 -", "6 001 +", "6 001 -", "m 010", "m 1-10", "m 100", "m 110", "m 120", "m 210"
    },
    "-6m2" : { # D3h OK
        "sub"  : "mm2", # C2v
        "splittable_rep" : ["E_'", "E_''"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 120",
            "m 010" : "m 100",
            "m 100" : "m 001",
        }
    # supergroup: "-6 001 +", "-6 001 -", "1 000", "2 1-10", "2 120", "2 210", "3 001 +", "3 001 -", "m 001", "m 010", "m 100", "m 110"
    },
    "6/mmm" : { # D6h OK
        "sub"  : "mmm", # D2h
        "splittable_rep" : ["E_1g", "E_1u", "E_2g", "E_2u"],
        "seitz_map" : {
            "-1 000" : "-1 000",
            "1 000" : "1 000",
            "2 001" : "2 001",
            "2 010" : "2 100",
            "2 100" : "2 120",
            "m 001" : "m 001",
            "m 010" : "m 100",
            "m 100" : "m 120",
        }
    # supergroup: "-1 000", "-3 001 +", "-3 001 -", "-6 001 +", "-6 001 -", "1 000", "2 001", "2 010", "2 1-10", "2 100", "2 110", "2 120", "2 210", "3 001 +", "3 001 -", "6 001 +", "6 001 -", "m 001", "m 010", "m 1-10", "m 100", "m 110", "m 120", "m 210"
    },
    "23" : { # T OK
        "sub"  : "222", # D2
        "splittable_rep" : ["T"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 001",
            "2 010" : "2 010",
            "2 100" : "2 100",
        }
    # supergroup: "1 000", "2 001", "2 010", "2 100", "3 -1-11 +", "3 -1-11 -", "3 -11-1 +", "3 -11-1 -", "3 1-1-1 +", "3 1-1-1 -", "3 111 +", "3 111 -"
    },
    "m-3" : { # Th OK
        "sub"  : "mm2", # C2v
        "splittable_rep" : ["T_g", "T_u"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 001",
            "m 010" : "m 010",
            "m 100" : "m 100",
        }
    # supergroup: "-1 000", "-3 -1-11 +", "-3 -1-11 -", "-3 -11-1 +", "-3 -11-1 -", "-3 1-1-1 +", "-3 1-1-1 -", "-3 111 +", "-3 111 -", "1 000", "2 001", "2 010", "2 100", "3 -1-11 +", "3 -1-11 -", "3 -11-1 +", "3 -11-1 -", "3 1-1-1 +", "3 1-1-1 -", "3 111 +", "3 111 -", "m 001", "m 010", "m 100"
    },
    "432" : { # O OK
        "sub"  : "222", # D2
        "splittable_rep" : ["E", "T_1", "T_2"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 001",
            "2 010" : "2 1-10",
            "2 100" : "2 110",
        }
    # supergroup: "1 000", "2 -101", "2 001", "2 01-1", "2 010", "2 011", "2 1-10", "2 100", "2 101", "2 110", "3 -1-11 +", "3 -1-11 -", "3 -11-1 +", "3 -11-1 -", "3 1-1-1 +", "3 1-1-1 -", "3 111 +", "3 111 -", "4 001 +", "4 001 -", "4 010 +", "4 010 -", "4 100 +", "4 100 -"
    },
    "-43m" : { # Td OK
        "sub"  : "mm2", # C2v
        "splittable_rep" : ["E", "T_1", "T_2"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 001",
            "m 010" : "m 1-10",
            "m 100" : "m 110",
        }
    # supergroup: "-4 001 +", "-4 001 -", "-4 010 +", "-4 010 -", "-4 100 +", "-4 100 -", "1 000", "2 001", "2 010", "2 100", "3 -1-11 +", "3 -1-11 -", "3 -11-1 +", "3 -11-1 -", "3 1-1-1 +", "3 1-1-1 -", "3 111 +", "3 111 -", "m -101", "m 01-1", "m 011", "m 1-10", "m 101", "m 110"
    },
    "m-3m" : { # Oh OK
        "sub"  : "mm2", # C2v
        "splittable_rep" : ["E_g", "E_u", "T_1g", "T_1u", "T_2g", "T_2u"],
        "seitz_map" : {
            "1 000" : "1 000",
            "2 001" : "2 001",
            "m 010" : "m 1-10",
            "m 100" : "m 110",
        }
    # supergroup: "-1 000", "-3 -1-11 +", "-3 -1-11 -", "-3 -11-1 +", "-3 -11-1 -", "-3 1-1-1 +", "-3 1-1-1 -", "-3 111 +", "-3 111 -", "-4 001 +", "-4 001 -", "-4 010 +", "-4 010 -", "-4 100 +", "-4 100 -", "1 000", "2 -101", "2 001", "2 01-1", "2 010", "2 011", "2 1-10", "2 100", "2 101", "2 110", "3 -1-11 +", "3 -1-11 -", "3 -11-1 +", "3 -11-1 -", "3 1-1-1 +", "3 1-1-1 -", "3 111 +", "3 111 -", "4 001 +", "4 001 -", "4 010 +", "4 010 -", "4 100 +", "4 100 -", "m -101", "m 001", "m 01-1", "m 010", "m 011", "m 1-10", "m 100", "m 101", "m 110"
    }
}

_group_symbol_map = {
    "1": "C1",
    "-1": "Ci",
    "2": "C2",
    "m": "Cs",
    "2/m": "C2h",
    "222": "D2",
    "mm2": "C2v",
    "mmm": "D2h",
    "4": "C4",
    "-4": "S4",
    "4/m": "C4h",
    "422": "D4",
    "4mm": "C4v",
    "-42m": "D2d",
    "4/mmm": "D4h",
    "3": "C3",
    "-3": "S6",
    "32": "D3",
    "3m": "C3v",
    "-3m": "D3d",
    "6": "C6",
    "-6": "C3h",
    "6/m": "C6h",
    "622": "D6",
    "6mm": "C6v",
    "-6m2": "D3h",
    "6/mmm": "D6h",
    "23": "T",
    "m-3": "Th",
    "432": "O",
    "-43m": "Td",
    "m-3m": "Oh"
}

_subduction_table = {
    "1":  "1",  # C1
    "2":  "1",  # C2
    "3":  "1",  # C3
    "4":  "1",    # C4
    "6":  "1", # C6
    "-1": "1",  # Ci
    "m":  "1",  # Cs
    "-4": "1",  # S4
    "-3": "1", # S6 
    "222":   "1", # D2
    "32":    "2", # D3
    "422":   "222", # D4
    "622":   "2", # D6
    "mmm":   "1", # D2h
    "-6m2":  "mm2", # D3h
    "4/mmm": "mm2", # D4h 
    "6/mmm": "mmm", # D6h
    "-42m":  "2", # D2d
    "-3m":   "2", # D3d
    "mm2":   "1", # C2v
    "3m":    "m", # C3v 
    "4mm":   "mm2", # C4v
    "6mm":   "mm2", # C6v
    "2/m":   "1", # C2h
    "-6":    "1", # C3h
    "4/m":   "1", # C4h
    "6/m":   "1", # C6h
    "432":   "222", # O
    "23":    "222", # T
    "m-3m":  "mm2", # Oh
    "m-3":   "mm2", # Th
    "-43m":  "mm2", # Td
}

def print_subduction():
    from .group_list import GroupsList
    from automaticTB.tools import write_json
    table = { group: "" for group in GroupsList }
    write_json(table, "tmp.txt")

def print_subduction_operations_template_real():
    from .group_list import GroupsList
    from .Bilbao.interface import PointGroupData
    pgd = PointGroupData(complex_character=False)
    for group in GroupsList:
        group_data = pgd.get_BilbaoPointGroup(group)
        subduction_group = _subduction_table[group]
        assert subduction_group in group_data.subgroups
        subgroup_data = pgd.get_BilbaoPointGroup(subduction_group)
        reps = list(group_data.operations[0].chi.keys())
        seitz_super = [ op.seitz for op in group_data.operations ]
        seitz_map = {
            op.seitz: op.seitz for op in subgroup_data.operations
        }
        print(f"\"{group}\" : {{ # {_group_symbol_map[group]}")
        print(f"    \"sub\"  : \"{subduction_group}\", # {_group_symbol_map[subduction_group]}")
        quoted_reps = [ f"\"{r}\"" for r in reps ]
        if subduction_group == "1":
            print( "    \"splittable_rep\" : [],")
        else:
            print( "    \"splittable_rep\" : [" + ", ".join(quoted_reps) + "],")

        print( "    \"seitz_map\" : {" )
        for key, value in seitz_map.items():
            print( f"        \"{key}\" : \"{value}\",")

        print( "    }")
        if subduction_group != "1":
            quoted_seitz = [ f"\"{s}\"" for s in seitz_super]
            print( " # supergroup: "+ ", ".join(quoted_seitz))
        print( "},")

if __name__ == "__main__":
    pass