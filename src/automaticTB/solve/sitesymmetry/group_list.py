import typing

sch_from_HM: typing.Dict[str, str] = {
    "1": "C1", "-1": "Ci",  "2": "C2", "m": "Cs", "2/m": "C2h",
    "222": "D2", "mm2": "C2v", "mmm": "D2h", "4": "C4", "-4": "S4",
    "4/m": "C4h", "422": "D4", "4mm": "C4v", "-42m": "D2d",
    "4/mmm": "D4h", "3": "C3", "-3": "S6", "32": "D3", "3m": "C3v",
    "-3m": "D3d", "6": "C6", "-6": "C3h", "6/m": "C6h", "622": "D6",
    "6mm": "C6v", "-6m2": "D3h", "6/mmm": "D6h", "23": "T",
    "m-3": "Th", "432": "O", "-43m": "Td", "m-3m": "Oh",
}

GroupsList: typing.List[str] = [
    "1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", 
    "422", "4mm", "-42m", "4/mmm", "3", "-3", "32", "3m", "-3m", "6", 
    "-6", "6/m", "622", "6mm", "-6m2", "6/mmm", "23", "m-3", "432", 
    "-43m", "m-3m"
]

HM_from_sch = {sch: hm for hm, sch in sch_from_HM.items()}

GroupsList_sch = [sch_from_HM[g] for g in GroupsList]