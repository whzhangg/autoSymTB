import typing
CrystalAxes: typing.Dict[str, typing.Dict[str, typing.Set[str]]] = {
    "1": {
     },
    "-1": {
     },
    "2": {
        "primiary" : {"2"}
     },
    "m": {
        "primiary" : {"m"}
     },
    "2/m": {
        "primiary" : {"m", "2"}
     },
    "222": {
        "primiary" : {"2"},
        "secondary": {"2"},
        "tertiary" : {"2"},
     },
    "mm2": {
        "primiary" : {"m"},
        "secondary": {"m"},
        "tertiary" : {"2"},
     },
    "mmm": {
        "primiary" : {"m", "2"},
        "secondary": {"m", "2"},
        "tertiary" : {"m", "2"},
     },
    "4": {
        "primiary" : {"2", "4"}
     },
    "-4": {
        "primiary" : {"2", "-4"}
     },
    "4/m": {
        "primiary" : {"m","2","-4","4"}
     },
    "422": {
        "primiary" : {"2","4"},
        "secondary": {"2"},
        "tertiary" : {"2"},
     },
    "4mm": {
        "primiary" : {"2", "4"},
        "secondary": {"m"},
        "tertiary" : {"m"},
     },
    "-42m": {
        "primiary" : {"-4","2"},
        "secondary": {"2"},
        "tertiary" : {"m"},
     },
    "4/mmm": {
        "primiary" : {"m", "2", "-4", "4"},
        "secondary": {"m","2"},
        "tertiary" : {"m","2"},
     },
    "3": {
        "primiary" : {"3"}
     },
    "-3": {
        "primiary" : {"-3", "3"}
     },
    "32": {
        "primiary" : {"3"},
        "secondary": {"2"}
     },
    "3m": {
        "primiary" : {"3"},
        "secondary": {"m"}
     },
    "-3m": {
        "primiary" : {"-3","3"},
        "secondary": {"m", "2"}
     },
    "6": {
        "primiary" : {"6","2","3"}
     },
    "-6": {
        "primiary" : {"m","3","-6"}
     },
    "6/m": {
        "primiary" : {"6", "-3", "-6", "m", "2", "3"}
     },
    "622": {
        "primiary" : {"6","2","3"},
        "secondary": {"2"},
        "tertiary" : {"2"},
     },
    "6mm": {
        "primiary" : {"6","2","3"},
        "secondary": {"m"},
        "tertiary" : {"m"},
     },
    "-6m2": {
        "primiary" : {"m", "3", "-6"},
        "secondary": {"m"},
        "tertiary" : {"2"},
     },
    "6/mmm": {
        "primiary" : {"6", "-3", "-6", "m", "2", "3"},
        "secondary": {"m", "2"},
        "tertiary" : {"m", "2"},
     },
    "23": {
        "primiary" : {"2"},
        "secondary": {"3"},
     },
    "m-3": {
        "primiary" : {"m", "2"},
        "secondary": {"-3", "3"}
     },
    "432": {
        "primiary" : {"2", "4"},
        "secondary": {"3"},
        "tertiary" : {"2"},
     },
    "-43m": {
        "primiary" : {"-4","2"},
        "secondary": {"3"},
        "tertiary" : {"m"},
     },
    "m-3m": {
        "primiary" : {"m","2","-4","4"},
        "secondary": {"-3", "3"},
        "tertiary" : {"m", "2"},
     }
}

def print_template():
    from group_list import GroupsList
    for group in GroupsList:
        print(f"    \"{group}\": {{")
        print( "        \"primiary\" : { },")
        print( "        \"secondary\": { },")
        print( "        \"tertiary\" : { },")
        print("     },")
