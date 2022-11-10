import typing
from .sch_symbol import sch_from_HM

GroupsList: typing.List[str] = [
    "1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422", 
    "4mm", "-42m", "4/mmm", "3", "-3", "32", "3m", "-3m", "6", "-6", "6/m", "622",
    "6mm", "-6m2", "6/mmm", "23", "m-3", "432", "-43m", "m-3m"
]

GroupsList_sch = [ sch_from_HM[g] for g in GroupsList ]