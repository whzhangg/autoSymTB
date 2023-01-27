import typing
import collections

import numpy as np


def operation_type_from_rotation_matrix(W:np.array) -> int:
    operation_mapper = {
        (-2, -1): "-6",
        (-1, -1): "-4",
        ( 0, -1): "-3",
        ( 1, -1):  "m",
        (-3, -1): "-1",
        ( 3,  1):  "1",
        (-1,  1):  "2",
        ( 0,  1):  "3",
        ( 1,  1):  "4",
        ( 2,  1):  "6" 
    }
    traceW = round(np.trace(W))
    detW = round(np.linalg.det(W))
    #assert np.isclose(traceW, int(traceW))
    #assert np.isclose(detW, int(detW))
    return operation_mapper[(traceW, detW)]


def get_pointgroupname_from_rotation_matrices(matrices: typing.List[np.array]) -> str:
    W_type = [ operation_type_from_rotation_matrix(m) for m in matrices ]
    counts = collections.Counter(W_type)

    number_type_of_W = {
        #     -6  -4  -3  -2  -1   1   2   3   4   6
        # Laue class -1
        "1" : (  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,),
        "-1" : (  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,),
        # Laue class 2/m
        "2" : (  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,),
        "m" : (  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,),
        "2/m" : (  0,  0,  0,  1,  1,  1,  1,  0,  0,  0,),
        # Laue class mmm
        "222" : (  0,  0,  0,  0,  0,  1,  3,  0,  0,  0,),
        "mm2" : (  0,  0,  0,  2,  0,  1,  1,  0,  0,  0,),
        "mmm" : (  0,  0,  0,  3,  1,  1,  3,  0,  0,  0,),
        # Laue class 4/m
        "4" : (  0,  0,  0,  0,  0,  1,  1,  0,  2,  0,),
        "-4": (  0,  2,  0,  0,  0,  1,  1,  0,  0,  0,),
        "4/m": (  0,  2,  0,  1,  1,  1,  1,  0,  2,  0,),
        # Laue class 4/mmm
        "422": (  0,  0,  0,  0,  0,  1,  5,  0,  2,  0,),
        "4mm": (  0,  0,  0,  4,  0,  1,  1,  0,  2,  0,),
        "-42m": (  0,  2,  0,  2,  0,  1,  3,  0,  0,  0,),
        "4/mmm": (  0,  2,  0,  5,  1,  1,  5,  0,  2,  0,),
        # Laue class -3
        "3": (  0,  0,  0,  0,  0,  1,  0,  2,  0,  0,),
        "-3": (  0,  0,  2,  0,  1,  1,  0,  2,  0,  0,),
        # Laue class -3m
        "32": (  0,  0,  0,  0,  0,  1,  3,  2,  0,  0,),
        "3m": (  0,  0,  0,  3,  0,  1,  0,  2,  0,  0,),
        "-3m": (  0,  0,  2,  3,  1,  1,  3,  2,  0,  0,),
        # Laue class 6/m
        "6": (  0,  0,  0,  0,  0,  1,  1,  2,  0,  2,),
        "-6": (  2,  0,  0,  1,  0,  1,  0,  2,  0,  0,),
        "6/m": (  2,  0,  2,  1,  1,  1,  1,  2,  0,  2,),
        # Laue class 6/mmm
        "622": (  0,  0,  0,  0,  0,  1,  7,  2,  0,  2,),
        "6mm": (  0,  0,  0,  6,  0,  1,  1,  2,  0,  2,),
        "-6m2": (  2,  0,  0,  4,  0,  1,  3,  2,  0,  0,),
        "6/mmm": (  2,  0,  2,  7,  1,  1,  7,  2,  0,  2,),
        # Laue class m-3
        "23": (  0,  0,  0,  0,  0,  1,  3,  8,  0,  0,),
        "m-3": (  0,  0,  8,  3,  1,  1,  3,  8,  0,  0,),
        # Laue class m-3m
        "432": (  0,  0,  0,  0,  0,  1,  9,  8,  6,  0,),
        "-43m": (  0,  6,  0,  6,  0,  1,  3,  8,  0,  0,),
        "m-3m": (  0,  6,  8,  9,  1,  1,  9,  8,  6,  0,)  
    }
    numbers = (
        counts.get("-6",0), counts.get("-4",0), counts.get("-3",0), 
        counts.get("m",0), counts.get("-1",0), counts.get( "1",0), 
        counts.get( "2",0), counts.get( "3",0), counts.get( "4",0), counts.get( "6",0)
    )
    number_to_pointgroup = { counts:ptg for ptg, counts in number_type_of_W.items() } 
    # just reverse the dictionary
    if numbers not in number_to_pointgroup:
        print(numbers)
        raise "_find_point_group_from_counter: this is not a subgroup"
    return number_to_pointgroup[numbers]
