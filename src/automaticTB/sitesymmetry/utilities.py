import numpy as np
import collections, typing

def random_rotation_matrix() -> np.ndarray:
    angles = np.random.rand(3) * 2 * np.pi
    sin = np.sin(angles)
    cos = np.cos(angles)
    yaw = np.array([[1,      0,       0],
                    [0, cos[0], -sin[0]],
                    [0, sin[0],  cos[0]]])
    pitch=np.array([[ cos[1], 0, sin[1]],
                    [0,       1,      0],
                    [-sin[1], 0, cos[1]]])
    roll = np.array([[cos[2],-sin[2], 0],
                    [sin[2], cos[2], 0],
                    [0,           0, 1]])
    return yaw.dot(pitch).dot(roll)

def rotation_fraction_to_cartesian(input_rotations: np.ndarray, cell: np.ndarray) -> np.ndarray:
    cellT = cell.transpose()
    cellT_inv = np.linalg.inv(cellT)
    if input_rotations.shape == (3,3):
        step1 = np.matmul(cellT, input_rotations)
        return np.matmul(step1, cellT_inv)
    else:
        step1 = np.einsum("ij, njk -> nik", cellT, input_rotations)
        return np.einsum("nik, kj -> nij", step1, cellT_inv)

def rotation_cartesian_to_fraction(input_rotations: np.ndarray, cell: np.ndarray) -> np.ndarray:
    cellT = cell.transpose()
    cellT_inv = np.linalg.inv(cellT)
    if input_rotations.shape == (3,3):
        step1 = np.matmul(cellT_inv, input_rotations)
        return np.matmul(step1, cellT)
    else:
        step1 = np.einsum("ij, njk -> nik", cellT_inv, input_rotations)
        return np.einsum("nik, kj -> nij", step1, cellT)

def get_pointgroupname_from_rotation_matrices(matrices: typing.List[np.array]) -> str:
    W_type = [ operation_type_from_rotation_matrix(m) for m in matrices ]
    counts = collections.Counter(W_type)
    return _find_point_group_from_counter(counts)

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



# is the order of function in python matter? see Dimitris Fasarakis Hilliard's answer
# https://stackoverflow.com/questions/44423786/does-the-order-of-functions-in-a-python-script-matter
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

def _find_point_group_from_counter(counts: collections.Counter) -> str:
    numbers = (
        counts.get("-6",0), counts.get("-4",0), counts.get("-3",0), 
        counts.get("m",0), counts.get("-1",0), counts.get( "1",0), 
        counts.get( "2",0), counts.get( "3",0), counts.get( "4",0), counts.get( "6",0)
    )
    number_to_pointgroup = { counts:ptg for ptg, counts in number_type_of_W.items() } # just reverse the dictionary
    if numbers not in number_to_pointgroup:
        print(numbers)
        raise "_find_point_group_from_counter: this is not a subgroup"
    return number_to_pointgroup[numbers]