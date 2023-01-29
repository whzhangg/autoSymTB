import typing

import numpy as np


def unordered_list_equivalent(array1: np.ndarray, array2: np.ndarray, eps: float) -> bool:
    """return True if the rows in two input array are the same"""
    rows_in_array2 = [row for row in array2] # because we want to pop
    for row1 in array1:
        found2 = -1
        for i2, row2 in enumerate(rows_in_array2):
            if np.allclose(row1, row2, atol=eps):
                found2 = i2
                break
        else:
            return False
        rows_in_array2.pop(found2) 
    return True


def find_first_rotation_btw_clusters(
    types1: typing.List[int], position1: np.ndarray, 
    types2: typing.List[int], position2: np.ndarray, possible_rotations: np.ndarray, tol: float
) -> np.ndarray:
    natom = len(types1)
    set1 = np.zeros((natom, 3), dtype=float)
    set2 = np.zeros((natom, 3), dtype=float)
    for i in range(natom):
        set1[i,0] = types1[i]
        set2[i,0] = types2[i]
        set1[i,1:] = position1[i]
        set2[i,1:] = position2[i]
    
    rot44 = np.eye(4,4)
    for rot33 in possible_rotations:
        rot44[1:4, 1:4] = rot33
        rotated_array1 = np.einsum("ij,zj->zi", rot44, set1)
        if unordered_list_equivalent(rotated_array1, set2, tol):
            return rot33
    else:
        print("=" * 60)
        print("Cluster 1")
        for row in set1:
            print("    {:>5.1f} @ {:>8.3f},{:>8.3f},{:>8.3f}".format(*row))
        print("Cluster 2")
        for row in set2:
            print("    {:>5.1f} @ {:>8.3f},{:>8.3f},{:>8.3f}".format(*row))
        print("=" * 60)
        raise RuntimeError("no rotation found")


def find_all_rotations_btw_clusters(
    types1: typing.List[int], position1: np.ndarray, 
    types2: typing.List[int], position2: np.ndarray, possible_rotations: np.ndarray, tol: float
) -> np.ndarray:
    natom = len(types1)
    set1 = np.zeros((natom, 3), dtype=float)
    set2 = np.zeros((natom, 3), dtype=float)
    for i in range(natom):
        set1[i,0] = types1[i]
        set2[i,0] = types2[i]
        set1[i,1:] = position1[i]
        set2[i,1:] = position2[i]
    
    rotations = []
    rot44 = np.eye(4,4)
    for rot33 in possible_rotations:
        rot44[1:4, 1:4] = rot33
        rotated_array1 = np.einsum("ij,zj->zi", rot44, set1)
        if unordered_list_equivalent(rotated_array1, set2, tol):
            rotations.append(rot33)
    