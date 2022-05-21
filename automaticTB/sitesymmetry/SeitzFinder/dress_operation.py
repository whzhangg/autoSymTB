import numpy as np
import dataclasses, typing
from ..utilities import get_pointgroupname_from_rotation_matrices
from ..Bilbao.group_operations import SeitzSymbol
from .crystal_axes import CrystalAxes
from .axes_finder import get_finder
from .symmetry_elements import (
    SymDirection_Set, Symmetries_on_Direction, Symmetries_on_Direction_Set,
    SymOp_on_Direction
)

def organize_by_index(items, indics) -> list:
    sets = {}
    for i, ref in enumerate(indics):
        sets.setdefault(ref, []).append(i)
    item_set = []
    for s in sets.values():
        item_set.append([ items[i] for i in s ])
    return item_set

def organize_operation_by_symmetry_direction(symops: typing.List[SymOp_on_Direction]) \
-> typing.List[Symmetries_on_Direction]:
    direction_ref = []
    for iop, op in enumerate(symops):
        found = [op.direction == symops[index].direction for index in direction_ref]
        if any(found):
            direction_ref.append(found.index(True))
        else:
            direction_ref.append(iop)
    
    classified = organize_by_index(symops, direction_ref)
    listSymOnDirection: typing.List[Symmetries_on_Direction] = []
    for c in classified:
        listSymOnDirection.append(Symmetries_on_Direction(c))
    
    return listSymOnDirection

def organize_symmetry_into_equivalent_directions(
    rotations: typing.List[np.ndarray], listSymOnDirection: typing.List[Symmetries_on_Direction] ) \
-> typing.List[Symmetries_on_Direction_Set]:

    d = listSymOnDirection[0].direction.vector
    direction_sets = [
        SymDirection_Set([ np.dot(rot, d) for rot in rotations ])
    ]
    
    which = []
    for d in listSymOnDirection:
        # check all the direction set to see if it belongs to any
        found = False
        for iset, equi_directions in enumerate(direction_sets):
            if d.direction.vector in equi_directions: 
                found = True
                which.append(iset)
        
        if not found:
            which.append(len(direction_sets))
            direction_sets.append(
                SymDirection_Set([ np.dot(rot, d.direction.vector) for rot in rotations ])
            )
    
    classified = organize_by_index(listSymOnDirection, which)
    listSymOnDirSet: typing.List[Symmetries_on_Direction_Set] = []
    for c in classified:
        listSymOnDirSet.append(Symmetries_on_Direction_Set(c))

    return listSymOnDirSet

def identify_primiary_secondary_tertiary_axes(groupname: str, 
    listSymOnDirSet: typing.List[Symmetries_on_Direction_Set]) \
-> typing.Dict[str, Symmetries_on_Direction_Set]:

    givencrystalaxis = CrystalAxes[groupname]
    main_axis: typing.Dict[str, Symmetries_on_Direction_Set] = {}
    for key, value in givencrystalaxis.items():
        for i, dirset in enumerate(listSymOnDirSet):
            if dirset.op_set == value:
                main_axis[key] = dirset
                listSymOnDirSet.pop(i)
                break

    return main_axis

def dress_symmetry_operation(matrixs: typing.List[np.ndarray]) -> typing.Dict[str, np.ndarray]:
    # give seitz symbol for matrix
    groupname = get_pointgroupname_from_rotation_matrices(matrixs)
    operations = [ SymOp_on_Direction.from_cartesian_rotation_matrix(m) for m in matrixs ]

    listSymOnDirection = organize_operation_by_symmetry_direction(operations)
    #direction_ref = []
    #for iop, op in enumerate(operations):
    #    found = [op.direction == operations[index].direction for index in direction_ref]
    #    if any(found):
    #        direction_ref.append(found.index(True))
    #    else:
    #        direction_ref.append(iop) 
    #classified = organize_by_index(operations, direction_ref)
    #listSymOnDirection: typing.List[Symmetries_on_Direction] = []
    #for c in classified:
    #    listSymOnDirection.append(Symmetries_on_Direction(c))

    listSymOnDirSet = organize_symmetry_into_equivalent_directions(matrixs, listSymOnDirection)
    #d = listSymOnDirection[0].direction.vector
    #direction_sets = [
    #    SymDirection_Set([ np.dot(op.matrix, d) for op in operations ])
    #]
    
    #which = []
    #for d in listSymOnDirection:
    #    # check all the direction set to see if it belongs to any
    #    found = False
    #    for iset, equi_directions in enumerate(direction_sets):
    #        if d.direction.vector in equi_directions: 
    #            found = True
    #            which.append(iset)
    #    
    #    if not found:
    #        which.append(len(direction_sets))
    #        direction_sets.append(
    #            SymDirection_Set([ np.dot(op.matrix, d.direction.vector) for op in operations ])
    #        )
    
    #classified = organize_by_index(listSymOnDirection, which)
    #listSymOnDirSet: typing.List[Symmetries_on_Direction_Set] = []
    #for c in classified:
    #    listSymOnDirSet.append(Symmetries_on_Direction_Set(c))
        
    # use the result to find set of symmetry axis
    main_axis = identify_primiary_secondary_tertiary_axes(groupname, listSymOnDirSet)
    #givencrystalaxis = CrystalAxes[groupname]
    #main_axis = {}
    #for key, value in givencrystalaxis.items():
    #    for i, dirset in enumerate(listSymOnDirSet):
    #        if dirset.op_set == value:
    #            main_axis[key] = dirset
    #            listSymOnDirSet.pop(i)
    #            break
    
    finder = get_finder(groupname, main_axis)

    dressedop = {}
    for op in operations:
        hkl = finder.find_hkl_from_direction(op.direction)
        finded_direction = finder.find_direction_from_hkl(hkl)
        op.set_new_sense_by_direction(finded_direction)
        sz = SeitzSymbol(op.operation, hkl, op.sense)
        dressedop[str(sz)] = op.matrix
    return dressedop

