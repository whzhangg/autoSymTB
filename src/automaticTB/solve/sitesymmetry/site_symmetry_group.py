import typing
import dataclasses

import numpy as np

from .utilities import get_pointgroupname_from_rotation_matrices
from .dress_operation import dress_symmetry_operation
from .bilbao import BilbaoGroupOperation, get_BilbaoPointGroup
from .group_list import sch_from_HM
from .subduction import symmetry_reduction


@dataclasses.dataclass
class SiteSymmetryGroup:
    groupname: str
    operations: typing.List[np.ndarray]
    irreps: typing.Dict[str, typing.List[float]]
    irrep_dimension: typing.Dict[str, int]
    subgroups: typing.List[str]
    dressed_op: typing.Dict[str, np.ndarray]

    @property
    def is_spherical_symmetry(self) -> bool:
        return self.groupname == "Kh"

    @property
    def sch_symbol(self) -> str:
        return sch_from_HM[self.groupname]

    @classmethod
    def from_groupname(cls, groupname: str) -> "SiteSymmetryGroup":
        groupdata = get_BilbaoPointGroup(groupname)
        rotations = [ op.matrix for op in groupdata.operations ]
        ssgroup = cls.from_cartesian_matrices(rotations)
        if ssgroup.groupname != groupname:
            raise RuntimeError("SiteSymmetryGroup.from_groupname, something wrong")
        return ssgroup

    @classmethod
    def from_cartesian_matrices(cls, 
        cartesian_matrices: typing.List[np.ndarray], groupname: str = ""
    ) -> "SiteSymmetryGroup":
        main_groupname = get_pointgroupname_from_rotation_matrices(cartesian_matrices)
        givengroup = groupname if groupname else main_groupname
        assert givengroup in symmetry_reduction[main_groupname]  
        
        dressed = dress_symmetry_operation(cartesian_matrices) # matrices with seitz symbol
        
        operation: typing.List[BilbaoGroupOperation] \
            = _find_corresponding_characters(givengroup, main_groupname, dressed)
        matrices = []
        index_of_identity = -1
        irreps = {}
        dressed_subgroup_operation = {}
        for op in operation:
            if index_of_identity < 0 and np.allclose(op.matrix, np.eye(3)):
                index_of_identity = len(matrices)
            matrices.append(op.matrix)
            dressed_subgroup_operation[op.seitz] = op.matrix
            for irrep,value in op.chi.items():
                irreps.setdefault(irrep, []).append(value)

        dimension = {}
        for key, value in irreps.items():
            dimension[key] = int(np.linalg.norm(value[index_of_identity]))

        return cls(
            givengroup, 
            matrices,
            irreps,
            dimension, 
            list(symmetry_reduction[givengroup].keys()),
            dressed_subgroup_operation
        )


    def get_subgroup(self, subgroup: str) -> "SiteSymmetryGroup":
        if self.is_spherical_symmetry:
            print("trying to get subgroup of spherial symmetric group")
            SystemExit()

        if subgroup not in self.subgroups:
            raise ValueError(f"{subgroup} is not a subgroup of the current group {self.groupname}")
        
        operation: typing.List[BilbaoGroupOperation] = _find_corresponding_characters(subgroup, self.groupname, self.dressed_op)

        matrices = []
        index_of_identity = -1
        irreps = {}
        dressed_subgroup_operation = {}
        for op in operation:
            if index_of_identity < 0 and np.allclose(op.matrix, np.eye(3)):
                index_of_identity = len(matrices)
            matrices.append(op.matrix)
            dressed_subgroup_operation[op.seitz] = op.matrix
            for irrep,value in op.chi.items():
                irreps.setdefault(irrep, []).append(value)

        dimension = {}
        for key, value in irreps.items():
            dimension[key] = int(np.linalg.norm(value[index_of_identity]))

        return SiteSymmetryGroup(
            subgroup, 
            matrices,
            irreps,
            dimension, 
            list(symmetry_reduction[subgroup].keys()),
            dressed_subgroup_operation
        )


    def get_subgroup_from_subgroup_and_seitz_map(
        self, subgroup: str, seitz_mapper: typing.Dict[str, str]
    ) -> "SiteSymmetryGroup":
        if self.is_spherical_symmetry:
            print("trying to get subgroup of spherial symmetric group")
            SystemExit()
        if subgroup not in self.subgroups:
            raise ValueError(f"{subgroup} is not a subgroup of the current group {self.groupname}")
        
        required_subgroup = get_BilbaoPointGroup(subgroup)
        operation: typing.List[BilbaoGroupOperation] = _find_cooresponding_characters_from_map(
            self.dressed_op, 
            required_subgroup.seitz_operation_dict,
            seitz_mapper
        )

        matrices = []
        index_of_identity = -1
        irreps = {}
        dressed_subgroup_operation = {}
        for op in operation:
            if index_of_identity < 0 and np.allclose(op.matrix, np.eye(3)):
                index_of_identity = len(matrices)
            matrices.append(op.matrix)
            dressed_subgroup_operation[op.seitz] = op.matrix
            for irrep,value in op.chi.items():
                irreps.setdefault(irrep, []).append(value)

        dimension = {}
        for key, value in irreps.items():
            dimension[key] = int(np.linalg.norm(value[index_of_identity]))

        return SiteSymmetryGroup(
            subgroup, 
            matrices,
            irreps,
            dimension, 
            list(symmetry_reduction[subgroup].keys()),
            dressed_subgroup_operation
        )

    @staticmethod
    def get_spherical_symmetry_group() -> "SiteSymmetryGroup":
        return SiteSymmetryGroup(
                    "Kh", 
                    [], {}, {}, [], {}
                )


def _find_cooresponding_characters_from_map(
    dressed: typing.Dict[str, np.ndarray], 
    required_group_reference: typing.Dict[str, BilbaoGroupOperation],
    reduction_reference: typing.Dict[str, str]) \
-> typing.List[BilbaoGroupOperation]:
    obtained_operation: typing.List[BilbaoGroupOperation] = []
    
    for subgroup_seitz, maingroup_seitz in reduction_reference.items():
        obtained_operation.append(
            BilbaoGroupOperation(
                subgroup_seitz,
                dressed[maingroup_seitz],
                required_group_reference[subgroup_seitz].chi
            )
        )
    return obtained_operation


def _find_corresponding_characters(
    givengroup: str, 
    maingroup: str,
    dressed: typing.Dict[str, np.ndarray]
) -> typing.List[BilbaoGroupOperation]:
    reduction_reference = symmetry_reduction[maingroup][givengroup]
    required_group_reference = get_BilbaoPointGroup(givengroup).seitz_operation_dict

    return _find_cooresponding_characters_from_map(
        dressed, required_group_reference, reduction_reference
    )

