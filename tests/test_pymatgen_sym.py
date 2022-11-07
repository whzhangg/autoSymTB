import dataclasses, typing, numpy as np
from automaticTB.sitesymmetry.Bilbao.interface import PointGroupData
from automaticTB.sitesymmetry import GroupsList, HM_from_sch
from automaticTB.structure.pymatgen_sym import get_schSymbol_symOperations_from_pos_types
from automaticTB.sitesymmetry.utilities import get_pointgroupname_from_rotation_matrices
from automaticTB.sitesymmetry.SeitzFinder.dress_operation import dress_symmetry_operation
reference = PointGroupData(complex_character=True)
tolerance = 1e-4

@dataclasses.dataclass
class Atoms:
    types: typing.List[int]
    pos: np.ndarray

    @classmethod
    def random_from_group(cls, groupnameHM: str) -> "Atoms":
        group = reference.get_BilbaoPointGroup(groupnameHM)
        random_point = np.array([0.23, 0.56, 0.4])
        points = [np.dot(op.matrix, random_point) for op in group.operations]
        determined = []
        for p in points:
            found = False
            for dp in determined:
                if np.allclose(p, dp, atol=tolerance):
                    found = True
                    break
            if not found:
                determined.append(p)

        return cls(
            [1] * len(determined),
            np.vstack(determined)
        )

for groupname in GroupsList:
    if groupname == "1": continue

    ref_group = reference.get_BilbaoPointGroup(groupname)
    print( "--------------------------------------------------")
    print(f"Group tested : {ref_group.name}")
    print(f"Num. of syms : {len(ref_group.operations)}")
    moleculer = Atoms.random_from_group(groupname)
    print(f"Num. of atoms: {len(moleculer.pos)}")

    sch, ops = get_schSymbol_symOperations_from_pos_types(moleculer.pos, moleculer.types)
    #
    dressed = dress_symmetry_operation(ops)
    print(dressed.keys())
    print(f"Found N. syms: {len(ops)}")
    found_group = get_pointgroupname_from_rotation_matrices(ops)
    print(f"Found group  : {found_group}")

