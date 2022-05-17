import typing, dataclasses
import numpy as np
from ..SALCs.linear_combination import Site, Orbitals, LinearCombination
from ..SALCs.vectorspace import VectorSpace
from ..structure.nncluster import CrystalSitesWithSymmetry


@dataclasses.dataclass
class NamedLC:
    name: str
    lc: LinearCombination


@dataclasses.dataclass
class SubspaceName:
    equivalent_index: int
    orbital: str
    irreps: str

    @property
    def irrep_sequence(self) -> typing.List[str]:
        return self.irreps.split("->")

    @property
    def __eq__(self, other) -> bool:
        return self.equivalent_index == other.equivalent_index and \
            self.orbital == other.orbital and \
            self.irreps == other.irreps

    # unique value so that we can put them in sets
    def __hash__(self):
        return hash(
            ", ".join([str(self.equivalent_index), self.orbital, self.irreps])
        )


@dataclasses.dataclass
class InteractingLCPair:
    rep1: SubspaceName
    rep2: SubspaceName
    lc1: LinearCombination
    lc2: LinearCombination


class MOCoefficient:
    def __init__(self, 
        sitewithsymmetry: CrystalSitesWithSymmetry, labelledvectorspace: typing.Dict[str, VectorSpace]
    ):
        self._crystalsite_with_sym = sitewithsymmetry
        list_named_lcs = _get_namedLC_from_decomposed_vectorspace(labelledvectorspace)

        self._lcs = [namedlc.lc for namedlc in list_named_lcs]
        coefficient_matrix = np.vstack(
            [ namedlc.lc.coefficients for namedlc in list_named_lcs ]
        )

        equivalent_atoms = self._crystalsite_with_sym.equivalent_atoms
        self._orbitals = list_named_lcs[0].lc.orbitals
        named_subspace = _divide_subspace(equivalent_atoms, self._orbitals, coefficient_matrix)
        
        self._subspacenames: typing.List[SubspaceName] = []
        for namedlc, ao_name in zip(list_named_lcs, named_subspace):
            self._subspacenames.append(
                SubspaceName(ao_name[0], ao_name[1], namedlc.name)
            )
        
        matrix_A = []
        self._tp_tuple = []
        for i, lcc_i in enumerate(coefficient_matrix):
            for j, lcc_j in enumerate(coefficient_matrix):
                self._tp_tuple.append((i,j))
                matrix_A.append(
                    np.tensordot(lcc_i, lcc_j, axes=0).flatten()
                )
        self._matrix_A = np.vstack(matrix_A)
        assert self._matrix_A.shape[0] == self._matrix_A.shape[1]

    @property
    def all_aos(self):
        result = []
        for i, orb in enumerate(self._orbitals):
            for ao in orb.aolist:
                result.append((i,ao.l,ao.m))
        return result

    @property
    def orbitals(self) -> typing.List[Orbitals]:
        return self._orbitals

    @property
    def matrixA(self) -> np.ndarray:
        return self._matrix_A

    @property
    def inv_matrixA(self) -> np.ndarray:
        return np.linalg.inv(self._matrix_A)

    @property
    def num_MOpairs(self) -> int:
        return len(self._tp_tuple)

    def get_paired_lc_with_name_by_index(self, input_index: int) \
    -> InteractingLCPair:
        i,j = self._tp_tuple[input_index]
        return InteractingLCPair(
            self._subspacenames[i], self._subspacenames[j], 
            self._lcs[i], self._lcs[j]
        )


def _get_namedLC_from_decomposed_vectorspace(vc_dicts: typing.Dict[str, VectorSpace]) -> typing.List[NamedLC]:
    result = []
    for irrepname, vs in vc_dicts.items():
        for lc, i in zip(vs.get_nonzero_linear_combinations(), range(vs.rank)):
            # only append the rank, temperory solution?
            result.append(
                NamedLC(irrepname, lc.get_normalized())
            )
    return result

def _divide_subspace(equivalent_dict: typing.Dict[int, typing.Set[int]], 
                    orbitals: typing.List[Orbitals], coefficients: np.ndarray) \
    -> typing.List[tuple]: # list of (i, "s"), i is the index of equivalent atom in the sites
    start = 0
    range_dicts = []
    
    for i, orb in enumerate(orbitals):
        subslice = orb.slice_dict
        range_dict = {}
        for k, s in subslice.items():
            end = start + s.stop - s.start
            range_dict[k] = list(range(start, end))
            start = end
        range_dicts.append(range_dict)

    eq_orb_slice = {}
    for ieq, eqset in equivalent_dict.items():
        orb = orbitals[ieq]
        for o in orb.orblist:
            all_index = []
            for eq_atom in eqset:
                all_index += range_dicts[eq_atom][o]
            eq_orb_slice[(ieq, o)] = all_index

    row_name = []
    subspace_namelist = list(eq_orb_slice.keys())
    for row in coefficients:
        row_correspond: typing.List[bool] = []
        for _, indices in eq_orb_slice.items():
            row_correspond.append(np.linalg.norm(row[indices]) > 1e-6)
        assert row_correspond.count(True) == 1
        row_name.append(
            subspace_namelist[row_correspond.index(True)]
        )
    
    return row_name

