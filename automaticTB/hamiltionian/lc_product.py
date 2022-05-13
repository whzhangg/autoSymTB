import numpy as np
import typing, dataclasses
from ..linear_combination import LinearCombination
from ..SALCs.vectorspace import VectorSpace

@dataclasses.dataclass
class NamedLC:
    name: str
    lc: LinearCombination

@dataclasses.dataclass
class InteractingLCPair:
    rep1: str
    rep2: str
    lc1: LinearCombination
    lc2: LinearCombination

class ListofNamedLC:
    # this is similar to VectorSpace, maybe we should extract a common parent
    def __init__(self, nlc_list: typing.List[NamedLC]) -> None:
        self._orbitals = nlc_list[0].lc.orbitals
        self._sites = nlc_list[0].lc.sites
        for nlc in nlc_list:
            if self._orbitals != nlc.lc.orbitals or self._sites != nlc.lc.sites:
                raise
        
        self._LC_coefficients = np.vstack(
            [ nlc.lc.coefficients.flatten() for nlc in nlc_list ]
        )
        self._LC_symbol = [ nlc.name for nlc in nlc_list ]

        matrix_A = []
        self._tp_tuple = []
        for i, lcc_i in enumerate(self._LC_coefficients):
            for j, lcc_j in enumerate(self._LC_coefficients):
                self._tp_tuple.append((i,j))
                matrix_A.append(
                    np.tensordot(lcc_i, lcc_j, axes=0).flatten()
                )
        self._matrix_A = np.vstack(matrix_A)
        
    @classmethod
    def from_decomposed_vectorspace(cls, vc_dicts: typing.Dict[str, VectorSpace]):
        result = []
        for irrepname, vs in vc_dicts.items():
            for lc in vs.get_nonzero_linear_combinations():
                result.append(
                    NamedLC(irrepname, lc.normalized())
                )
        return cls(result)

    @property
    def matrixA(self):
        return self._matrix_A

    @property
    def rank(self):
        return len(self._tp_tuple)

    @property
    def tp_tuple(self) -> typing.List[typing.Tuple[int, int]]:
        return self._tp_tuple

    @property
    def lc_symbols(self) -> typing.List[str]:
        return self._LC_symbol

    def get_pair_LCs(self, i: int) -> InteractingLCPair:
        tp = self._tp_tuple[i]
        lcshape = (len(self._sites), self._orbitals.num_orb)
        return InteractingLCPair(
            self._LC_symbol[tp[0]],
            self._LC_symbol[tp[1]],
            LinearCombination(self._sites, self._orbitals, self._LC_coefficients[tp[0]].reshape(lcshape)),
            LinearCombination(self._sites, self._orbitals, self._LC_coefficients[tp[1]].reshape(lcshape)),
        )

