import abc, typing, dataclasses
import numpy as np
from ..interaction import InteractionPairs
from ..atomic_orbitals import AO
from .secular_solver import solve_secular, solve_secular_sorted

@dataclasses.dataclass
class Pindex_lm:
    pindex: int  # index in primitive cell
    l: int
    m: int
    translation: np.ndarray

    def __eq__(self, other) -> bool:
        return self.pindex == other.pindex and \
               self.l == other.l and self.m == other.m and \
               np.allclose(self.translation, other.translation)


@dataclasses.dataclass
class HijR:
    left: Pindex_lm
    right: Pindex_lm
    value: float


@dataclasses.dataclass
class SijR:
    left: Pindex_lm
    right: Pindex_lm
    value: float


def gather_InteractionPairs_into_HijRs(AOinteractions: typing.List[InteractionPairs]) -> typing.List[HijR]:
    HijR_list: typing.List[HijR] = []
    for interaction in AOinteractions:
        for pair, value in zip(interaction.pairs, interaction.interactions):
            left: AO = pair.left
            right: AO = pair.right
            HijR_list.append(
                HijR(
                    Pindex_lm(left.primitive_index, left.l, left.m, left.translation),
                    Pindex_lm(right.primitive_index, right.l, right.m, right.translation),
                    value
                )
            )
    return HijR_list


def gather_InteractionPairs_into_SijRs(AOinteractions: typing.List[InteractionPairs]) -> typing.List[SijR]:
    SijR_list: typing.List[HijR] = []
    for interaction in AOinteractions:
        for pair, value in zip(interaction.pairs, interaction.interactions):
            left: AO = pair.left
            right: AO = pair.right
            SijR_list.append(
                SijR(
                    Pindex_lm(left.primitive_index, left.l, left.m, left.translation),
                    Pindex_lm(right.primitive_index, right.l, right.m, right.translation),
                    value
                )
            )
    return SijR_list


class TightBindingBase(abc.ABC):

    @property
    @abc.abstractmethod
    def cell(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def positions(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def types(self) -> typing.List[int]:
        pass

    @property
    @abc.abstractmethod
    def basis(self) -> typing.List[Pindex_lm]:
        pass

    @abc.abstractmethod 
    def Hijk(self, frac_k: np.ndarray) -> np.ndarray:
        pass 

    @abc.abstractmethod
    def Sijk(self, frac_k: np.ndarray) -> np.ndarray:
        pass

    def solveE_at_k(self, k: np.ndarray) -> np.ndarray:
        # solve at a single kpoints
        h = self.Hijk(k)
        #w, v = np.linalg.eig(h)
        s = self.Sijk(k)
        w, c = solve_secular_sorted(h, s)
        return w.real, c

    def solveE_c2_at_ks(self, ks: np.ndarray) -> np.ndarray:
        ek = []
        coeff = []
        for k in ks:
            w, complex_c =self.solveE_at_k(k)
            ek.append(w)
            c_T = (np.abs(complex_c)**2).T # [ibnd, iorb]
            coeff.append(c_T[np.newaxis,...])
        ek = np.vstack(ek)
        coeff = np.vstack(coeff)
        assert coeff.shape[0] == ek.shape[0]
        return ek, coeff

    def solveE_at_ks(self, ks: np.ndarray) -> np.ndarray:
        return np.vstack(
            [ self.solveE_at_k(k)[0] for k in ks ]
        )


class TightBindingModel(TightBindingBase):
    def __init__(self, 
        cell: np.ndarray,
        positions: np.ndarray,
        types: typing.List[int],
        HijR_list: typing.List[HijR],
        SijR_list: typing.List[SijR]
    ) -> None:
        self._cell = cell
        self._positions = positions
        self._types = types
        self._HijR = HijR_list
        self._SijR = SijR_list

        self._basis: typing.List[Pindex_lm] = []
        for hijR in self._HijR:
            if hijR.left not in self._basis:
                self._basis.append(hijR.left)
                
        self._index_ref: typing.Dict[tuple, int] = {
            (basis.pindex, basis.l, basis.m):i for i, basis in enumerate(self._basis)
        }

    @property
    def basis(self) -> typing.List[Pindex_lm]:
        return self._basis
    
    @property
    def cell(self) -> np.ndarray:
        return self._cell

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def types(self) -> typing.List[int]:
        return self._types
        
    def Hijk(self, k: typing.Tuple[float]):
        k = np.array(k)
        nbasis = len(self._basis)
        ham = np.zeros((nbasis,nbasis), dtype=complex)
        for HijR in self._HijR:
            assert np.allclose(HijR.left.translation, np.zeros(3))
            r = HijR.right.translation
            index_i = self._index_ref[
                (HijR.left.pindex, HijR.left.l, HijR.left.m)
            ]
            index_j = self._index_ref[
                (HijR.right.pindex, HijR.right.l, HijR.right.m)
            ]
            kR = np.dot(k, r + self._positions[HijR.right.pindex] - self._positions[HijR.left.pindex])
            ham[index_i,index_j] += HijR.value * np.exp(2j * np.pi * kR)
        return ham

    def Sijk(self, k: typing.Tuple[float]):
        k = np.array(k)
        nbasis = len(self._basis)
        ham = np.zeros((nbasis,nbasis), dtype=complex)
        for HijR in self._SijR:
            assert np.allclose(HijR.left.translation, np.zeros(3))
            r = HijR.right.translation
            index_i = self._index_ref[
                (HijR.left.pindex, HijR.left.l, HijR.left.m)
            ]
            index_j = self._index_ref[
                (HijR.right.pindex, HijR.right.l, HijR.right.m)
            ]
            kR = np.dot(k, r + self._positions[HijR.right.pindex] - self._positions[HijR.left.pindex])
            ham[index_i,index_j] += HijR.value * np.exp(2j * np.pi * kR)
        return ham