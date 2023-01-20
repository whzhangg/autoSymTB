import typing, dataclasses
import numpy as np

from automaticTB.tools import Pair, atomic_numbers, parse_orbital
from automaticTB.parameters import zero_tolerance
from automaticTB.solve.SALCs import NamedLC

__all__ = ["AO", "AOPair", "AOPairWithValue", "AOSubspace"]


@dataclasses.dataclass
class AO:
    """defines an atomic orbital in a molecular"""
    equivalent_index: int
    primitive_index: int
    absolute_position: np.ndarray
    translation: np.ndarray
    chemical_symbol: str
    n: int
    l: int
    m: int

    @property
    def atomic_number(self) -> int:
        return atomic_numbers[self.chemical_symbol]

    def __eq__(self, o: "AO") -> bool:
        return  self.primitive_index == o.primitive_index and \
                self.l == o.l and \
                self.m == o.m and \
                self.n == o.n and \
                np.allclose(self.translation, o.translation, zero_tolerance)

    def __repr__(self) -> str:
        result = f"{self.chemical_symbol:>2s} (i_p={self.primitive_index:>2d})"
        result+= f" n ={self.n:>2d}; l ={self.l:>2d}; m ={self.m:>2d}"
        result+=  " @({:>5.1f},{:>5.1f},{:>5.1f})".format(*self.translation)
        return result

    def as_tuple(self) -> tuple:
        t1 = int(self.translation[0])
        t2 = int(self.translation[1])
        t3 = int(self.translation[2])
        return (self.primitive_index, t1, t2, t3, self.n, self.l, self.m)


    def __hash__(self) -> int:
        return hash(
            self.as_tuple()
        )


@dataclasses.dataclass
class AOPair:
    """
    just a pair of AO that enable comparison, AOs can be compared directly
    """
    l_AO: AO
    r_AO: AO

    @property
    def left(self) -> AO:
        return self.l_AO
    
    @property
    def right(self) -> AO:
        return self.r_AO

    @property
    def vector_right_to_left(self) -> np.ndarray:
        return self.l_AO.absolute_position - self.r_AO.absolute_position

    @classmethod
    def from_pair(cls, aopair: Pair) -> "AOPair":
        return cls(aopair.left, aopair.right)

    def as_pair(self) -> Pair:
        return Pair(self.l_AO, self.r_AO)

    def strict_match(self, o: "AOPair") -> bool:
        return  self.l_AO == o.l_AO and self.r_AO == o.r_AO

    def __eq__(self, o: "AOPair") -> bool:
        return self.l_AO == o.l_AO and self.r_AO == o.r_AO
        #return  (self.l_AO == o.l_AO and self.r_AO == o.r_AO) or \
        #        (self.l_AO == o.r_AO and self.r_AO == o.l_AO)

        def check_ao(ao1: AO, ao2: AO) -> bool:
            """
            the same ao if orbital is the same and both belong to the same equivalent atoms
            """
            return ao1.equivalent_index == ao2.equivalent_index and \
                   ao1.l == ao2.l and ao1.m == ao2.m and ao1.n == ao2.n

        # case 1: the same sequence
        vector_same = np.allclose(
            self.vector_right_to_left, o.vector_right_to_left, 
            atol = zero_tolerance
        )
        ao_same = check_ao(self.l_AO, o.l_AO) and check_ao(self.r_AO, o.r_AO)
        if vector_same and ao_same: return True

        vector_reverse = np.allclose(
            self.vector_right_to_left, -1.0 * o.vector_right_to_left, 
            atol = zero_tolerance
        )
        ao_reverse = check_ao(self.l_AO, o.r_AO) and check_ao(self.r_AO, o.l_AO)
        if vector_reverse and ao_reverse: return True

        return False


    def __str__(self) -> str:
        result = " > Pair: "
        left = self.l_AO; right = self.r_AO
        rij = right.absolute_position - left.absolute_position
        result += f"{left.chemical_symbol:>2s}-{left.primitive_index:0>2d} " 
        result += f"{parse_orbital(left.n, left.l, left.m):>7s} -> "
        result += f"{right.chemical_symbol:>2s}-{right.primitive_index:0>2d} "
        result += f"{parse_orbital(right.n, right.l, right.m):>7s} "
        result += "r = ({:>6.2f},{:>6.2f},{:>6.2f})".format(*rij)
        #result += " t = ({:>3d}{:>3d}{:>3d})".format(*np.array(right.translation, dtype=int))
        return result

    def __hash__(self) -> int:
        return hash(
            self.l_AO.as_tuple() + self.r_AO.as_tuple()
        )

@dataclasses.dataclass
class AOPairWithValue(AOPair):
    """AOPair with value additionally associated with it, overwrite the __eq__ method"""
    value: typing.Union[float, complex]

    def __eq__(self, o: "AOPairWithValue") -> bool:
        pair_OK = super()
        valueOK = np.isclose(self.value, o.value, atol = zero_tolerance)
        pair_OK = (self.l_AO == o.l_AO and self.r_AO == o.r_AO) or \
                  (self.l_AO == o.r_AO and self.r_AO == o.l_AO)
        
        return  valueOK and pair_OK

    def __str__(self) -> str:
        result = " > Pair: "
        left = self.l_AO; right = self.r_AO
        rij = right.absolute_position - left.absolute_position
        result += f"{left.chemical_symbol:>2s}-{left.primitive_index:0>2d} " 
        result += f"{parse_orbital(left.n, left.l, left.m):>7s} -> "
        result += f"{right.chemical_symbol:>2s}-{right.primitive_index:0>2d} "
        result += f"{parse_orbital(right.n, right.l, right.m):>7s} "
        result += "@ ({:>6.2f},{:>6.2f},{:>6.2f}) ".format(*rij)
        result += f"Hij = {self.value:>.4f}"
        return result

    
@dataclasses.dataclass
class AOSubspace:
    """Defines AOs and a list of their linear combination with different representation
    
    this object is simply a container of a set of named linear combinations with basis specified by aos
    """
    aos: typing.List[AO]
    namedlcs: typing.List[NamedLC] # only the components belonging to the AOs

    def __repr__(self) -> str:
        result = ["subspace consists of:"]
        for ao in self.aos:
            result.append(f" {str(ao)}")
        result.append("With linear combinations")
        for nlc in self.namedlcs:
            result.append(f" {str(nlc.name)}")
        return "\n".join(result)

