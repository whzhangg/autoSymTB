from .symmetry_elements import SymDirection, Symmetries_on_Direction_Set
import typing, abc, re
import numpy as np

def get_vector_from_hkl(hkl: str) -> np.ndarray:
    parts = re.findall(r"-?\d", hkl)
    return np.array(parts, dtype=float)

def is_righthand(x, y, z) -> bool:
    cross = np.cross(x, y)
    return np.dot(cross, z) > 0.0

class AxisFinder(abc.ABC):

    @abc.abstractmethod
    def find_hkl_from_direction(self, input_direction: SymDirection) -> str:
        pass

    @abc.abstractmethod
    def find_direction_from_hkl(self, hkl: str) -> np.ndarray:
        pass

    
class OriginFinder(AxisFinder):
    possible_value = ["000"]

    def __init__(self) -> None:
        pass

    def find_hkl_from_direction(self, input_direction: SymDirection) -> str:
        if np.zeros(3) in input_direction:
            return "000"
        raise ValueError("OriginFinder, input vector is problematic")
        

    def find_direction_from_hkl(self, hkl: str) -> np.ndarray:
        if hkl in self.possible_value:
            return np.zeros(3)
        raise ValueError("OriginFinder, input hkl is problematic")
    

class ZAxisFinder(AxisFinder):
    possible_value = ["000", "001"]

    def __init__(self, z: np.ndarray) -> None:
        self._zdirection = z

    def find_hkl_from_direction(self, input_direction: SymDirection) -> str:
        if np.zeros(3) in input_direction:
            return "000"
        elif self._zdirection in input_direction:
            return "001"
        else:
            raise ValueError("ZAxisFinder, input vector is problematic")

    def find_direction_from_hkl(self, hkl: str) -> np.ndarray:
        if hkl == "000":
            return np.zeros(3)
        elif hkl == "001":
            return self._zdirection
        else:
            raise ValueError("ZAxisFinder, input hkl is problematic")


class CubicFinder(AxisFinder):
    possible_value = [
        "000", "100", "010", "001", 
        "-1-11", "1-1-1", "-11-1", "111",
        "01-1", "1-10", "-101", "110", "011", "101"]
    # we generate all the directions and then compare the input direction
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self._xyz = np.vstack([x, y, z]).T
        self._directions = []
        for hkl in self.possible_value:
            v = get_vector_from_hkl(hkl)
            self._directions.append(np.dot(self._xyz, v))
    
    def find_hkl_from_direction(self, input_direction: SymDirection) -> str:
        for direction, hkl in zip(self._directions, self.possible_value):
            if direction in input_direction:
                return hkl
        raise ValueError(f"CubicFinder, input vector {input_direction} is problematic")

    def find_direction_from_hkl(self, hkl: str) -> np.ndarray:
        if hkl in self.possible_value:
            i = self.possible_value.index(hkl)
            return self._directions[i]
        else:
            raise ValueError(f"CubicFinder, input hkl {hkl} is problematic")


class HexagonalFinder(AxisFinder):
    possible_value = [
        "000", "001", "100", "010", "110", "120", "210", "1-10"
    ]
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self._xyz = np.vstack([x, y, z]).T
        self._directions = []
        for hkl in self.possible_value:
            v = get_vector_from_hkl(hkl)
            self._directions.append(np.dot(self._xyz, v))
    
    def find_hkl_from_direction(self, input_direction: SymDirection) -> str:
        for direction, hkl in zip(self._directions, self.possible_value):
            if direction in input_direction:
                return hkl
        raise ValueError(f"HexagonalFinder, input vector {input_direction} is problematic")

    def find_direction_from_hkl(self, hkl: str) -> np.ndarray:
        if hkl in self.possible_value:
            i = self.possible_value.index(hkl)
            return self._directions[i]
        else:
            raise ValueError(f"HexagonalFinder, input hkl {hkl} is problematic")


def get_finder(
    groupname: str, 
    directions: typing.Dict[str, Symmetries_on_Direction_Set]
) -> AxisFinder:
    if groupname in ["1", "-1"]:
        return OriginFinder()

    elif groupname in ["2", "m", "2/m", "4", "-4", "4/m", "3", "-3", "6", "-6", "6/m"]:
        z = directions["primiary"].directions[0].vector
        return ZAxisFinder(z)

    elif groupname in ["222", "mm2", "mmm"]:
        x = directions["primiary"].directions[0].vector
        y = directions["secondary"].directions[0].vector
        z = directions["tertiary"].directions[0].vector
        if not is_righthand(x, y, z):
            z = -1 * z
        return CubicFinder(x, y, z)

    elif groupname in ["422", "4mm", "-42m", "4/mmm"]:
        #tetragonal
        z = directions["primiary"].directions[0].vector
        x = directions["secondary"].directions[0].vector
        y = directions["secondary"].directions[1].vector
        if not is_righthand(x, y, z):
            z = -1 * z
        return CubicFinder(x, y, z)

    elif groupname in ["32", "3m", "-3m", "622", "6mm", "-6m2", "6/mmm"]:
        # hexagonal
        z = directions["primiary"].directions[0].vector
        x = directions["secondary"].directions[0].vector
        y = directions["secondary"].directions[1].vector
        # make sure that x, y axis have angle 120 degree
        if np.dot(x, y) > 0:
            y = -1 * y
        if not is_righthand(x, y, z):
            z = -1 * z
        return HexagonalFinder(x, y, z)

    elif groupname in ["23", "m-3", "432", "-43m", "m-3m"]:
        #cubic
        x = directions["primiary"].directions[0].vector
        y = directions["primiary"].directions[1].vector
        z = directions["primiary"].directions[2].vector
        if not is_righthand(x, y, z):
            z = -1 * z
        return CubicFinder(x, y, z)

    else:
        # this cannot happen
        raise ValueError("get_finder, groupname is not included")

reference = """
# origin only -----------------------------------
       1 : 000
      -1 : 000
# z axis only -----------------------------------
       2 : 000 001
       m : 000 001
     2/m : 000 001
       4 : 000 001
      -4 : 000 001
     4/m : 000 001
       3 : 000 001
      -3 : 000 001
       6 : 000 001
      -6 : 000 001
     6/m : 000 001
# orthorhombic ----------------------------------
     222 : 100 000 001 010
     mm2 : 100 000 001 010
     mmm : 100 000 001 010
# tetragonal ------------------------------------
     422 : 000 001 100 1-10 110 010
     4mm : 000 001 100 1-10 110 010
    -42m : 000 001 100 1-10 110 010
   4/mmm : 000 001 100 1-10 110 010
# hexagonal -------------------------------------
      32 : 000 001 100 010 110
      3m : 000 001 100 110 010
     -3m : 000 001 100 010 110
     622 : 000 001 120 210 100 1-10 110 010
     6mm : 000 001 120 210 100 1-10 110 010
    -6m2 : 000 001 120 210 100 1-10 110 010
   6/mmm : 000 001 120 210 100 1-10 110 010
# cubic ----------------------------------------- 
      23 : -1-11 1-1-1 000 001 -11-1 100 111 010
     m-3 : -1-11 1-1-1 000 001 -11-1 100 111 010
     432 : 01-1 -1-11 1-1-1 000 001 -11-1 100 1-10 110 111 -101 010 011 101
    -43m : 01-1 -1-11 1-1-1 000 001 -11-1 100 1-10 110 111 -101 010 011 101
    m-3m : 01-1 -1-11 1-1-1 000 001 -11-1 100 1-10 110 111 -101 010 011 101
"""