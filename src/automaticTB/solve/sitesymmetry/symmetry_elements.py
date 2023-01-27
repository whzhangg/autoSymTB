# We attempt to detemint the seitz symbol for the operation and dress them
import typing
import dataclasses

import numpy as np

from automaticTB import parameters as params
from .utilities import operation_type_from_rotation_matrix


class SymDirection:
    # a normalized vector presenting a direction
    def __init__(self, vector: np.ndarray, label: str = "") -> None:
        self.label = label
        self._is_zero:bool = False
        close_to_zero = np.isclose(vector, np.zeros(3), atol = params.stol)
        if all(close_to_zero):
            self._is_zero:bool = True
            self._vector: np.ndarray = vector
            return

        larger = vector > (-1 * vector)
        if larger[0] or (close_to_zero[0] and larger[1]) or \
                        (close_to_zero[0] and close_to_zero[1] and larger[2]):
            self._vector: np.ndarray = vector / np.linalg.norm(vector)
        else:
            self._vector: np.ndarray = -1 * vector / np.linalg.norm(vector)

    @property
    def is_zero(self) -> bool:
        return self._is_zero

    @property
    def vector(self) -> np.ndarray:
        return self._vector

    @property
    def opposite(self) -> np.ndarray:
        return -1 * self._vector

    def __eq__(self, other) -> bool:
        if self.is_zero or other.is_zero:
            if self.is_zero and other.is_zero: 
                return True
            else:
                return False
            
        cos_theta = np.dot(self.vector, other.vector) / np.linalg.norm(self.vector) / np.linalg.norm(other.vector)
        return np.isclose(np.abs(cos_theta), 1.0, atol = params.stol)

    def __repr__(self) -> str:
        return "({:>7.3f}{:>7.3f}{:>7.3f})".format(*self._vector)

    def __contains__(self, input_d: np.ndarray) -> bool:
        # if the given vector belong to this direction
        if self.is_zero or np.allclose(input_d, np.zeros(3)): 
            if self.is_zero and np.allclose(input_d, np.zeros(3)):
                return True
            else: 
                return False

        cos_theta = np.dot(self.vector, input_d) / np.linalg.norm(self.vector) / np.linalg.norm(input_d)
        return np.isclose(np.abs(cos_theta), 1.0, atol = params.stol)


class SymDirection_Set:
    # equivalent directions
    def __init__(self, directions: np.ndarray):
        self._directions: typing.List[SymDirection] = []
        for input_direction in directions:
            if any([input_direction in directions for directions in self._directions]):
                continue
            self._directions.append(SymDirection(input_direction))

    def __contains__(self, input_d: np.ndarray) -> bool:
        return any([input_d in directions for directions in self._directions])

    def __repr__(self) -> str:
        direction_str = [
            str(d) for d in self._directions
        ]
        return ", ".join(direction_str)

    @property
    def directions(self) -> typing.List[SymDirection]:
        return self._directions

    @property
    def num_directions(self) -> int:
        return len(self._directions)
  

@dataclasses.dataclass
class SymOp_on_Direction:
    operation: str
    direction: SymDirection
    sense: str
    matrix: np.ndarray

    def change_sense(self):
        if self.sense == "+":
            self.sense = "-"
        elif self.sense == "-":
            self.sense = "+"

    @property
    def rotation_direction(self):
        return self.direction.vector

    def set_new_sense_by_direction(self, new_direction: np.ndarray):
        if self.sense == "": return
        assert new_direction in self.direction
        if np.dot(new_direction, self.direction.vector) < 0:
            self.change_sense()

    @classmethod
    def from_cartesian_rotation_matrix(cls, matrix: np.ndarray):
        if np.allclose(matrix, np.eye(3)):
            return cls("1", SymDirection(np.array([0.0, 0.0, 0.0])), 0,  matrix)
        elif np.allclose(matrix, np.eye(3) * -1.0):
            return cls("-1", SymDirection(np.array([0.0, 0.0, 0.0])), 0, matrix)
        
        operation_name = operation_type_from_rotation_matrix(matrix)
        rotation = matrix
        det = round(np.linalg.det(matrix))
        if np.isclose(det, -1.0, atol = params.stol):
            rotation = np.matmul(rotation, np.eye(3) * -1.0)
        
        w, v = np.linalg.eig(rotation)
        vector_index = np.argwhere(np.isclose(w.real, 1.0, atol = params.stol))[0]
        selected_eigenvector = v[:,vector_index].real.reshape((3,))
        direction = SymDirection(selected_eigenvector)

        if operation_name in ["6","-6","3","-3","4","-4"]:
            n = direction.vector
            N = np.array([[ 0, -n[2], n[1]], 
                        [ n[2], 0, -n[0]],
                        [-n[1], n[0], 0]])
            sin_theta = - np.trace(np.matmul(N, rotation)) / 2

            if sin_theta >= 0:
                sense = "+"
            else:
                sense = "-" 
        else:
            sense = ""

        return cls(operation_name, direction, sense, matrix)
    

    def __repr__(self) -> str:
        result = self.operation 
        direction = "{:>7.3f}{:>7.3f}{:>7.3f}".format(*self.direction.vector)
        return ", ".join([result, direction, self.sense])


@dataclasses.dataclass
class Symmetries_on_Direction:
    # symmetry on the same direction
    symmetries: typing.List[SymOp_on_Direction]

    @property
    def op_set(self) -> typing.Set[str]:
        return { sym.operation for sym in self.symmetries }
    
    @property
    def direction(self) -> SymDirection:
        return self.symmetries[0].direction

    def __repr__(self) -> str:
        operations = " ".join(self.op_set)
        return operations + " @ " + str(self.direction)


@dataclasses.dataclass
class Symmetries_on_Direction_Set:
    SymOnDirections: typing.List[Symmetries_on_Direction]

    @property
    def op_set(self) -> typing.Set[str]:
        return self.SymOnDirections[0].op_set

    @property
    def directions(self) -> typing.List[SymDirection]:
        return [direct.direction for direct in self.SymOnDirections]

    def __repr__(self) -> str:
        operations = " ".join(self.SymOnDirections[0].op_set)
        directions = ", ".join([str(direct.direction) for direct in self.SymOnDirections])
        return operations + " @ " + directions


CrystalAxes: typing.Dict[str, typing.Dict[str, typing.Set[str]]] = {
    "1": {
     },
    "-1": {
     },
    "2": {
        "primiary" : {"2"}
     },
    "m": {
        "primiary" : {"m"}
     },
    "2/m": {
        "primiary" : {"m", "2"}
     },
    "222": {
        "primiary" : {"2"},
        "secondary": {"2"},
        "tertiary" : {"2"},
     },
    "mm2": {
        "primiary" : {"m"},
        "secondary": {"m"},
        "tertiary" : {"2"},
     },
    "mmm": {
        "primiary" : {"m", "2"},
        "secondary": {"m", "2"},
        "tertiary" : {"m", "2"},
     },
    "4": {
        "primiary" : {"2", "4"}
     },
    "-4": {
        "primiary" : {"2", "-4"}
     },
    "4/m": {
        "primiary" : {"m","2","-4","4"}
     },
    "422": {
        "primiary" : {"2","4"},
        "secondary": {"2"},
        "tertiary" : {"2"},
     },
    "4mm": {
        "primiary" : {"2", "4"},
        "secondary": {"m"},
        "tertiary" : {"m"},
     },
    "-42m": {
        "primiary" : {"-4","2"},
        "secondary": {"2"},
        "tertiary" : {"m"},
     },
    "4/mmm": {
        "primiary" : {"m", "2", "-4", "4"},
        "secondary": {"m","2"},
        "tertiary" : {"m","2"},
     },
    "3": {
        "primiary" : {"3"}
     },
    "-3": {
        "primiary" : {"-3", "3"}
     },
    "32": {
        "primiary" : {"3"},
        "secondary": {"2"}
     },
    "3m": {
        "primiary" : {"3"},
        "secondary": {"m"}
     },
    "-3m": {
        "primiary" : {"-3","3"},
        "secondary": {"m", "2"}
     },
    "6": {
        "primiary" : {"6","2","3"}
     },
    "-6": {
        "primiary" : {"m","3","-6"}
     },
    "6/m": {
        "primiary" : {"6", "-3", "-6", "m", "2", "3"}
     },
    "622": {
        "primiary" : {"6","2","3"},
        "secondary": {"2"},
        "tertiary" : {"2"},
     },
    "6mm": {
        "primiary" : {"6","2","3"},
        "secondary": {"m"},
        "tertiary" : {"m"},
     },
    "-6m2": {
        "primiary" : {"m", "3", "-6"},
        "secondary": {"m"},
        "tertiary" : {"2"},
     },
    "6/mmm": {
        "primiary" : {"6", "-3", "-6", "m", "2", "3"},
        "secondary": {"m", "2"},
        "tertiary" : {"m", "2"},
     },
    "23": {
        "primiary" : {"2"},
        "secondary": {"3"},
     },
    "m-3": {
        "primiary" : {"m", "2"},
        "secondary": {"-3", "3"}
     },
    "432": {
        "primiary" : {"2", "4"},
        "secondary": {"3"},
        "tertiary" : {"2"},
     },
    "-43m": {
        "primiary" : {"-4","2"},
        "secondary": {"3"},
        "tertiary" : {"m"},
     },
    "m-3m": {
        "primiary" : {"m","2","-4","4"},
        "secondary": {"-3", "3"},
        "tertiary" : {"m", "2"},
     }
}
