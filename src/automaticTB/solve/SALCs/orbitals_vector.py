"""provide functionality to rotate a vector inside a vectorspace"""
import typing
import dataclasses

import numpy as np

from automaticTB import parameters as params


def rotate_orbitalvector(
    orbv: "OrbitalsVector", cart_rot: np.ndarray, orb_rot: np.ndarray
) -> "OrbitalsVector":
    rot44 = np.eye(4)
    rot44[1:,1:] = cart_rot
    rotated_pos = np.einsum("ij,kj->ki", rot44, orbv.tpos4)
    rotated_orbitals = orb_rot @ orbv.coefficients
    return OrbitalsVector(rotated_pos, rotated_orbitals, orbv.atom_slice)


def symrotate_orbitalvector(
    orbv: "OrbitalsVector", cart_rot: np.ndarray, orb_rot: np.ndarray, tol: float
) -> "OrbitalsVector":
    """rotation but with reordered site"""
    rot44 = np.eye(4)
    rot44[1:,1:] = cart_rot
    tpos = orbv.tpos4

    from_which = np.zeros(len(tpos), dtype=np.intc)
    for i0, pos0 in enumerate(tpos):
        new_pos = np.dot(rot44, pos0)
        for i1, pos1 in enumerate(tpos):
            diff = new_pos - pos1
            if diff[0]**2 + diff[1]**2 + diff[2]**2 < tol:
                from_which[i1] = i0
                break
        else:
            raise RuntimeError("cannot find the rotated atoms")
    
    new_coefficients = orb_rot @ orbv.coefficients
    ordered_coeff = np.zeros_like(new_coefficients)
    coe_slice = orbv.atom_slice
    for current_pos, origin_pos in enumerate(from_which):
        ordered_coeff[coe_slice[current_pos]] = new_coefficients[coe_slice[origin_pos]]

    return OrbitalsVector(orbv.tpos4, orbv.atom_slice, ordered_coeff)


@dataclasses.dataclass
class OrbitalsVector:
    """a vector in the vector space for fast operation"""
    tpos4: np.ndarray
    atom_slice: typing.List[slice]
    coefficients: np.ndarray

    @property
    def norm(self) -> float:
        return np.linalg.norm(self.coefficients)

    @property
    def nonzero(self) -> bool:
        return np.any(np.abs(self.coefficients) > params.ztol)

    def create_new_with_coefficients(self, coefficients: np.ndarray) -> "OrbitalsVector":
        """obtain a new LC with the given coefficients"""
        assert len(coefficients) == len(self.coefficients)
        assert coefficients.dtype == self.coefficients.dtype
        return OrbitalsVector(self.tpos4, self.atom_slice, coefficients)


    def get_normalized(self) -> "OrbitalsVector":
        """return a normalized LC"""
        coefficients = self.coefficients
        norm = np.linalg.norm(self.coefficients)
        if norm > params.ztol:
            coefficients /= norm
        return OrbitalsVector(self.tpos4, self.atom_slice, coefficients)


    def __repr__(self) -> str:
        result = [f"Vector composed of {len(self.coefficients)} atomic orbitals"]
        for tp, aslice in zip(self.tpos4, self.atom_slice):
            n = aslice.stop - aslice.start
            type_ = tp[0]
            pos = tp[1:]
            line = f"atomic number {type_:>3d} with {n:>2d} orbitals " 
            line+= "@ ({:>8.3f}{:>8.3f}{:>8.3f}) with coefficients: ".format(*pos)
            line+= ",".join("{:>16.3f}".format(c) for c in self.coefficients[aslice])
            result.append(line)
        return "\n".join(result)

    '''
    @property
    def atom_slice(self) -> typing.List[slice]:
        start = 0
        result = []
        for n in self.norbs:
            end = start + n
            result.append(slice(start, end))
            start = end
        return result
    '''