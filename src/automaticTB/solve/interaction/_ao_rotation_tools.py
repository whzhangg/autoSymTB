import typing
import numpy as np

from automaticTB.parameters import complex_coefficient_type, tolerance_structure
from automaticTB.solve.atomic_orbitals import Orbitals
from automaticTB.solve.rotation import orbital_rotation_from_symmetry_matrix
from .interaction_pairs import AO, AOPair

class Position:
    """Position info. for AO
    
    Contains all the position information of an AO class: i.e., except n, l, m attributes
    """
    def __init__(self,
        equivalent_index: int,
        primitive_index: int,
        absolute_position: np.ndarray,
        translation: np.ndarray,
        chemical_symbol: str
    ) -> None:
        self.equivalent_index = equivalent_index
        self.primitive_index = primitive_index
        self.absolute_position = absolute_position
        self.translation = translation
        self.chemical_symbol = chemical_symbol

    def __eq__(self, o: "AO") -> bool:
        return  self.primitive_index == o.primitive_index and \
                np.allclose(self.translation, o.translation, tolerance_structure)

    @classmethod
    def from_AO(cls, ao: AO) -> "Position":
        return Position(
                    equivalent_index=ao.equivalent_index,
                    primitive_index=ao.primitive_index,
                    absolute_position=ao.absolute_position,
                    translation=ao.translation,
                    chemical_symbol=ao.chemical_symbol
                )

class AOpairRotater:
    """Rotate an AO, return rotated coefficient under `all_ao_pairs` input
    
    it returns complex coefficient of rotated AO
    """
    def __init__(self, 
        all_ao_pairs: typing.List[AOPair]
    ) -> None:
        self.all_ao_pairs = all_ao_pairs
        self.unique_positions: typing.List[Position] = []
        found = []
        for ao_pair in all_ao_pairs:
            for f in found:
                if np.allclose(ao_pair.l_AO.absolute_position, f, atol=tolerance_structure):
                    break
            else:
                found.append(ao_pair.l_AO.absolute_position)
                self.unique_positions.append(
                    Position.from_AO(ao_pair.l_AO)
                )
            for f in found:
                if np.allclose(ao_pair.r_AO.absolute_position, f, atol=tolerance_structure):
                    break
            else:
                found.append(ao_pair.r_AO.absolute_position)
                self.unique_positions.append(
                    Position.from_AO(ao_pair.r_AO)
                )
        
    def rotate_aopair(
        self, ao_pair: AOPair, rotation: np.ndarray, translation: np.ndarray, 
        print_debug: bool = False
    ) -> np.ndarray:
        
        if print_debug:
            print("rot: ", ("{:>6.2f}"*9).format(*rotation.flatten()))
            print("> input AOPair" + str(ao_pair))
        l_ao = ao_pair.l_AO
        r_ao = ao_pair.r_AO
        
        dr = r_ao.absolute_position - l_ao.absolute_position
        new_r_absolute_position = np.dot(rotation, dr) + l_ao.absolute_position + translation

        new_r_Position = None
        for upos in self.unique_positions:
            if np.allclose(new_r_absolute_position, upos.absolute_position, atol=tolerance_structure):
                new_r_Position = upos
                break
        else:
            raise RuntimeError("AOpairRotator: cannot identify the rotated positions")
        
        new_l_absolute_position = l_ao.absolute_position + translation
        new_l_Position = None
        for upos in self.unique_positions:
            if np.allclose(new_l_absolute_position, upos.absolute_position, atol = tolerance_structure):
                new_l_Position = upos
                break
        else:
            raise RuntimeError("AOpairRotator: cannot identify the rotated positions")

        l_rotated_results = self.rotate_ao_simple(l_ao, rotation)
        r_rotated_results = self.rotate_ao_simple(r_ao, rotation)

        result_coefficients = np.zeros(len(self.all_ao_pairs), dtype=complex_coefficient_type)
        for ln,ll,lm,lcoe in l_rotated_results:
            for rn,rl,rm,rcoe in r_rotated_results:
                l_ao = AO(
                    equivalent_index=new_l_Position.equivalent_index,
                    primitive_index=new_l_Position.primitive_index,
                    absolute_position=new_l_Position.absolute_position,
                    translation=new_l_Position.translation,
                    chemical_symbol=new_l_Position.chemical_symbol,
                    n = ln, l = ll, m = lm
                )
                r_ao = AO(
                    equivalent_index=new_r_Position.equivalent_index,
                    primitive_index=new_r_Position.primitive_index,
                    absolute_position=new_r_Position.absolute_position,
                    translation=new_r_Position.translation,
                    chemical_symbol=new_r_Position.chemical_symbol,
                    n = rn, l = rl, m = rm
                )
                if print_debug and np.abs(np.conjugate(lcoe) * rcoe) > 1e-6:
                    print(">> output AOPair" + str(AOPair(l_ao, r_ao)) \
                        + f" coefficient: {lcoe.real:>8.4f}, {rcoe.real:>8.4f}")

                for i, pair in enumerate(self.all_ao_pairs):
                    if l_ao == pair.l_AO and r_ao == pair.r_AO:
                        result_coefficients[i] = np.conjugate(lcoe) * rcoe
                        break
        
        return result_coefficients

    @staticmethod
    def rotate_ao_simple(ao, cartesian_rotation):
        orbitals = Orbitals([(ao.n,ao.l)])
        coefficient = np.zeros(len(orbitals.sh_list), dtype = complex_coefficient_type)
        for i, sh in enumerate(orbitals.sh_list):
            if sh.n == ao.n and sh.l == ao.l and sh.m == ao.m:
                coefficient[i] = 1.0
                break
        
        orb_rotation = orbital_rotation_from_symmetry_matrix(
            cartesian_rotation, orbitals.irreps_str
        )
        rotated_coefficient = np.dot(orb_rotation, coefficient)

        result = []
        for i, sh in enumerate(orbitals.sh_list):
            result.append((sh.n, sh.l, sh.m, rotated_coefficient[i]))
        return result