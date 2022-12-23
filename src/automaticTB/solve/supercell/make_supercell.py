from ..interaction import AO, InteractingAOSubspace, AOSubspace
import dataclasses, numpy as np, typing
from .sc_tools import make_supercell_using_ase
from ...parameters import precision_decimal, tolerance_structure


def get_index_translation_for_positions(
    check_frac_pos: np.ndarray, reference_frac_pos: np.ndarray
) -> typing.List[typing.Tuple[int, typing.Tuple[int, int, int]]]:
    """
    for provided atoms, we get the translation as well as the index in the 
    home unit cell as given by reference_frac_pos
    """
    p_index = []
    zero3 = np.zeros(3)
    for to_check in check_frac_pos:
        found = False
        difference = reference_frac_pos - to_check
        # the point is the same if their difference is almost int
        fractional_part = difference - np.rint(difference)
        for iref, frac in enumerate(fractional_part):
            if np.allclose(frac, zero3, atol=tolerance_structure):
                found = True
                p_index.append(iref)
                break
        
        if not found:
            print("the position to check:")
            print(to_check)
            print("possible positions")
            print(reference_frac_pos)
            raise RuntimeError("position checking is not successful")
    
    p_index_array = np.array(p_index, dtype=int)

    translation = check_frac_pos - reference_frac_pos[p_index_array]
    return [ 
        (int(h), (int(t[0]), int(t[1]), int(t[2])))
        for h, t in zip(p_index_array, translation) 
    ]


class SupercellTransformation:
    def __init__(self,
        pcell: np.ndarray,
        pfrac: np.ndarray,
        scell: np.ndarray,
        sfrac: np.ndarray
    ) -> None:
        
        self.pcell = pcell 
        self.pfrac = pfrac
        self.scell = scell
        self.sfrac = sfrac
        
        self.inv_pcell_T = np.linalg.inv(self.pcell.T)
        self.inv_scell_T = np.linalg.inv(self.scell.T)
        
    def translate_AO(self, ao_in: AO, pcell_translation: typing.Tuple[int, int, int]) -> AO:
        abs_translation = np.dot(self.pcell.T, np.array(pcell_translation))
        new_abs_position = ao_in.absolute_position + abs_translation

        new_fraction = np.dot(self.inv_scell_T, new_abs_position) # fractional_position_in_SC
        s_index, s_translation = get_index_translation_for_positions(
            np.array([new_fraction]), self.sfrac
        )[0]
        
        return AO(
            cluster_index= ao_in.cluster_index,
            equivalent_index= ao_in.equivalent_index,
            primitive_index= s_index,
            absolute_position= new_abs_position,
            translation= s_translation,
            chemical_symbol= ao_in.chemical_symbol,
            n = ao_in.n,
            l = ao_in.l,
            m = ao_in.m
        )


@dataclasses.dataclass
class StructureAndEquation:
    cell: np.ndarray
    positions: np.ndarray
    types: np.ndarray
    iaosubspaces: typing.List[InteractingAOSubspace]


    def generate_supercell(self, supercell_matrix: np.ndarray) -> "StructureAndEquation":
        s_cell, s_positions, s_types = make_supercell_using_ase(
            (self.cell, self.positions, self.types), supercell_matrix
        )
        s_positions = np.around(s_positions, decimals = precision_decimal)
        s_positions = s_positions - np.floor(s_positions)
        # now the s_cell, s_positions should not be changed
        
        # sanity check
        if not np.allclose(s_cell, np.dot(supercell_matrix, self.cell), atol=tolerance_structure):
            raise RuntimeError("supercell transformation in ase is wrong")

        # the primitive index and primitive translation of atoms in the supercell
        index_translation = get_index_translation_for_positions(
            np.einsum("ji,kj -> ki", supercell_matrix, s_positions), 
            self.positions
        )

        # mapping from primitive cell index to the corresponding list of InteractingAOSubspace
        organized_iaosubspace: typing.Dict[int, typing.List[InteractingAOSubspace]] = {}
        for iaosub in self.iaosubspaces:
            center_index = iaosub.l_subspace.aos[0].primitive_index
            organized_iaosubspace.setdefault(center_index, []).append(iaosub)

        # transformer for AO
        transformer = SupercellTransformation(
            self.cell, self.positions,
            s_cell, s_positions
        )

        new_subspaces = []
        for p_index, t in index_translation:
            for iaosub in organized_iaosubspace[p_index]:
                # iterate interaction subspace centered on it
                l_sub = iaosub.l_subspace
                r_sub = iaosub.r_subspace
                new_l_ao = [ transformer.translate_AO(ao, t) for ao in l_sub.aos ]
                new_r_ao = [ transformer.translate_AO(ao, t) for ao in r_sub.aos ]
                new_subspaces.append(
                    InteractingAOSubspace(
                        l_subspace=AOSubspace(new_l_ao, l_sub.namedlcs),
                        r_subspace=AOSubspace(new_r_ao, r_sub.namedlcs),
                    )
                )

        return StructureAndEquation(s_cell, s_positions, s_types, new_subspaces)


        