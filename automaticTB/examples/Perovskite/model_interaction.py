# 0 for Pb atom, 1 - 3 for Cl atoms
import numpy as np
import typing
from automaticTB.hamiltionian.MOcoefficient import InteractionMatrix, AO

parameters = {
        "V_ss"   :  -1.10,
        "V_s0p1" :   1.10,
        "V_p0s1" :   0.70,
        "V_pps"  :  -3.65,
        "V_ppp"  :   0.55,
        "E_s0"   :  -9.01,
        "E_s1"   : -13.01,
        "E_p0"   :   2.34,
        "E_p1"   :  -1.96
    }

l_to_symbol = ["s", "p", "d"]
p_direction = {
    -1: np.array([0.0, 1.0, 0.0]),
     0: np.array([0.0, 0.0, 1.0]),
     1: np.array([1.0, 0.0, 0.0])
}

def obtain_AO_interaction_from_AOlists(cell: np.ndarray, positions: np.ndarray, aolist: typing.List[AO]) -> InteractionMatrix:
    cpos = np.einsum("ji, kj -> ki", cell, positions)
    interaction = InteractionMatrix.zero_from_states(aolist)
    for pair in interaction.flattened_braket:
        left: AO = pair.bra
        right: AO = pair.ket
        value = 0
        if (left.cluster_index == right.cluster_index) \
            and (left.l == right.l) and (left.m == right.m):
            # self interaction
            symbol = "E_" + l_to_symbol[left.l]
            if left.chemical_symbol == "Pb":
                symbol += "0"
            else: 
                symbol += "1"
            value = parameters[symbol]

        
        elif left.chemical_symbol != right.chemical_symbol:
            left_position = cpos[left.primitive_index] + cell.T.dot(left.translation)
            right_position = cpos[right.primitive_index] + cell.T.dot(right.translation)
            d_r = right_position - left_position

            if (left.l == right.l) and (left.l == 0):
                # both s
                value = parameters["V_ss"]
            elif left.l != right.l:
                # s and p
                if left.l == 1:
                    m = left.m
                else:
                    m = right.m
                    d_r = -1 * d_r # direction from l=1 to l=0
                p1_direction = p_direction[m]
                if abs(np.dot(p1_direction, d_r)) < 1e-6:
                    value = 0.0

                else:
                    sign = np.sign(np.dot(p1_direction, d_r))
                    if (left.l == 0 and left.chemical_symbol == "Pb") \
                    or (right.l == 0 and right.chemical_symbol == "Pb"):
                        symbol = "V_s0p1"
                    else:
                        symbol = "V_p0s1"
                    value = sign * parameters[symbol]
            
            else: # left.1 
                # both p
                p1_direction = p_direction[left.m]
                p2_direction = p_direction[right.m]
                p1_parallel_p2 = abs(np.dot(p1_direction, p2_direction)) > 1e-6

                if p1_parallel_p2:
                    p1_parallel_dr = abs(np.dot(p1_direction, d_r)) > 1e-6
                    if p1_parallel_dr:
                        value = parameters["V_pps"]
                    else:
                        value = parameters["V_ppp"]
                else:
                    value = 0.0
            
        #print(left.chemical_symbol, left.primitive_index, left.l, left.m, right.chemical_symbol, right.primitive_index, right.l, right.m, value)
        indices = interaction.get_index(left, right)
        interaction.interactions[indices] = value
    #print(interaction.states)
    return interaction
