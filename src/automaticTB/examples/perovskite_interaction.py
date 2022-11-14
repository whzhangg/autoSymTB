import numpy as np
import typing
from automaticTB.solve.interaction import AO
from automaticTB.tools import Pair

"""
this module create AO interactions matrix from the parameter used in the article. 
In the parameters, index 0 indicate the Pb atom and index 1-3 indicate Cl atoms, the minimum parameters 
are stored here.
"""

__all__ = [
    "get_interaction_values_from_list_AOpairs",
    "get_overlap_values_from_list_AOpairs"    
]

parameters = {
        "V_ss"   :  -1.10,
        "V_s0p1" :   1.25,
        "V_p0s1" :   0.60,
        "V_pps"  :  -3.50,
        "V_ppp"  :   0.80,
        "E_s0"   :  -9.21,
        "E_s1"   : -13.21,
        "E_p0"   :   2.39,
        "E_p1"   :  -2.16
    }

p_direction = {
    -1: np.array([0.0, 1.0, 0.0]), # py
     0: np.array([0.0, 0.0, 1.0]), # pz
     1: np.array([1.0, 0.0, 0.0])  # px
     # the direction of p orbital determines the p-p sigma or pi bond, as well as the interaction between s and p bond
}

def get_interaction_values_from_list_AOpairs(
    cell: np.ndarray, positions: np.ndarray, aopairs: typing.List[Pair]
) -> typing.List[float]:
    # this is an ad-hoc way to create the AO interaction
    cpos = np.einsum("ji, kj -> ki", cell, positions)

    values = []
    for pair in aopairs:
        left: AO = pair.left
        right: AO = pair.right
        value = 0

        if (left.cluster_index == right.cluster_index) \
            and (left.l == right.l) and (left.m == right.m):
            # self interaction
            symbol = "E_" + ["s", "p", "d"][left.l]
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
                if right.l == 1:
                    m = right.m
                else:
                    m = left.m
                if right.chemical_symbol == "Pb":
                    d_r = -1 * d_r  # dr always start from Pb
                p1_direction = p_direction[m]
                if abs(np.dot(p1_direction, d_r)) < 1e-6:
                    """when p is perpendicular to dr"""
                    value = 0.0

                else:
                    sign = np.sign(np.dot(p1_direction, d_r))

                    if (left.l == 0 and left.chemical_symbol == "Pb") or \
                       (right.l == 0 and right.chemical_symbol == "Pb"):
                        symbol = "V_s0p1"
                    elif (left.l == 1 and left.chemical_symbol == "Pb") or \
                        (right.l == 1 and right.chemical_symbol == "Pb"):
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
            
        values.append(value)
    
    return values


def get_overlap_values_from_list_AOpairs(aopairs: typing.List[Pair]) -> typing.List[float]:
    values = []
    for pair in aopairs:
        left: AO = pair.left
        right: AO = pair.right
        value = 0

        if (left.cluster_index == right.cluster_index) \
            and (left.l == right.l) and (left.m == right.m):
            value = 1.0

        values.append(value)
    
    return values