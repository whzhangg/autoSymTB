import dataclasses, typing, collections
from .MOcoefficient import MOCoefficient, InteractingLCPair
from .interaction import Interaction
import numpy as np


class Index_lm(typing.NamedTuple):
    i: int
    l: int
    m: int

def get_interaction_matrix_from_MO_and_interaction(mocoeff: MOCoefficient, interaction: Interaction) \
-> typing.Tuple[ typing.List[Index_lm], np.ndarray ]:
    mointeractions = []
    for i in range(mocoeff.num_MOpairs):
        pair: InteractingLCPair = mocoeff.get_paired_lc_with_name_by_index(i)
        mointeractions.append( interaction.get_interactions(pair) )
    mointeractions = np.array(mointeractions)

    ilm = []
    for i, orb in enumerate(mocoeff.orbitals):
        for ao in orb.aolist:
            ilm.append(Index_lm(i, ao.l, ao.m))
    num_orbital = len(ilm)

    h_ij = np.dot(mocoeff.inv_matrixA, mointeractions)
    h_ij = h_ij.reshape((num_orbital,num_orbital))

    return ilm, h_ij
