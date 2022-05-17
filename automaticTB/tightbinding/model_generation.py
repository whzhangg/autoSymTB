import typing
import numpy as np
from .model import TightBindingBase
from .tightbinding_data import HijR, Pindex_lm, TBData

from ..structure.nncluster import NNCluster
from ..hamiltionian.NNInteraction import OrbitalandInteraction

def generate_tightbinding_data(
    cell: np.ndarray, 
    positions: typing.List[np.ndarray],
    types: typing.List[int],
    clusters: typing.List[NNCluster], 
    interactions: typing.List[OrbitalandInteraction]
):
    HijR_list: typing.List[HijR] = []
    for cluster, interaction in zip(clusters, interactions):
        cell_index_ilms = []
        for ilm in interaction.ilms:
            pindex = cluster.crystalsites[ilm.i].index_pcell
            translation = cluster.crystalsites[ilm.i].translation
            cell_index_ilms.append(
                Pindex_lm(pindex, ilm.l, ilm.m, translation)
            )
        

        origin_index = cluster.origin_index
        for i1, ilm1 in enumerate(interaction.ilms):
            for i2, ilm2 in enumerate(interaction.ilms):
                if ilm1.i == origin_index:
                    HijR_list.append(
                        HijR(cell_index_ilms[i1], cell_index_ilms[i2], interaction.interaction[i1,i2])
                    )

    return TBData(cell, positions, types, HijR_list)