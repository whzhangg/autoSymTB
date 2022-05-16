import typing
import numpy as np
from .lc_product import RepLabelledBasis
from .interaction import Interaction
from ..linear_combination import Site, Orbitals


class NNInteraction:
    def __init__(self, 
        sites: typing.List[Site], 
        orbital: Orbitals, 
        interaction_matrix: np.ndarray
    ) -> None:
        self._sites = sites
        self._orbital = orbital
        self.interaction_matrix = interaction_matrix

        assert self.interaction_matrix.shape[0] == self.interaction_matrix.shape[1] \
            and self.interaction_matrix.shape[0] == len(self._sites) * self._orbital.num_orb
        
    @classmethod
    def from_labelledbasis_and_interaction(
        cls, basis: RepLabelledBasis, interaction: Interaction): # NNInteraction

        Hij = []
        for i in range(basis.rank):
            pairedLC = basis.get_pair_LCs(i)
            Hij.append(interaction.get_interactions(pairedLC.rep1, pairedLC.rep2))
        #print_matrix(lcs.matrixA, "{:>6.3}")
        invA = np.linalg.inv(basis.matrixA)
        x = invA.dot(np.array(Hij))
        x = x.reshape((basis.num_AOs, basis.num_AOs))

        return cls(
            basis._sites, basis._orbitals, x
        )

    