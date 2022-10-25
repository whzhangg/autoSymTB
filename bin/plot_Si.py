from automaticTB.examples.structures import get_Si_structure_2s2p3s
from automaticTB.printing import print_ao_pairs
from automaticTB.functions import (
    get_namedLCs_from_nncluster, get_combined_equation_from_structure
)
from automaticTB.tightbinding import TightBindingModel, gather_InteractionPairs_into_HijRs
from automaticTB.properties import Kpath, Kline
from automaticTB.tools import find_RCL
import numpy as np
from automaticTB.properties.bandstructure import BandStructureResult

bulksi = get_Si_structure_2s2p3s()

combined_equation = get_combined_equation_from_structure(bulksi)

interaction_values = np.array([
    8.23164,  #  1 > Si     3s -> Si     3s @ (  0.00,  0.00,  0.00) OK
    -3.3179,  #  2 > Si     2s -> Si     2s @ (  0.00,  0.00,  0.00) OK
    1.67862,  #  3 > Si    2py -> Si    2py @ (  0.00,  0.00,  0.00) OK
    0,      #  4 > Si     3s -> Si     2s @ (  1.36, -1.36, -1.36) OK
    7.2505/4,   #  5 > Si     3s -> Si    2px @ (  1.36, -1.36, -1.36) OK
    -9.59895/4, #  6 > Si     2s -> Si     2s @ (  1.36, -1.36, -1.36) OK
    0,      #  7 > Si     3s -> Si     3s @ (  1.36, -1.36, -1.36) OK
    7.1423/4,   #  8 > Si     2s -> Si    2px @ (  1.36, -1.36, -1.36) OK
    0,      #  9 > Si     2s -> Si     3s @ (  1.36, -1.36, -1.36) OK
   -7.1423/4,   # 10 > Si    2px -> Si     2s @ (  1.36, -1.36, -1.36) OK
   -4.77573/4,  # 11 > Si    2px -> Si    2pz @ (  1.36, -1.36, -1.36)
    1.69552/4,  # 12 > Si    2px -> Si    2px @ (  1.36, -1.36, -1.36) OK
   -7.2505/4,   # 13 > Si    2px -> Si     3s @ (  1.36, -1.36, -1.36)
])

interaction_pairs = combined_equation.solve_interactions_to_InteractionPairs(interaction_values)
hijRs = gather_InteractionPairs_into_HijRs(interaction_pairs)

model = TightBindingModel(
    bulksi.cell, bulksi.positions, bulksi.positions, hijRs
)

rcl = find_RCL(bulksi.cell)
path = Kpath(rcl, [
        Kline("L", np.array([0.5,0.5,0.5]), "G", np.array([0.0,0.0,0.0])),
        Kline("G", np.array([0.0,0.0,0.0]), "X", np.array([0.5,0.0,0.5]))
    ]
)

bandresult = BandStructureResult.from_tightbinding_and_kpath(model, path)
bandresult.plot_data("Si.pdf")
e, _ = model.solveE_at_k(np.array([0, 0, 0]))
print(e)