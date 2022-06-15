from automaticTB.SALCs import LinearCombination, VectorSpace
from automaticTB.examples.AH3.structure import get_nncluster_AH3
from automaticTB.plotting import DensityCubePlot
import numpy as np

cluster = get_nncluster_AH3()
vectorspace = VectorSpace.from_NNCluster(cluster)
lc = vectorspace.get_nonzero_linear_combinations()[0]

e1 = lc.create_new_with_coefficients(np.array([0,0,0,0,2,-1,-1], dtype = lc.coefficients.dtype)).get_normalized()
e2 = lc.create_new_with_coefficients(np.array([0,0,0,0,-1,2,-1], dtype = lc.coefficients.dtype)).get_normalized()

plot = DensityCubePlot.from_linearcombinations([e2])
plot.plot_to_file("E2.cube")