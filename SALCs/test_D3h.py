import math
import torch
from torch_type import float_type
from atomic_orbital import (
    CenteredAO, get_coefficient_dimension, get_linear_combination, VectorSpace
)
from decompose import find_SALCs
from group_subgroup import GroupTable


D3h_positions = torch.tensor([
        [ 2.0,   0.0, 0.0] ,
        [-1.0, math.sqrt(3), 0.0],
        [-1.0,-math.sqrt(3), 0.0],
    ], dtype = float_type)
D3h_positions = torch.index_select(D3h_positions, 1, torch.tensor([0,2,1]))
D3h_AOs = [ '1x1o' for pos in D3h_positions ]
aolist = []
for i, (pos, ao) in enumerate(zip(D3h_positions, D3h_AOs)):
    aolist.append(CenteredAO(f"C{i+1}", pos, ao))
n = get_coefficient_dimension(aolist)
lc = get_linear_combination(torch.eye(n, dtype = float_type), aolist)
        
rep = VectorSpace(lc)

if __name__ == "__main__":
    result = find_SALCs(rep, "D3h")
    print(result.display_str)
    #print(result)
    #v1 = result["E\'"]["A\""]
    #lc1 = v1.LCs[1]
    from plot_mo import plot_density
    for i, (name, lc) in enumerate(zip(result.names, result.LCs)):
        iso = 0.3
        if name == "A2'":
            iso = 0.1
        
        #xs, ys, zs, density = lc.density()
        #plot_density(xs, ys, zs, density, f"C4/{i+1}-{name}.html", iso)
    #xs, ys, zs, density = lc1.density()
    #from plot_mo import plot_density
    #plot_density(xs, ys, zs, density, "E-A\".html")

