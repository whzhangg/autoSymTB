from read_proj import get_datas
from gbf import molecularfuntion
from plot_mo import plot_density
import numpy as np
#     0   1   2   3   4    5    6    7     8
# My: s  px  py  pz  dxy  dyz  dxz  dz2    dx2-y2
# QE: s  pz  px  py  dz2  dzx  dzy  dx2-y2 dxy
permutation = [0, 2, 3, 1, 8, 6, 5, 4, 7]

def plot_Data(data, linrange, filename):
    """take in a linear range and a torch_geometry data object"""
    xs, ys, zs = np.meshgrid(linrange, linrange, linrange)
    positions = data.pos.numpy() - np.array([5.0,5.0,5.0])
    coefficient = data.x.numpy()[:, permutation[0:4]]
    

    value = molecularfuntion(positions, coefficient, xs, ys, zs )
    print(np.max(value))
    print(np.min(value))
    plot_density(xs, ys, zs, value, filename, iso=0.1)

def main():
    datas = get_datas("scf/scf.out", "scf/proj_coeff.out", feature_max=2)
    three_mos = datas[1:4]

    for i,mo in enumerate(three_mos):
        plot_Data(mo, np.arange(-4,4,0.1), f"SF6_{i+1}.html")

if __name__ == "__main__":
    main()