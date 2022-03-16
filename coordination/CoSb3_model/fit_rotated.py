from fit import torch_dtype, ef
import os
folder = "rotated_distorted"
import numpy as np
import torch
from torch_geometric.data import Data

def make_dataset():
    from DFTtools.DFT.output_parser import parse_pdos_folder
    from mldos.data.milad_utility import Description
    list_of_data = []
    for i in range(1,4):
        folder_rot = os.path.join(folder, f"rot{i}")
        scf = os.path.join(folder_rot, 'scf.in')

        dos_result = parse_pdos_folder(folder_rot)
        occupation = {}
        for iatom, dos in dos_result.items():
            if iatom != 4:
                continue
            energy = dos['3d']['energy']
            mask = energy < ef
            result = []
            for orb in ['pdos_1','pdos_2','pdos_3','pdos_4','pdos_5']:
                d = dos['3d'][orb]
                result.append( np.sum(d[mask]) * 0.05 )
            occupation[iatom] = result

        allpoints = [iatom-1 for iatom in occupation.keys()]
        description = Description(filename=scf, keepstructure=True)
        neighbors = description.get_neighbors(allpoints, bonded=False, firstshell=True, cutoff=5)

        for iatom, occ in occupation.items():
            types = [ description.std_types[n.index] for n in neighbors[iatom - 1] ]
            pos = [ n.rij for n in neighbors[iatom - 1] ]

            x = torch.tensor([[1.0]] * len(pos), dtype=torch_dtype)
            pos = torch.tensor(np.array(pos), dtype=torch_dtype)
            occ = torch.tensor(occ, dtype=torch_dtype)
                    
            list_of_data.append(Data(x=x, pos = pos, y = occ))

    return list_of_data

if __name__ == "__main__":
    data = make_dataset()
    print(data[0].pos)