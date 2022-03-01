import os

if not os.path.exists('qm7.mat'): 
    # http://quantum-machine.org/datasets/
    os.system('wget http://www.quantum-machine.org/data/qm7.mat')

if not os.path.exists('qm9_v3.pt'):
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9
    os.system('wget https://data.pyg.org/datasets/qm9_v3.zip')
    os.system('unzip qm9_v3.zip')
    os.system('rm qm9_v3.zip')