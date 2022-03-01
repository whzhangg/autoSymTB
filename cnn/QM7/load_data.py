from scipy import io
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph
import torch

one_hot_dict = {
    1:  [1,0,0,0,0],  # H
    6:  [0,1,0,0,0],  # C 
    7:  [0,0,1,0,0],  # N
    8:  [0,0,0,1,0],  # O
    9:  [0,0,0,0,1],  # F
    16: [0,0,0,0,1]   # S
}
# in QM7, the element set is 1,6,7,8,16
# in QM9, the element set is 1,6,7,8,9, therefore, we can duplicate 9 and 16 without conflict

class QM7:
    file = 'raw/qm7.mat'

    """
    it will basically be an iterable that return tuple (c,p,t)
    an entry is a tuple (c,p,t) with the target
    """

    def __init__(self, rcut = 5.0, datafile: str = None) -> None:
        if datafile:
            all_data = io.loadmat(self.file)
        else:
            all_data = io.loadmat(self.file)
        self.rcut = rcut

        self.charge = all_data['Z']
        self.position = all_data['R']
        self.target = torch.tensor(all_data['T'])
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f"dataset in {self.device}")

    def __len__(self):
        return self.charge.shape[0]

    def _get_sample(self,i):
        """
        return (pos,type), target
        where pos.shape = (natom, 3)
              type = (natom, 5) giving one hot encoding vector
              target is a scalar energy
        """
        if i >= len(self):
            raise IndexError("index out of range")

        pos = torch.from_numpy(self.position[i][self.charge[i] > 0])
        t = torch.tensor([one_hot_dict[charge] for charge in self.charge[i] if charge > 0], dtype = torch.float32)
        transform = RadiusGraph(self.rcut)
        return transform(Data(x=t, pos = pos, y = self.target[0,i])).to(self.device)
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        return self._get_sample(i)

class QM9:
    file = 'raw/qm9_v3.pt'
    prop_ref = {
        "dipole moment" : 0,
        "polarizability": 1,
        "HOMO": 2,
        "LUMO": 3,
        "Gap" : 4,
        "spatial extent": 5,
        "e vibration": 6,
        "e internal 0K": 7,
        "e internal RT": 8,
        "enthalpy": 9,
        "e free RT": 10,
        "cp RT": 11,
        "e atomization 0K": 12,
        "e atomization RT": 13,
        "atomization entropy": 14,
        "atomization free energy": 15,
        "rotation A": 16,
        "rotation B": 17,
        "rotation C": 18,
    }

    def __init__(self, required_prop: str, rcut: float = 5.0, datafile: str = None) -> None:
        
        if datafile:
            self.data = torch.load(datafile)
        else:
            self.data = torch.load(self.file)
        print(f'QM9 loaded !')
        self.prop_index = self.prop_ref[required_prop]
        self.rcut = rcut

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f"dataset in {self.device}")

    def __len__(self):
        return len(self.data)

    def _get_sample(self, i):
        if i >= len(self):
            raise IndexError("index out of range")
        
        entry = self.data[i]
        pos = entry['pos']

        t = entry['x'][:,0:5]

        properties = entry['y'][0,self.prop_index]

        transform = RadiusGraph(self.rcut)
        return transform(Data(x=t, pos = pos, y = properties).to(self.device))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        return self._get_sample(i)

def test_batch_qm7():
    import torch
    from torch_geometric.loader import DataLoader
    
    data = QM7(5.0)
    dataset = [ d for d in data ]
    loader = DataLoader(dataset, batch_size=3)
    for batch in loader:
        print(batch.x)
        print(batch.y)
        print(batch.batch)
        print(batch.pos)
        print(batch.edge_index[batch.edge_index == [1,1]])
        break

def test_qm9():
    d = QM9("HOMO")
    print(d[10])

if __name__ == "__main__":
    test_qm9()