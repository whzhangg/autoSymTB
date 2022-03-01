from torch_geometric.datasets import QM7b
import scipy

dataset = QM7b(root = '.')
datasetmat = scipy.io.loadmat('raw/qm7b.mat')

print(datasetmat['names'])
print(type(dataset))
print(dataset[0])