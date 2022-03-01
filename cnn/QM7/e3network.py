from config import device
from load_data import QM7, QM9
from dataset import InputData
from train import Training, EarlyStop
from network import molecular_network

import torch

def n_neighbors(dataset):
    nedge = 0
    nnode = 0
    for d in dataset: 
        nedge += d.edge_index.shape[1]
        nnode += d.x.shape[0]
    # there is on average 6.3 neighbors within this cutoff
    return nedge / nnode


def train_QM7():
    R_cut = 5.0
    dataset = InputData(QM7(R_cut))
    num_neighbor = n_neighbors(dataset)
    net = molecular_network('5x0e','1x0e', nlayer=1, num_neighbors = num_neighbor)
    net.to(device)
    stop_function = EarlyStop(min_epochs=10)
    optim = torch.optim.Adam(net.parameters(), lr=1e-2)

    trainner = Training(
        net, stop_function, optim
    )
    trainner.train(dataset)


def train_QM9():
    R_cut = 3.0
    dataset = InputData(QM9("HOMO", R_cut))
    #num_neighbor = n_neighbors(dataset)
    #print(num_neighbor)  
    # the calculated value is below
    num_neighbor = 8.0

    net = molecular_network('5x0e','1x0e', nlayer=2, num_neighbors = num_neighbor)
    net.to(device)
    stop_function = EarlyStop(min_epochs=10, max_epochs=30)
    optim = torch.optim.Adam(net.parameters(), lr=1e-2)

    trainner = Training(
        net, stop_function, optim
    )
    trainner.train(dataset)

if __name__ == '__main__':
    train_QM9()
