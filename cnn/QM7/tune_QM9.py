from config import device
from load_data import QM7, QM9
from dataset import InputData
from network import molecular_network

import torch
from functools import partial
from ray import tune
import ray
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

from torch.utils.data import random_split
import torch_geometric.loader as geoloader

"""
This script tune hyperparameters of a tensor MPN for QM9 dataset
"""


def trainQM9(config, datafilepath: str = "/home/wenhao/work/ml_dos/mldos/cnn/QM7/raw/qm9_v3.pt", 
                    random_seed = 424242, verbose: bool = True):
    # this is the target function to train
    
    R_cut = config["rcut"]
    dataset = InputData(QM9("HOMO", R_cut, datafile=datafilepath))

    num_neighbor = R_cut**2.5 # should scale as r^3

    net = molecular_network('5x0e','1x0e',
        lmax = config["lmax"],
        nlayer=config["nlayer"],
        hidden_multiplicity = config["multiplicity"],
        embedding_nbasis = config["embedding_nbasis"],
        linear_num_hidden = config["linear_num_hidden"],
        linear_num_layer = config["linear_num_nlayer"],
        rcut = R_cut,
        num_neighbors = num_neighbor,
        )
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
    validation_size = config['validationsize']
    input_dropout = config['dropout_rate']

    epoch = 0
    loss_result = None

    # split only a part of dataset 10k
    dataset, _ = random_split(dataset, 
                                [10000, len(dataset)-10000], 
                                generator=torch.Generator().manual_seed(random_seed)
                            )
        
    nvalidation = int(len(dataset) * validation_size)
    validation, train = random_split(dataset, 
                                [nvalidation, len(dataset)-nvalidation], 
                                generator=torch.Generator().manual_seed(random_seed)
                            )

    validation_loader = geoloader.DataLoader(validation, config["batchsize"])
    train_dataloader = geoloader.DataLoader(train, config["batchsize"])

    train_scores = []
    test_scores = []
    for epoch in range(config["num_epoch"]):

        if input_dropout > 1e-10:
            n_small = int(len(train) * input_dropout)
            small, large = random_split(train, [n_small,len(train) - n_small], generator=torch.Generator().manual_seed(random_seed))
            train_dataloader = geoloader.DataLoader(large, config["batchsize"])

        for batch in train_dataloader:
            optimizer.zero_grad()

            predictions = net(batch)
            loss_result = (batch.y.unsqueeze(1) - predictions).pow(2).sum()

            loss_result.backward()
            optimizer.step()

        with torch.no_grad():
                # if more than 1 batch, we evaluate loss for the entire training set
            train_score = torch.zeros((1,)).to(device)
            test_score = torch.zeros((1,)).to(device)

            counts = 0
            for batch in train_dataloader:
                predictions = net(batch)
                counts += config["batchsize"]
                train_score += (batch.y.unsqueeze(1) - predictions).pow(2).sum()
            train_score /= counts
                
            counts = 0
            for batch in validation_loader:
                predictions = net(batch)
                counts += config["batchsize"]
                test_score += (batch.y.unsqueeze(1) - predictions).pow(2).sum()
            test_score /= counts

            train_scores.append(train_score.item())
            test_scores.append(test_score.item())

            if verbose:
                print("Epoch {:>5d} = train_score : {:>16.6e}; test_score : {:>16.6e}".format(epoch+1, train_scores[-1], test_scores[-1]))
            else:
                tune.report(train_loss = train_scores[-1], test_loss = test_scores[-1])

    return loss_result


def trainQM9_step(config, datafilepath: str = "/home/wenhao/work/ml_dos/mldos/cnn/QM7/raw/qm9_v3.pt", 
                    random_seed = 424242, verbose: bool = True, report_step: int = 500, subset_size: int = 10000):
    # this is the target function to train, compare to the previous one, this version
    # report to tune when one epoch is not finished
    # we only use 10k data for optimizing parameters

    validation_size = 0.1
    validation_batchsize = int(config["batchsize"] * 1)
    
    R_cut = config["rcut"]
    dataset = InputData(QM9("HOMO", R_cut, datafile=datafilepath))

    num_neighbor = R_cut**2.5 # should scale as r^3

    net = molecular_network('5x0e','1x0e',
        lmax = config["lmax"],
        nlayer=config["nlayer"],
        hidden_multiplicity = config["multiplicity"],
        embedding_nbasis = config["embedding_nbasis"],
        linear_num_hidden = config["linear_num_hidden"],
        linear_num_layer = config["linear_num_nlayer"],
        rcut = R_cut,
        num_neighbors = num_neighbor,
        )
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
    input_dropout = config['dropout_rate']

    epoch = 0
    loss_result = None

    # split only a part of dataset 10k
    dataset, _ = random_split(dataset, 
                                [subset_size, len(dataset)-subset_size], 
                                generator=torch.Generator().manual_seed(random_seed)
                            )
        
    nvalidation = int(len(dataset) * validation_size)
    validation, train = random_split(dataset, 
                                [nvalidation, len(dataset)-nvalidation], 
                                generator=torch.Generator().manual_seed(random_seed)
                            )

    validation_loader = geoloader.DataLoader(validation, validation_batchsize)
    train_dataloader = geoloader.DataLoader(train, config["batchsize"])

    test_scores = []
    test_scores_mae = []

    nbatch_counts = 0
    num_batch_step = int(report_step / config["batchsize"])
    report_step = num_batch_step * config["batchsize"]

    for epoch in range(config["num_epoch"]):

        if input_dropout > 1e-10:
            n_small = int(len(train) * input_dropout)
            small, large = random_split(train, [n_small,len(train) - n_small], generator=torch.Generator().manual_seed(random_seed))
            train_dataloader = geoloader.DataLoader(large, config["batchsize"])

        for batch in train_dataloader:
            nbatch_counts += config["batchsize"]

            optimizer.zero_grad()

            predictions = net(batch)
            loss_result = (batch.y.unsqueeze(1) - predictions).pow(2).sum()

            loss_result.backward()
            optimizer.step()

            if nbatch_counts % report_step == 0:
                with torch.no_grad():
                    # if more than 1 batch, we evaluate loss for the entire training set
                    test_mse = torch.zeros((1,)).to(device)
                    test_mae = torch.zeros((1,)).to(device)
                        
                    counts = 0
                    for batch in validation_loader:
                        predictions = net(batch)
                        counts += validation_batchsize
                        test_mse += (batch.y.unsqueeze(1) - predictions).pow(2).sum()
                        test_mae += torch.abs(batch.y.unsqueeze(1) - predictions).sum()
                    test_mse /= counts
                    test_mae /= counts
                                    
                    test_scores.append(test_mse.item())
                    test_scores_mae.append(test_mae.item())
                

                if verbose:
                    print("Step {:>10d} sample passed,test_score : {:>16.6e}; test_mae : {:>16.6e}".format(
                                epoch+1, test_scores[-1], test_scores_mae[-1]))
                else:
                    tune.report(test_mse = test_scores[-1], test_mae = test_scores_mae[-1])

    return loss_result


def main_ASHA():
    #ray.init(log_to_driver=False)
    scheduler = ASHAScheduler(
                    metric="test_loss",
                    mode="min",
                    max_t=150,
                    grace_period=5
                )
    
    config = {
        # learning related
        "lr": tune.loguniform([1e-5, 1.0]),
        "batchsize": tune.choice([16, 32, 64]),
        "num_epoch": tune.choice([20]),
        "dropout_rate": 0.0,
        # model parameters
        "rcut": tune.choice([3,4,5]),
        "embedding_nbasis": tune.choice([20, 30, 40]),
        "linear_num_hidden": tune.choice([100, 200]),
        "linear_num_nlayer": tune.choice([1,2]),
        "lmax": tune.choice([1, 2, 3]),
        "nlayer": tune.choice([2,3]),
        "multiplicity": tune.choice([20,40])
    }

    result = tune.run(
        trainQM9,
        config = config,
        num_samples = 4,
        scheduler = scheduler,
        local_dir = 'tune_QM9',
        name = 'tune_0219',
        resources_per_trial={'gpu': 1}
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))

def main_PBT():
    scheduler_PBT = PopulationBasedTraining(
                    metric='test_mse',
                    mode='min',
                    perturbation_interval=5,
                    hyperparam_mutations={
                        "lr": tune.loguniform(1e-5, 1.0),
                        "batchsize": tune.choice([16, 32, 64]),
                        "rcut": tune.choice([3,4,5]),
                        "embedding_nbasis": tune.choice([20, 30, 40]),
                        "linear_num_hidden": tune.choice([100, 200]),
                        "linear_num_nlayer": tune.choice([1,2]),
                        "lmax": tune.choice([1, 2, 3]),
                        "nlayer": tune.choice([2,3]),
                        "multiplicity": tune.choice([20,40])
                    }
    )
    config = {
        # learning related
        "lr": tune.loguniform(1e-5, 1.0),
        "batchsize": tune.choice([16, 32, 64]),
        "num_epoch": 20,
        "dropout_rate": 0.0,
        # model parameters
        "rcut": tune.choice([3,4,5]),
        "embedding_nbasis": tune.choice([20, 30, 40]),
        "linear_num_hidden": tune.choice([100, 200]),
        "linear_num_nlayer": tune.choice([1,2]),
        "lmax": tune.choice([1, 2, 3]),
        "nlayer": tune.choice([2,3]),
        "multiplicity": tune.choice([20,40])
    }

    result = tune.run(
        trainQM9_step,
        config = config,
        num_samples = 10,
        scheduler = scheduler_PBT,
        local_dir = 'raytune_QM9',
        name = 'QM9_PBT',
        resources_per_trial={"gpu":1}
    )

    best_trial = result.get_best_trial("test_mse", "min", "last")
    print("Best trial config: {}".format(best_trial.config))


def fixed_run():
    config = {
        # learning related
        "lr": 1e-3,
        "batchsize": 32,
        "num_epoch": 20,
        "validationsize": 0.1, 
        "dropout_rate": 0.1,
        # model parameters
        "rcut": 3,
        "embedding_nbasis": 30,
        "linear_num_hidden": 100,
        "linear_num_nlayer": 1,
        "lmax": 2,
        "nlayer": 2,
        "multiplicity": 20
    }
    trainQM9_step(config, verbose=True)


if __name__ == "__main__":
    main_PBT()