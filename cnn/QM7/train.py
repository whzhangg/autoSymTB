# 
# this script contains training algrothim, as well as stop function
#
from config import device, random_seed
#
#
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import torch_geometric.loader as geoloader


class EarlyStop:
    """ callable that give true if training should be stopped
        it either use early stopping or does not use early stopping. 
        For early stopping, algorithm 7.1 is used from DL Goodfellow.

        step defines an average on which we compare the test error
    """
    def __init__(self, 
        min_epochs: int = 30,
        max_epochs: int = 300, 
        early_stop: bool = True, 
        patience: int = 2, 
        step: int = 5,
        verbose: bool = False
    ) -> None:
        self.min_epoch = min_epochs
        self.max_epoch = max_epochs
        self.early_stop = early_stop
        self.patience = patience
        self.step = step
        self.verbose = verbose

        self.tmp_error = []
        self.test_errors = []
        self.i = 0
        self.j = 0

        self.criteria_satisfied = False

    def judge_stop(self):
        # update internal values
        self.i += 1
        if self.i % self.step == 0:
            tmp_error = sum(self.tmp_error) / len(self.tmp_error)

            if (np.abs(np.array(self.tmp_error) - tmp_error) < 1e-10).all():
                return True   # this should prevent training to continue if all error values are the same
            
            if self.early_stop and len(self.test_errors) > 0:
                if tmp_error > self.test_errors[-1]:
                    self.j += 1
                else:
                    self.j = 0

                if self.j > self.patience:
                    return True

            self.test_errors.append(tmp_error)
            self.tmp_error = []

        return False

    def __call__(self, epoch, test_error):
        # update internal values
        self.tmp_error.append(test_error)

        if not self.criteria_satisfied:
            self.criteria_satisfied = self.judge_stop()

        # hard requirement
        if epoch+1 >= self.max_epoch:
            return True
        if epoch+1 <= self.min_epoch:
            return False

        if self.criteria_satisfied:
            return True
        else:
            return False


class Training():
    """
    the abstract base class, we need to implement
    method __init__, _fit and _forward
    It 
    """
    def __init__(self, 
        network,
        stop_function,
        optimizer,
        validation_size: float = 0.1,
        batchsize: int = 16,
        input_dropout: float = 0.0,
        verbose: bool = True
    ):
        self._network = network
        self._stop_function = stop_function

        self.optimizer = optimizer
        self.validation_size = validation_size
        self.batchsize = batchsize
        self.verbose = verbose
        self.input_dropout = input_dropout

        self.train_scores = []
        self.test_scores = []

        self.device = device


    def predict(self, X):
        """a general method that will yield real y"""
        return self._forward(X)

    def train(self, training_set):
        """this is a wrapper for convenient overloading"""
        return self._fit(training_set)

    def estimate_error(self, input_dataset: Dataset):
        """we require input data to be on the correct device"""
        dataset = geoloader.DataLoader(input_dataset)
        loss = torch.zeros((1,)).to(self.device)
        with torch.no_grad():
            counts = 0
            for batch in dataset:
                predictions = self._forward(batch)
                
                counts += 1
                loss += self._loss_function(batch.y.unsqueeze(1), predictions)
                
        loss = loss / counts
        return loss
    
    def _loss_function(self, target, predict) -> torch.tensor:
        """this yield the error function
            it's important to ensure that the shape are the same. 
        """
        return (predict - target).pow(2).sum()


    def _fit(self, training_set: Dataset):
        """training data will be a list of Data object of torch_geometry"""
        epoch = 0
        loss_result = None
        
        nvalidation = int(len(training_set)*self.validation_size)
        validation, train = random_split(training_set, [nvalidation, len(training_set)-nvalidation], generator=torch.Generator().manual_seed(424242))

        validation_loader = geoloader.DataLoader(validation)
        train_dataloader = geoloader.DataLoader(train, self.batchsize)

        print("training start")
        while True:

            if self.input_dropout > 1e-10:
                n_small = int(len(train)*self.input_dropout)
                small, large = random_split(train, [n_small,len(train) - n_small], generator=torch.Generator().manual_seed(424242))
                train_dataloader = geoloader.DataLoader(large, self.batchsize)

            for batch in train_dataloader:
                self.optimizer.zero_grad()

                predictions = self._forward(batch)
                loss_result = self._loss_function(batch.y.unsqueeze(1), predictions)

                loss_result.backward()
                self.optimizer.step()

            with torch.no_grad():
                # if more than 1 batch, we evaluate loss for the entire training set
                train_score = torch.zeros((1,)).to(self.device)
                test_score = torch.zeros((1,)).to(self.device)

                counts = 0
                for batch in train_dataloader:
                    predictions = self._forward(batch)
                    counts += self.batchsize
                    train_score += self._loss_function(batch.y.unsqueeze(1), predictions)
                train_score /= counts
                
                counts = 0
                for batch in validation_loader:
                    predictions = self._forward(batch)
                    counts += 1
                    test_score += self._loss_function(batch.y.unsqueeze(1), predictions)
                test_score /= counts

                self.train_scores.append(train_score.item())
                self.test_scores.append(test_score.item())

                if self.verbose:
                    print("Epoch {:>5d} = train_score : {:>16.6e}; test_score : {:>16.6e}".format(epoch+1, self.train_scores[-1], self.test_scores[-1]))

                if self._stop_function(epoch, self.test_scores[-1]):
                    break
            
            epoch += 1

        return loss_result


    def _forward(self, X):
        return self._network(X)
