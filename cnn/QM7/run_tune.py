from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from tune_QM9 import trainQM9_step

default_config = {
    # learning related
    "lr": tune.loguniform(1e-5, 1.0),
    "batchsize": tune.choice([16, 64]),
    "num_epoch": 5,
    "dropout_rate": 0.0,
    # model parameters
    "rcut": tune.choice([3,4,5]),
    "embedding_nbasis": 30,
    "linear_num_hidden": 100,
    "linear_num_nlayer": 1,
    "lmax": tune.choice([2, 3]),
    "nlayer": tune.choice([2,3]),
    "multiplicity": 50,
    "skip": True,
    "alpha": True
}

gridsearch_config = {
    # learning related
    "lr": tune.grid_search([1e-5, 1e-3, 1e-1]),
    "batchsize": 32,
    "num_epoch": 5,
    "dropout_rate": 0.0,
    # model parameters
    "rcut": 3,
    "embedding_nbasis": 30,
    "linear_num_hidden": 100,
    "linear_num_nlayer": 1,
    "lmax": tune.grid_search([2, 3]),
    "nlayer": tune.grid_search([2,3]),
    "multiplicity": 50,
    "skip": True,
    "alpha": True
}

tune_time = "step"
tune_metric = "test_mse"
tune_mode = "min"
tune_maxstep = 20

asha_scheduler = ASHAScheduler(
                    time_attr = tune_time,
                    metric = tune_metric,
                    mode = tune_mode,
                    max_t = tune_maxstep,
                    grace_period=5
                )

scheduler_PBT = PopulationBasedTraining(
                    time_attr = tune_time,
                    metric = tune_metric,
                    mode = tune_mode,
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

def tune_QM9(config, nsample: int = 10, name:str = "QM9", schedular = None):
    result = tune.run(
        trainQM9_step,
        config = config,
        num_samples = nsample,
        scheduler = schedular,
        local_dir = 'ray_tune',
        name = name,
        resources_per_trial={'gpu': 1}
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))

if __name__ == "main":
    tune_QM9(gridsearch_config, nsample = 1, name = "QM9_gridsearch")
