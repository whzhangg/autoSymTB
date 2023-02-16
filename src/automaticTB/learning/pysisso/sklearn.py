import copy
import typing
import datetime
import tempfile
import shutil

import numpy as np
import pandas as pd
from monty import os as montyos
from sklearn.base import BaseEstimator, RegressorMixin

from .sisso_in import SISSOIn
from .sisso_out import SISSOOut
from .data_in import SISSOSingleRegDat, SISSOUnitIn
from .job import runSISSO


def get_timestamp() -> object:
    """Get a string representing the a time stamp."""
    tstamp = datetime.datetime.now()
    return (
        f"{str(tstamp.year).zfill(4)}_{str(tstamp.month).zfill(2)}_"
        f"{str(tstamp.day).zfill(2)}_"
        f"{str(tstamp.hour).zfill(2)}_{str(tstamp.minute).zfill(2)}_"
        f"{str(tstamp.second).zfill(2)}_{str(tstamp.microsecond).zfill(6)}"
    )


class SISSORegressor(RegressorMixin, BaseEstimator):

    def __init__(self,
        feature_units: typing.Optional[np.ndarray] = None,
        target_unit: typing.Optional[np.ndarray] = None,
        run_dir: typing.Optional[str] = None,
        clean_run_dir: bool = False,
        scmt: bool = False,
        desc_dim: int = 2,
        ops: str = "(+)(-)(*)(/)(^2)(^3)(^-1)",
        fcomplexity: int = 3,
        fmax_min: float = 1e-3,
        fmax_max: float = 1e5,
        nf_sis: int = 20,
        method_so: str = 'L0',
        nl1l0: int = 1,
        L1_max_iter: int = int(1e6),
        L1_dens: int = 120,
        L1_tole: float = 1e-6,
        L1_minrmse: float = 1e-3,
        L1_warm_start: bool = True,
        fit_intercept: bool = True,
        metric: str = "RMSE",
        nmodels: int = 50,
        isconvex: str = '(1,1)',
        bwidth: float = 1e-3,
    ) -> None:
        self.feature_units = feature_units
        self.target_unit = target_unit
        self.run_dir = run_dir
        self.clean_run_dir = clean_run_dir
        #
        self.scmt = scmt
        self.desc_dim = desc_dim
        self.ops = ops
        self.fcomplexity = fcomplexity
        self.fmax_min = fmax_min
        self.fmax_max = fmax_max
        self.nf_sis = nf_sis
        self.method_so = method_so
        self.nl1l0 = nl1l0
        self.L1_max_iter = L1_max_iter
        self.L1_dens = L1_dens
        self.L1_tole = L1_tole
        self.L1_minrmse = L1_minrmse
        self.L1_warm_start = L1_warm_start
        self.fit_intercept = fit_intercept
        self.metric = metric
        self.nmodels = nmodels
        self.isconvex = isconvex
        self.bwidth = bwidth

        self._sisso_out = None
        self._columns = None

    def fit(self, X, y, index=None, feature_names=None):
        if len(y) != X.shape[0]:
            raise ValueError("number of y not equal to number of X")
        if feature_names is not None and len(feature_names) != X.shape[1]:
            raise ValueError("number of feature names are not correct")
        
        nsample = len(y)
        nsf = X.shape[1]
        indices = index or [f"sample{i+1}" for i in range(nsample)]
        self._columns = feature_names or [f"feature{i+1}" for i in range(nsf)]
        data = pd.DataFrame(X, index=indices, columns=self._columns)
        data.insert(0, "target", y)

        if self.run_dir is None:
            montyos.makedirs_p("SISSO_runs")
            timestamp = get_timestamp()
            self.run_dir = tempfile.mkdtemp(
                suffix=None, prefix=f"SISSO_dir_{timestamp}_", dir="SISSO_runs"
            )
        else:
            montyos.makedirs_p(self.run_dir)

        runner = SISSORunner(
            data, self.feature_units, self.target_unit, 
            **{
                "scmt": self.scmt, "desc_dim": self.desc_dim, "ops": self.ops, 
                "fcomplexity": self.fcomplexity, "fmax_min": self.fmax_min, 
                "fmax_max": self.fmax_max, "nf_sis": self.nf_sis,
                "method_so": self.method_so, "nl1l0": self.nl1l0, 
                "L1_max_iter": self.L1_max_iter, "L1_dens": self.L1_dens, 
                "L1_tole": self.L1_tole, "L1_minrmse": self.L1_minrmse, 
                "L1_warm_start": self.L1_warm_start, 
                "fit_intercept": self.fit_intercept, "metric": self.metric, 
                "nmodels": self.nmodels, "isconvex": self.isconvex, "bwidth": self.bwidth
            }
        )
        with montyos.cd(self.run_dir):
            runner.run()
            self._sisso_out = SISSOOut.from_file(filepath="SISSO.out")

        if self.clean_run_dir:
            shutil.rmtree(self.run_dir)

    def predict(self, X, index=None):
        X = np.array(X)
        index = index or ["item{:d}".format(ii) for ii in range(X.shape[0])]
        data = pd.DataFrame(X, index=index, columns=self._columns)
        return self._sisso_out.model.predict(data)
    

class SISSORunner:
    
    def __init__(self, 
        yX: pd.DataFrame, 
        feature_units: typing.Optional[np.ndarray] = None, 
        target_unit: typing.Optional[np.ndarray] = None,
        **addargs
    ) -> None:
        nsample = yX.shape[0]
        nsf = yX.shape[1] - 1
        if feature_units is None:
            feature_units = np.ones((nsf, 1))
        args = copy.deepcopy(addargs)
        args["nsample"] = nsample
        args["nsf"] = nsf
        args["funit"] = self.create_funit(feature_units, target_unit)
        self.input = SISSOIn(**args)
        self.train_data = SISSOSingleRegDat(yX)
        self.unit_data = SISSOUnitIn(feature_units, target_unit)

    def run(self, run_dir: typing.Optional[str] = None):
        """run the job"""
        def _run():
            self.input.to_file(filename="SISSO.in")
            self.train_data.to_file(filename="train.dat")
            self.unit_data.to_file(filename="feature_units")
            runSISSO()

        if run_dir is None:
            _run()
        else:
            with montyos.cd(run_dir):
                _run()


    def create_funit(
            self, feature_units: np.ndarray, target_unit: typing.Optional[np.ndarray]) -> str:
        if target_unit is None:
            all_units = feature_units
        else:
            all_units = np.vstack([target_unit, feature_units])
        
        feature_ndim = all_units.shape[1]
        feature_rank = np.linalg.matrix_rank(all_units, tol=1e-6)
        if feature_ndim != feature_rank:
            raise ValueError(f"unit matrix seems to be not full rank!")
        
        def list_is_continuity(my_list):
            return all(a+1==b for a, b in zip(my_list, my_list[1:]))

        # create the string
        discovered = []
        _all_discoverd = []
        for i, irow in enumerate(feature_units):
            if i in _all_discoverd: continue
            _same_index = []
            for j, jrow in enumerate(feature_units):
                if np.allclose(irow, jrow):
                    _same_index.append(j)
                    _all_discoverd.append(j)
            if list_is_continuity(_same_index):
                discovered.append(_same_index)
            else:
                raise ValueError("Feature of the same unit need to continous")
        
        result = ""
        for sameunit in discovered:
            result += f"({sameunit[0]+1}:{sameunit[-1]+1})"

        return result