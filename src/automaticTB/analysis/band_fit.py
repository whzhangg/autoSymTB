import numpy as np
from SALib.analyze import sobol

from ._analysis_params import which_sobol, calc_2order
from sklearn import base
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing


class SobolFeatureSelector(base.BaseEstimator, base.TransformerMixin):
    """feature selection using sobol analysis

    The class is written following the template at this webpage:
    https://www.andrewvillazon.com/custom-scikit-learn-transformers/
    """
    def __init__(self, problem: dict, y: np.ndarray, threshold: float) -> None:
        """read sobol file and determine the needed features"""
        self.threshold = threshold
        sobol_data = sobol.analyze(problem, y, calc_second_order=calc_2order)
        
        indices = np.array(sobol_data[which_sobol])
        confidence_interval = np.array(sobol_data[f"{which_sobol}_conf"])
        sorted_index = np.argsort(indices)[::-1]
        output = []
        self.feature_names = []
        for si in sorted_index:
            if indices[si] + confidence_interval[si] > self.threshold:
                output.append(si)
                self.feature_names.append(problem["names"][si])
        
        self.selected_columns = np.array(output)


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        X = X[:, self.selected_columns]
        return X 


    def get_feature_names_out(self):
        return self.feature_names


def train_polynominal_regression(
    trainx: np.ndarray, trainy: np.ndarray, problem: dict,
    polynomial_order = 1,
    sobol_threshold: float = 1e-3,
    use_sobol: bool = True,
    use_ridge: bool = True,
) -> linear_model.LinearRegression:
    """train a polynominal regressor for band energy with Sobol
    
    perhaps easier to use a sparse model instead but here I use sobol
    method for consistency.
    """
    if use_ridge:
        model = linear_model.RidgeCV(fit_intercept=True)
    else:
        model = linear_model.LinearRegression(fit_intercept=True)

    if use_sobol:
        poly_regressor = pipeline.make_pipeline(
            SobolFeatureSelector(problem, trainy, sobol_threshold),
            preprocessing.PolynomialFeatures(polynomial_order, include_bias=False),
            model
        )
    else:
        poly_regressor = pipeline.make_pipeline(
            preprocessing.PolynomialFeatures(polynomial_order, include_bias=False),
            model
        )

    poly_regressor.fit(trainx, trainy)
    
    return poly_regressor