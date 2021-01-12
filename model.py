'''Train an XGBoost classifier on Fisher's iris dataset'''

# Imports ---------------------------------------------------------------------

# Standard Library
import pickle as pkl
from pathlib import Path
from typing import Any

# External Libraries
import numpy as np
import pandas as pd
import xgboost as xgb


# Model Class -----------------------------------------------------------------

class Model():
    '''Wrapper for XGBoost model'''
    model: Any = None
    predictors = [
        'sepal_length',
        'sepal_width',
        'petal_length'
    ]
    response = 'species'
    respose_encoding = {
        'Setosa': 0,
        'Versicolor': 1,
        'Virginica':2
    }
    response_decoding = {
        0: 'Setosa',
        1: 'Versicolor',
        2: 'Virginica'
    }

    def __init__(self, model: Any) -> None:
        self.model = model

    def design_matrix(self, data: Any) -> np.ndarray:
        '''Generate the design matrix given input data'''
        if isinstance(data, pd.DataFrame):
            return data[self.predictors].to_numpy()
        elif isinstance(data, list):
            return np.array([[datum.get(pred, 0.0) for pred in self.predictors]
                             for datum in data], dtype=float)
        else:
            raise ValueError('data must be pandas.DataFrame or list')

    def response_vector(self, data: Any) -> np.ndarray:
        '''Generate the response vector given input data'''
        encoder = self.respose_encoding.get
        if isinstance(data, pd.DataFrame):
            return data[self.response].apply(encoder).to_numpy()
        elif isinstance(data, list):
            return np.array([[encoder(datum.get(self.response))]
                             for datum in data], dtype=int)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_vector(self, Z: np.ndarray):
        return self.model.predict(Z)

    def predict_class(self, Z: np.ndarray):
        y = self.predict_vector(Z)
        return [self.response_decoding.get(int(yi)) for yi in y]


# Module Functions ------------------------------------------------------------

def train():
    iris_df = pd.read_csv(Path()/'data'/'iris.csv')

    model = Model(xgb.XGBClassifier(n_jobs=-1, verbosity=1))

    X = model.design_matrix(iris_df)
    y = model.response_vector(iris_df)

    model.fit(X, y)

    with open(Path()/'model'/'demo-model.pkl', 'wb') as model_file:
        pkl.dump(model, model_file)


# Script ----------------------------------------------------------------------

if __name__ == '__main__':
    train()
