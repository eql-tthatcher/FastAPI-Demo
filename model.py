'''Train an XGBoost classifier on Fisher's iris dataset'''

# Imports ---------------------------------------------------------------------

# Standard Library
import pickle as pkl
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional, Union

# External Libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, PositiveFloat, Field


# Data Classes and Enumerations -----------------------------------------------

class IrisMeasurements(BaseModel):
    '''Length and the width of the iris flower's sepals and petals'''
    sepal_length: float = Field(..., title='Sepal Length', ge=0, le=10)
    sepal_width: float = Field(..., title='Sepal Width', ge=0, le=10)
    petal_length: float = Field(..., title='Petal Length', ge=0, le=10)
    petal_width: float = Field(..., title='Petal Width', ge=0, le=10)


class IrisSpecies(IntEnum):
    '''Species of Iris flower'''
    Setosa = 0
    Versicolor = 1
    Virginica = 2


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

    def __init__(self, model: Optional[Any] = None) -> None:
        self.model = model

    def serialize(self, path: Union[Path,str]) -> None:
        '''Serialize the model to a pickle file'''
        with open(path, 'wb') as model_file:
            pkl.dump(self.model, model_file)

    def deserialize(self, path: Union[Path,str]) -> None:
        '''Deserialize a pickled model'''
        with open(path, 'rb') as model_file:
            self.model = pkl.load(model_file)

    def design_matrix(self, data: Any) -> np.ndarray:
        '''Generate the design matrix given input data'''
        if isinstance(data, pd.DataFrame):
            return data[self.predictors].to_numpy(float)
        elif isinstance(data, list):
            return np.array([[datum.dict().get(pred, 0.0)
                              for pred in self.predictors]
                             for datum in data], dtype=float)
        else:
            raise ValueError('data must be pandas.DataFrame or list')

    def response_vector(self, data: Any) -> np.ndarray:
        '''Generate the response vector given input data'''
        encoder = lambda x: IrisSpecies[x]
        if isinstance(data, pd.DataFrame):
            return data[self.response].apply(encoder).to_numpy(int)
        elif isinstance(data, list):
            return np.array([[encoder(datum.dict().get(self.response))]
                              for datum in data], dtype=int)
        else:
            raise ValueError('data must be pandas.DataFrame or list')

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, Z: np.ndarray):
        return self.model.predict(Z)


# Module Functions ------------------------------------------------------------

def train():
    data = pd.read_csv(Path()/'data'/'iris.csv')
    model = Model(xgb.XGBClassifier(n_jobs=-1, verbosity=1))
    X = model.design_matrix(data)
    y = model.response_vector(data)
    model.fit(X, y)
    model.serialize(Path('model')/'demo-model.pkl')


# Script ----------------------------------------------------------------------

if __name__ == '__main__':
    train()
