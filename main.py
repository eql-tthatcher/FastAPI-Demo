'''Demo XGBoost Model API'''

# Imports ---------------------------------------------------------------------

# Standard Library
from pathlib import Path

# External Libraries
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Local Modules
from model import Model, IrisMeasurements, IrisSpecies


# Globals ---------------------------------------------------------------------

APP = FastAPI()
MODEL = Model()


# Application -----------------------------------------------------------------

@APP.on_event("startup")
async def load_model():
    MODEL.deserialize(Path('model')/'demo-model.pkl')


@APP.post('/predict')
async def predict(measurements: List[IrisMeasurements]) -> List[str]:
    X = MODEL.design_matrix(measurements)
    y = MODEL.predict(X)
    return [IrisSpecies(yi).name for yi in y]
