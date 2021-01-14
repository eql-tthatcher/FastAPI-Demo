'''Demo XGBoost Model API'''

# Imports ---------------------------------------------------------------------

# Standard Library
from pathlib import Path

# External Libraries
from fastapi import FastAPI
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


@APP.post('/predict', response_model=List[IrisSpecies])
async def predict(measurements: List[IrisMeasurements]):
    X = MODEL.design_matrix(measurements)
    decoder = IrisSpecies.decoder()
    return [decoder[y] for y in MODEL.predict(X)]
