.PHONY: help environment remove dependencies model
.DEFAULT_GOAL := help

# Globals ---------------------------------------------------------------------

PROJECT_ENV=fastapi_demo
PYTHON=python


# Configuration ---------------------------------------------------------------

# Conda environment activation
ACTIVATE_ENV:=. $$(conda info --base)/etc/profile.d/conda.sh;conda activate $(PROJECT_ENV)

# Prepend Conda activation to Python
PYTHON:=$(ACTIVATE_ENV);$(PYTHON)


# Environment -----------------------------------------------------------------

environment:  ## Create the demo virtual environment
	conda env create --name $(PROJECT_ENV) --file="environment.yml"

remove:  ## Remove the demo virtual environment
	conda remove --name $(PROJECT_ENV) --all --yes

dependencies:  ## Export the environment snapshot to environment.yml
	conda env export --name $(PROJECT_ENV) --no-builds | grep -v "^prefix:" > "environment.yml"


# Pipeline --------------------------------------------------------------------

model:  # Run the training procedure
	$(PYTHON) model.py


# Debugging -------------------------------------------------------------------

debug:  # Run the API in debug mode
	$(ACTIVATE_ENV);uvicorn api:APP --reload
