.PHONY: help environment install_environment remove_environment export_environment
.DEFAULT_GOAL := help

### Globals ######################################

PROJECT_ENV=api_demo
PYTHON=python


### Configuration ################################

# Prepend Conda activation to Python
PYTHON:=. $$(conda info --base)/etc/profile.d/conda.sh;conda activate $(PROJECT_ENV);$(PYTHON)


### Environment ##################################

install_environment:  ## Create the demo virtual environment
	conda env create --name $(PROJECT_ENV) --file="environment.yml"

remove_environment:  ## Remove the demo virtual environment
	conda remove --name $(PROJECT_ENV) --all --yes

export_environment:  ## Export the environment snapshot to environment.yml
	conda env export --name $(PROJECT_ENV) --no-builds | grep -v "^prefix:" > "environment.yml"
