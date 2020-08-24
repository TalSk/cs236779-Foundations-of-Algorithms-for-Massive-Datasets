# cs236779 - Foundations of Algorithms for Massive Datasets - Final project
This repository contains files used for the final project of cs236779.

In this project, Collaborative Filtering techniques were explored, two of then implemented and experimented upon.

## Repository format
- Under the root directory, there are three Python files - `upwa.py` containing the implementation of the first method,
 `ncf_mlp.py` containing the implementation of the second method, and `utils.py` containing common classes and functions.
- `outputs` directory containing text output of different experiments as detailed in the project.
- `models` directory containing best checkpoints of the deep learning model for each set of parameters.
- `datasets` directory, containing the 3 datasets that were used in this project.
- `figures` directory, containing few excel sheets used to create some figures.

## How to run
Both `upwa.py` and `ncf_mlp.py` can be run directly, as they include a `main` with few calls to test function, the relevant one should be 
uncommented depending on the experiments you wish to run.

The tests depend on some relative paths to the datasets and models directories, so make sure to keep them alike the 
repository.