# Micromagnetic GNN

This repository contains code and data for training Graph Neural Networks on micromagnetic grain structures. It contains the folders 'Coercivity_Prediction', 'BHMax_Prediction', 'Coercivity_Prediction_Uncertainty', and 'Coercivity_Prediction_OOD'. The folders 'PKL' and 'PKL_BHmax' contain the different node features and graphs, labels, edge features, extracted from micromagnetic simulations.

## Contents

Generally each folder - except PKLs - contains the files
- `config.yaml` : The GNN Configuration
- `data_loader.py` : Loading of data and preprocessing
- `models.py` : The respective GNN architectures
- `train.py` : Training of the model
- `evaluate.py` : Evaluation metrics and plotting of the trained model results
- `main.py` : The main script to run all files 

## How to use

After installing the required modules just run `python main.py`
Note: The 'Coercivity_Prediction_OOD' script should be run multiple times to account for randomness and identify the best-performing model weights.




