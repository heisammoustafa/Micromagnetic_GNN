# Micromagnetic GNN

This repository contains code and data for training Graph Neural Networks on micromagnetic grain structures. It contains the folders 'Coercivity_Prediction', 'BHMax_prediction', 'Coercivity_Uncertainty', 'Coercivity_Generalization', and 'BHMax_Generalization'. The folder 'PKL' contains the different node features and graphs, labels, edge features, extracted from micromagnetic simulations.

## Contents

Generally each folder - except PKL - contains the files
- `config.yaml` : The GNN Configuration
- `data_loader.py` : Loading of data and preprocessing
- `models.py` : The respective GNN architectures
- `train.py` : Training of the model
- `evaluate.py` : Evaluation metrics and plotting of the trained model results
- `main.py` : The main script to run all files 

## How to use

After installing the required models just run `python main.py`


