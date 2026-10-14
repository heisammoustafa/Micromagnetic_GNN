# Micromagnetic GNN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23360688.svg)](https://doi.org/10.5281/zenodo.23360688)

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

## Paper

This repository accompanies the paper [Graph neural networks to predict coercivity and maximum energy product of hard magnetic microstructures](https://www.sciencedirect.com/science/article/pii/S0304885325008261?via%3Dihub) (*Journal of Magnetism and Magnetic Materials* 634, 173594, 2025). See [Citation](#citation) below.

## Licensing

- **Code**: licensed under the [MIT License](LICENSE).
- **Data** (the `PKL/` and `PKL_BHmax/` folders): licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [DATA_LICENSE.md](DATA_LICENSE.md) for details and the citation.

## Citation

If you use this work, please cite:

Moustafa, H., Kovacs, A., Fischbacher, J., Gusenbauer, M., Ali, Q., Breth, L., Schrefl, T., Oezelt, H. (2025). Graph neural networks to predict coercivity and maximum energy product of hard magnetic microstructures. *Journal of Magnetism and Magnetic Materials*, 634, 173594. [https://doi.org/10.1016/j.jmmm.2025.173594](https://doi.org/10.1016/j.jmmm.2025.173594)

The code and data of this repository are archived on Zenodo: [https://doi.org/10.5281/zenodo.23360688](https://doi.org/10.5281/zenodo.23360688) (this DOI represents all versions and always resolves to the latest one).

```bibtex
@article{moustafa_graph_2025,
	title = {Graph neural networks to predict coercivity and maximum energy product of hard magnetic microstructures},
	volume = {634},
	issn = {03048853},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S0304885325008261},
	doi = {10.1016/j.jmmm.2025.173594},
	language = {en},
	journal = {Journal of Magnetism and Magnetic Materials},
	author = {Moustafa, H. and Kovacs, A. and Fischbacher, J. and Gusenbauer, M. and Ali, Q. and Breth, L. and Schrefl, T. and Oezelt, H.},
	month = nov,
	year = {2025},
	pages = {173594},
}
```

## Acknowledgement

This work was funded by the Austrian Science Fund (FWF) [10.55776/I6159](https://www.fwf.ac.at/forschungsradar/10.55776/I6159).
