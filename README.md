
THIS PROJECT IS NOT COOKED YET. THE FILES ARE JUST AS PLACEHOLDERS.

This project provides a modular Python toolkit for predicting gas permeabilities in polymer membranes using machine learning. It's designed with flexibility in mind, so individual labs can test and validate models on their own curated datasets. The goal is to provide both traditional machine learning and deep learning baselines, with optional hyperparameter optimization.

The input is expected to be a `.csv` file with the following columns:

- `Monomer SMILES`: required molecular structure input.
- `membrane thickness (micrometers)`
- `h-bonding (yes/no)`
- `ladder polymer (yes/no)`
- `treatment temperature (Celcius)`
- `TFC (yes/no)`
- `film age (days)`
- Gas permeabilities (multi-target regression outputs):
    - `Gas permeability(Barrer)`


You can put in as many gases as you like.

If SMILES is missing, the script will exit with an error. Other metadata columns will default to reasonable values if left empty. Permeability values will be imputed using the mean from the available data. In near future I hope to implement a wiser approach.

This tool can:

- Generate molecular fingerprints using RDKit (Morgan, MACCS, Topological).
- Use all three fingerprints together for a more complex descriptor.
- Run scikit-learn regressors with support for multi-output prediction.
- Train feedforward deep learning models using PyTorch.
- Perform 5-fold cross-validation.
- Optionally run Bayesian optimization via Optuna for selected models.


To use this project, clone the repository and install it in editable mode:

Installation instructions are coming soon when the project is ready!

Valid values for `--fp` are: `morgan`, `maccs`, `topological`, or `all`.
For `--ml`: choose between `sklearn` or `dl`.
To skip Bayesian optimization, just use `--optimize none`.

## Notes

GNN support is not yet implemented, but a placeholder is in place if you want to extend this toolkit using DeepChem or PyTorch Geometric.

This is intended to be used as a flexible testing environment. It's not a finalized predictive model but rather a scaffold to help you explore how different descriptors and algorithms perform on your membrane datasets.
See `requirements.txt` for more details.

---

Let us know if you run into issues or want to contribute improvements. This is an open foundation for lab-scale exploration.
