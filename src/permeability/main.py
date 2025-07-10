import argparse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from fingerprints import get_fingerprints
from traditional_ml import run_models
from dlml import run_deep_learning
import sys

# constants for default values
DEFAULTS = {
    'membrane thickness (micrometers)': 100.0,
    'h-bonding (yes/no)': 'no',
    'ladder polymer (yes/no)': 'no',
    'treatment temperature (Celcius)': 25.0,
    'TFC (yes/no)': 'no',
    'film age (days)': 0.0,
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Gas Permeability Prediction")

    parser.add_argument("--mode", required=True, choices=["train", "predict"],
                        help="Select operation mode.")

    parser.add_argument("--input", required=True, type=str,
                        help="Path to the input CSV file.")

    parser.add_argument("--ml", required=True, choices=["sklearn", "dl"],
                        help="ML backend to use: sklearn (traditional) or dl (deep learning).")

    parser.add_argument("--model", choices=[
        "linear", "ridge", "lasso", "elasticnet", "svr",
        "decision_tree", "random_forest", "gradient_boosting",
        "xgboost", "lightgbm", "adaboost", "bagging", "mlp", "all"
    ], help="Model to train or use for prediction.")

    parser.add_argument("--fingerprint", choices=["morgan", "maccs", "topological", "combined"],
                        help="Fingerprinting method for SMILES.")

    parser.add_argument("--model_path", type=str,
                        help="Path to the trained model file for prediction.")

    parser.add_argument("--optimize", choices=["none", "bayesian"], default="none",
                        help="Enable hyperparameter optimization (only applies to training).")

    args = parser.parse_args()

    # Validation logic
    if args.mode == "train":
        if not args.model:
            sys.exit("Error: --model is required in training mode.")
        if not args.fingerprint:
            sys.exit("Error: --fingerprint is required in training mode.")
        if args.ml == "sklearn" and args.model == "mlp":
            sys.exit("Error: 'mlp' is only valid for --ml dl.")
        if args.ml == "dl" and args.model != "mlp":
            sys.exit("Error: Only 'mlp' is supported with --ml dl.")
    elif args.mode == "predict":
        if not args.model_path:
            sys.exit("Error: --model_path is required in prediction mode.")

    return args

def validate_and_prepare_data(df):
    if 'Monomer SMILES' not in df.columns or df['Monomer SMILES'].isnull().any():
        raise ValueError("Missing SMILES in input data.")

    df = df.copy()

    for col, val in DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val

    return df

def impute_permeabilities(df):
    permeability_cols = [
        col for col in df.columns 
        if 'permeability' in col.lower() and 'barrer' in col.lower()
    ]

    if not permeability_cols:
        raise ValueError("No permeability columns found.")

    imputer = SimpleImputer(strategy='mean')
    df[permeability_cols] = imputer.fit_transform(df[permeability_cols])
    return df, permeability_cols

def main():
    args = parse_arguments()

    # Load data
    data = pd.read_csv(args.input)
    data = validate_and_prepare_data(data)
    data, target_cols = impute_permeabilities(data)

    # SMILES and meta
    smiles = data['Monomer SMILES'].values
    meta = data[[
        'membrane thickness (micrometers)', 'h-bonding (yes/no)',
        'ladder polymer (yes/no)', 'treatment temperature (Celcius)',
        'TFC (yes/no)', 'film age (days)'
    ]].copy()

    for col in meta.columns:
        if meta[col].dtype == object:
            meta[col] = (meta[col].str.lower() == 'yes').astype(int)

    # Fingerprints
    fp_matrix = get_fingerprints(smiles, method=args.fingerprint)

    # Feature matrix
    X = np.hstack((fp_matrix, meta.values))
    y = data[target_cols].values

    if args.ml == 'sklearn':
        run_models(X, y, model_name=args.model, optimize=(args.optimize == 'bayesian'))
    elif args.ml == 'dl':
        run_deep_learning(X, y, smiles=smiles, optimize=(args.optimize == 'bayesian'), use_gnn=(args.fingerprint == 'combined'))
    else:
        raise ValueError("Unsupported ML backend.")

if __name__ == '__main__':
    main()
