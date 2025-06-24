import argparse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from fingerprints import get_fingerprints
from traditional_ml import run_models
from dlml import run_deep_learning

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
    parser = argparse.ArgumentParser(description="Membrane Gas Permeability Prediction Tool")
    parser.add_argument('--fp', type=str, choices=['morgan', 'maccs', 'topological', 'all'], required=True,
                        help='Fingerprinting method to use')
    parser.add_argument('--ml', type=str, choices=['sklearn', 'dl'], required=True,
                        help='Modeling method: sklearn or deep learning')
    parser.add_argument('--optimize', type=str, choices=['none', 'bayesian'], default='none',
                        help='Enable Bayesian optimization with Optuna')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file')
    return parser.parse_args()

def validate_and_prepare_data(df):
    if 'Monomer SMILES' not in df.columns or df['Monomer SMILES'].isnull().any():
        raise ValueError("Missing SMILES in input data. Please ensure all rows contain valid SMILES strings.")

    df = df.copy()

    # fill in defaults for missing non-critical columns
    for col, val in DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val  # if missing entirely, insert with default

    return df

def impute_permeabilities(df):
    # Identify all columns that contain both 'permeability' and 'Barrer'
    permeability_cols = [
        col for col in df.columns 
        if 'permeability' in col.lower() and 'barrer' in col.lower()
    ]

    if not permeability_cols:
        raise ValueError("No permeability columns found. Column names must include 'permeability' and 'Barrer'.")

    imputer = SimpleImputer(strategy='mean')
    df[permeability_cols] = imputer.fit_transform(df[permeability_cols])
    return df, permeability_cols

def main():
    args = parse_arguments()

    # load dataset
    data = pd.read_csv(args.input)

    # handle missing or invalid entries
    data = validate_and_prepare_data(data)
    data, target_cols = impute_permeabilities(data)

    # extract SMILES and metadata
    smiles = data['Monomer SMILES'].values
    meta = data[[
        'membrane thickness (micrometers)', 'h-bonding (yes/no)', 'ladder polymer (yes/no)',
        'treatment temperature (Celcius)', 'TFC (yes/no)', 'film age (days)'
    ]].copy()

    # convert yes/no categorical data to binary 0/1
    for col in meta.columns:
        if meta[col].dtype == object:
            meta[col] = (meta[col].str.lower() == 'yes').astype(int)

    # get molecular descriptors
    fp_matrix = get_fingerprints(smiles, method=args.fp)

    # combine fingerprint with meta features
    X = np.hstack((fp_matrix, meta.values))
    y = data[target_cols].values

    # choose modeling strategy
    if args.ml == 'sklearn':
        run_models(X, y, optimize=(args.optimize == 'bayesian'))
    elif args.ml == 'dl':
        run_deep_learning(X, y, smiles=smiles, optimize=(args.optimize == 'bayesian'), use_gnn=(args.fp == 'all'))
    else:
        raise ValueError("Unsupported ML method.")

if __name__ == '__main__':
    main()
