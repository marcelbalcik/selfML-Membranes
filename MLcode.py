###Placeholder until the final version of the code is published.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, BaggingRegressor, HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

import deepchem as dc

# === Step 1: Load dataset from CSV ===
file_path = "./membrane_data_50.csv"
data = pd.read_csv(file_path)

# === Step 2: Clean and prepare data ===

# Drop rows where any target value is missing (precautionary, shouldn't happen with clean data)
target_cols = ['CO2', 'CH4', 'H2', 'O2', 'N2']
data.dropna(subset=target_cols, inplace=True)

# Remove rows with missing or invalid SMILES strings
featurizer = dc.feat.CircularFingerprint(size=4096, radius=5)

def is_valid_smiles(smiles):
    try:
        return smiles is not None and featurizer.featurize([smiles])[0] is not None
    except:
        return False

data = data[data['SMILES'].apply(is_valid_smiles)]

# Create the Morgan fingerprint generator
fp_generator = GetMorganGenerator(radius=5, fpSize=1024)

# Function to convert SMILES to fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(fp_generator.GetFingerprint(mol))

# Generate and stack fingerprints
data['features'] = data['SMILES'].apply(smiles_to_fp)
data = data[data['features'].notna()]  # Drop invalid SMILES
X_fp = np.stack(data['features'].values)

# === Step 3: Handle additional features ===

X_other = data[['thickness', 'h_bonding', 'ladder']]
y = data[target_cols]

# Column transformer for preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), ['thickness']),
    
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='if_binary'))
    ]), ['h_bonding', 'ladder'])
])

X_other_processed = preprocessor.fit_transform(X_other)
X_combined = np.hstack([X_fp, X_other_processed.toarray() if hasattr(X_other_processed, "toarray") else X_other_processed])

# === Step 4: Split dataset ===

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# === Step 5: Define models ===

regressors = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'KNeighbors': KNeighborsRegressor(),
    'SVR': SVR(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'ExtraTrees': ExtraTreesRegressor(),
    'Bagging': BaggingRegressor(),
    'HistGradientBoosting': HistGradientBoostingRegressor(),
    'MLPRegressor': MLPRegressor(max_iter=500)
}

if xgb_available:
    regressors['XGBoost'] = XGBRegressor()

# === Step 6: Train and evaluate models ===

results = []

for gas in y.columns:
    for name, model in regressors.items():
        model.fit(X_train, y_train[gas])
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test[gas], preds)
        mae = mean_absolute_error(y_test[gas], preds)
        r2 = r2_score(y_test[gas], preds)
        results.append({
            'Gas': gas,
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })

results_df = pd.DataFrame(results)

print(results_df)  # Print in console

# Optional: Save results to CSV
results_df.to_csv("model_results.csv", index=False)
