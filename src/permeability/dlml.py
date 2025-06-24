import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import optuna

# Simple feedforward model for multi-output regression
def build_mlp(input_dim, output_dim, hidden_dim=128):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )

# Main training loop for a single fold
def train_one_fold(model, X_train, y_train, X_val, y_val, device):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Fixed number of epochs; can make this tunable later
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val_tensor).cpu().numpy()
        true = y_val_tensor.cpu().numpy()
    return np.sqrt(mean_squared_error(true, preds))

# Same training logic wrapped in Optuna objective
def optuna_objective(trial, X, y, input_dim, output_dim, device):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 50, 200)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32).to(device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32).to(device)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_train)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val).cpu().numpy()
            targets = y_val.cpu().numpy()
            scores.append(np.sqrt(mean_squared_error(targets, preds)))

    return np.mean(scores)

# High-level runner function; called from main.py
def run_deep_learning(X, y, smiles=None, optimize=False, use_gnn=False):
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_gnn:
        raise NotImplementedError("GNN support not implemented yet.")

    if optimize:
        print("\nRunning Bayesian optimization for neural net...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: optuna_objective(trial, X, y, input_dim, output_dim, device), n_trials=30)
        print("  Best params:", study.best_params)
        hidden_dim = study.best_params['hidden_dim']
    else:
        hidden_dim = 128  # fallback default if not optimizing

    print("\nEvaluating model with 5-fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        model = build_mlp(input_dim, output_dim, hidden_dim)
        score = train_one_fold(model, X[train_idx], y[train_idx], X[val_idx], y[val_idx], device)
        scores.append(score)

    print(f"  Average RMSE over folds: {np.mean(scores):.4f}")
