import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.svm import SVR
import optuna

# This is the core function that evaluates a model using k-fold cross validation
def evaluate_model(ModelClass, X, y, model_params=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = MultiOutputRegressor(ModelClass(**(model_params or {})))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Compute RMSE for each fold and store it
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        scores.append(rmse)

    return np.mean(scores)

# This runs Bayesian optimization to find good hyperparameters using Optuna
def optimize_hyperparams(ModelClass, X, y, param_space):
    def objective(trial):
        params = {k: v(trial) for k, v in param_space.items()}
        return evaluate_model(ModelClass, X, y, model_params=params)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    return study.best_params

# Main entry point: runs all supported models and prints their scores
def run_models(X, y, optimize=False):
    models = {
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'Ridge': Ridge,
        'BayesianRidge': BayesianRidge,
        'SVR': SVR
    }

    # Define hyperparameter search space for models that benefit from tuning
    param_spaces = {
        'RandomForest': {
            'n_estimators': lambda t: t.suggest_int('n_estimators', 50, 300),
            'max_depth': lambda t: t.suggest_int('max_depth', 3, 15),
        },
        'GradientBoosting': {
            'n_estimators': lambda t: t.suggest_int('n_estimators', 50, 300),
            'learning_rate': lambda t: t.suggest_float('learning_rate', 0.01, 0.3),
        },
        'Ridge': {
            'alpha': lambda t: t.suggest_float('alpha', 0.1, 10.0),
        },
        'SVR': {
            'C': lambda t: t.suggest_float('C', 0.1, 100.0),
            'epsilon': lambda t: t.suggest_float('epsilon', 0.01, 1.0),
        }
    }

    for name, ModelClass in models.items():
        print(f"\nRunning {name}...")

        if optimize and name in param_spaces:
            print("  Performing Bayesian optimization...")
            best_params = optimize_hyperparams(ModelClass, X, y, param_spaces[name])
            print(f"  Best hyperparameters: {best_params}")
        else:
            best_params = {}

        score = evaluate_model(ModelClass, X, y, model_params=best_params)
        print(f"  Average RMSE over folds: {score:.4f}")
