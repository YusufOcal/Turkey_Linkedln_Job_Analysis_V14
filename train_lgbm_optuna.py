import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError:
    print("Optuna is required. Install with `pip install optuna lightgbm`.")
    sys.exit(1)

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

# -----------------------------
DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = ["jobWorkplaceTypes", "skill_categories", "exp_level_final"]
TARGET_COL = "apply_rate"
LEAKAGE_COLS = ["pop_applies_log", "pop_views_log", "apply_rate"]
DEFAULT_N_ESTIMATORS = 2000
CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50

# -----------------------------

def load_data(path: str):
    df = pd.read_csv(path)
    y = (df[TARGET_COL] > df[TARGET_COL].quantile(0.75)).astype(int)
    X = df.drop(columns=[TARGET_COL])
    X = X.drop(columns=[c for c in LEAKAGE_COLS if c in X.columns])
    return X, y


def build_transformer(cat_features):
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    return ColumnTransformer([
        ("cat", cat_encoder, cat_features)
    ], remainder="passthrough")


def create_model(trial: "optuna.trial.Trial"):
    params = {
        "n_estimators": DEFAULT_N_ESTIMATORS,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 32, 256, step=32),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60, step=10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5, step=0.05),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5, step=0.05),
        "objective": "binary",
        "n_jobs": -1,
        "random_state": 42,
    }
    return LGBMClassifier(**params)


def objective(trial):
    X, y = load_data(DATA_PATH)
    cat_features = [c for c in CATEGORICAL_COLS if c in X.columns]
    transformer = build_transformer(cat_features)
    model = create_model(trial)
    pipe = Pipeline([
        ("prep", transformer),
        ("model", model)
    ])

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(
            X_train,
            y_train,
        )
        pred = pipe.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, pred))

    mean_auc = np.mean(aucs)
    return mean_auc


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for LightGBM with CV.")
    parser.add_argument("--trials", type=int, default=40, help="Number of Optuna trials.")
    parser.add_argument("--output", type=str, default=str(MODEL_DIR / "lgbm_optuna_best.pkl"))
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("Best ROC-AUC:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Retrain model on full dataset with best params
    X, y = load_data(DATA_PATH)
    cat_features = [c for c in CATEGORICAL_COLS if c in X.columns]
    transformer = build_transformer(cat_features)
    best_model = LGBMClassifier(
        n_estimators=DEFAULT_N_ESTIMATORS,
        objective="binary",
        n_jobs=-1,
        random_state=42,
        **study.best_params,
    )
    best_pipe = Pipeline([
        ("prep", transformer),
        ("model", best_model)
    ])

    best_pipe.fit(X, y)
    joblib.dump(best_pipe, args.output)
    print(f"Saved best model to {args.output}")

    # Save study results
    df_trials = study.trials_dataframe()
    df_trials.to_csv(MODEL_DIR / "lgbm_optuna_trials.csv", index=False)
    print("Trial history saved to lgbm_optuna_trials.csv")


if __name__ == "__main__":
    main() 