import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import optuna
    try:
        from optuna.integration import LightGBMPruningCallback  # Optuna <4.0
    except ImportError:
        from optuna.integration.lightgbm import LightGBMPruningCallback  # Optuna 4.x
except ImportError:
    print("Optuna is required. Install with `pip install optuna lightgbm`. ")
    sys.exit(1)

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = ["jobWorkplaceTypes", "skill_categories", "exp_level_final"]
TARGET_COL = "apply_rate"
LEAKAGE_COLS = ["pop_applies_log", "pop_views_log", "apply_rate"]

CV_FOLDS_TUNE = 3  # faster search
CV_FOLDS_EVAL = 5  # final check


def load_and_encode():
    """Load dataset, drop leakage, one-hot encode once, return sparse matrix."""
    df = pd.read_csv(DATA_PATH)
    y = (df[TARGET_COL] > df[TARGET_COL].quantile(0.75)).astype(int).values
    X = df.drop(columns=[TARGET_COL])
    X = X.drop(columns=[c for c in LEAKAGE_COLS if c in X.columns])

    cat_features = [c for c in CATEGORICAL_COLS if c in X.columns]
    ohe = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ], remainder="passthrough")

    X_sparse = ohe.fit_transform(X)  # csr_matrix
    return X_sparse, y, ohe


def tune_lightgbm(X, y, n_trials):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=32),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60, step=10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5, step=0.05),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5, step=0.05),
            "n_estimators": 10000,
            "verbosity": -1,
        }

        skf = StratifiedKFold(n_splits=CV_FOLDS_TUNE, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

            gbm = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_val],
                verbose_eval=False,
                early_stopping_rounds=50,
                callbacks=[LightGBMPruningCallback(trial, "auc")],
            )
            aucs.append(gbm.best_score["valid_0"]["auc"])
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def train_best_full(X, y, ohe, params):
    params = params.copy()
    params.update({"objective": "binary", "metric": "auc", "verbosity": -1})
    skf = StratifiedKFold(n_splits=CV_FOLDS_EVAL, shuffle=True, random_state=42)
    aucs = []
    best_iters = []
    for train_idx, val_idx in skf.split(X, y):
        clf = lgb.LGBMClassifier(**params, n_estimators=10000)
        clf.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            eval_metric="auc",
            early_stopping_rounds=50,
            verbose=False,
        )
        proba = clf.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], proba))
        best_iters.append(clf.best_iteration_)

    print(f"Final 5-fold AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # retrain on full data with average best_iteration
    n_estimators_final = int(np.mean(best_iters)) if any(best_iters) else 1000
    final_clf = lgb.LGBMClassifier(**params, n_estimators=n_estimators_final)
    final_clf.fit(X, y, verbose=False)

    return Pipeline([
        ("prep", ohe),
        ("model", final_clf)
    ])


def main():
    parser = argparse.ArgumentParser(description="Fast Optuna LightGBM tuning with pre-encoded features.")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--output", type=str, default=str(MODEL_DIR / "lgbm_optuna_fast.pkl"))
    args = parser.parse_args()

    print("Loading & encoding data…")
    X, y, ohe = load_and_encode()
    print("Tuning LightGBM hyperparameters…")
    study = tune_lightgbm(X, y, args.trials)

    print("Best params:", study.best_params)
    best_model = train_best_full(X, y, ohe, study.best_params)
    joblib.dump(best_model, args.output)
    print(f"Saved final model → {args.output}")

    study.trials_dataframe().to_csv(MODEL_DIR / "lgbm_optuna_fast_trials.csv", index=False)


if __name__ == "__main__":
    main() 