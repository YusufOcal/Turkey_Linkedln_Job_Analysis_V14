import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
CATEGORICAL_COLS = ["jobWorkplaceTypes", "skill_categories", "exp_level_final"]
TARGET_COL = "apply_rate"

# Heuristic leakage-prone cols to drop
LEAKAGE_COLS = ["pop_applies_log", "pop_views_log", "apply_rate"]


def load_data(path: str):
    df = pd.read_csv(path)
    y = (df[TARGET_COL] > df[TARGET_COL].quantile(0.75)).astype(int)
    X = df.drop(columns=[TARGET_COL])

    # drop suspected leakage cols if present
    X = X.drop(columns=[c for c in LEAKAGE_COLS if c in X.columns])
    return X, y


def build_pipeline(cat_features):
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    transformer = ColumnTransformer([
        ("cat", cat_encoder, cat_features)
    ], remainder="passthrough")

    clf = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="binary",
        n_jobs=-1,
    )

    return Pipeline([
        ("prep", transformer),
        ("model", clf)
    ])


def main():
    parser = argparse.ArgumentParser(description="LightGBM 5-fold CV to inspect overfitting.")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    X, y = load_data(DATA_PATH)
    cat_features = [c for c in CATEGORICAL_COLS if c in X.columns]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    roc_scores, pr_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = build_pipeline(cat_features)

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val, y_pred)
        pr = average_precision_score(y_val, y_pred)
        roc_scores.append(roc)
        pr_scores.append(pr)
        print(f"Fold {fold}: ROC {roc:.4f} | PR {pr:.4f} | best_iters {pipe.named_steps['model'].best_iteration_}")

    print("\nCV Summary:")
    print(f"ROC-AUC: {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
    print(f"PR-AUC : {np.mean(pr_scores):.4f} ± {np.std(pr_scores):.4f}")


if __name__ == "__main__":
    main() 