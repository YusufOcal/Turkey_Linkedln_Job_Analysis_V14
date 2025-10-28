import os
import joblib
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = ["jobWorkplaceTypes", "skill_categories", "exp_level_final"]
TARGET_COL = "apply_rate"


def build_parser():
    parser = argparse.ArgumentParser(description="Train baseline Logistic Regression classifier.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size fraction.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--output", type=str, default=str(MODEL_DIR / "logreg_classifier.pkl"), help="Path to save trained model.")
    return parser


def load_data(path: str):
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")
    df = df.dropna(subset=[TARGET_COL])

    threshold = df[TARGET_COL].quantile(0.75)
    y = (df[TARGET_COL] > threshold).astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def build_pipeline(num_features: list[str], cat_features: list[str]):
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    transformer = ColumnTransformer([
        ("cat", cat_encoder, cat_features),
        ("num", StandardScaler(with_mean=False), num_features)
    ])

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)

    pipeline = Pipeline([
        ("prep", transformer),
        ("model", clf)
    ])
    return pipeline


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("Loading data…")
    X, y = load_data(DATA_PATH)

    cat_features = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    pipeline = build_pipeline(num_features, cat_features)

    print("Training Logistic Regression…")
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_pred_proba)
    prc = average_precision_score(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc:.4f}\tPR-AUC: {prc:.4f}")

    print(f"Saving model → {args.output}")
    joblib.dump(pipeline, args.output)


if __name__ == "__main__":
    main() 