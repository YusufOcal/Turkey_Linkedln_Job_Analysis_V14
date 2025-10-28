import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = ["jobWorkplaceTypes", "skill_categories", "exp_level_final"]
TARGET_COL = "apply_rate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RandomForest classifier on job dataset.")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--output", type=str, default=str(MODEL_DIR / "rf_classifier.pkl"))
    return parser


def load_data(path: str):
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError("Target column not found")
    threshold = df[TARGET_COL].quantile(0.75)
    y = (df[TARGET_COL] > threshold).astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def build_pipeline(num_features: list[str], cat_features: list[str], n_estimators: int, max_depth: int, random_state: int):
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    transformer = ColumnTransformer([
        ("cat", cat_encoder, cat_features),
        ("num", StandardScaler(with_mean=False), num_features),
    ])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )

    return Pipeline([
        ("prep", transformer),
        ("model", clf),
    ])


def main():
    parser = build_parser()
    args = parser.parse_args()

    X, y = load_data(DATA_PATH)
    cat_features = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    pipe = build_pipeline(num_features, cat_features, args.estimators, args.max_depth, args.random_state)

    print("Training RandomForest…")
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)
    prc = average_precision_score(y_test, y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc:.4f}\tPR-AUC: {prc:.4f}")

    print(f"Saving model to {args.output}")
    joblib.dump(pipe, args.output)


if __name__ == "__main__":
    main() 