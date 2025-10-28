import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = ["jobWorkplaceTypes", "skill_categories", "exp_level_final"]
TARGET_COL = "apply_rate"


def build_parser():
    parser = argparse.ArgumentParser(description="Train MLPClassifier on job dataset.")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(MODEL_DIR / "mlp_classifier.pkl"))
    parser.add_argument("--layers", type=str, default="128,64", help="Comma-separated hidden layer sizes")
    return parser


def load_data(path: str):
    df = pd.read_csv(path)
    threshold = df[TARGET_COL].quantile(0.75)
    y = (df[TARGET_COL] > threshold).astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def build_pipeline(num_features, cat_features, hidden_layer_sizes, random_state):
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    transformer = ColumnTransformer([
        ("cat", cat_encoder, cat_features),
        ("num", StandardScaler(), num_features)
    ])

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=random_state,
        early_stopping=True,
    )

    return Pipeline([
        ("prep", transformer),
        ("model", clf)
    ])


def main():
    parser = build_parser()
    args = parser.parse_args()
    hidden = tuple(int(x) for x in args.layers.split(",") if x.strip())

    X, y = load_data(DATA_PATH)
    cat_features = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    pipe = build_pipeline(num_features, cat_features, hidden, args.random_state)

    print("Training MLP…")
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