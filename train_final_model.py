import argparse
from pathlib import Path
from typing import List

import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------
# Configuration
# ---------------------------
DEFAULT_DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
DEFAULT_OUT_PATH = "job_apply_lgbm_pipeline.pkl"
TARGET_COL = "apply_rate"
LEAKAGE_COLS = ["pop_applies_log", "pop_views_log", "apply_rate"]
CATEGORICAL_COLS = ["jobWorkplaceTypes", "skill_categories", "exp_level_final"]

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    n_jobs=-1,
)


# ---------------------------
# Utility functions
# ---------------------------

def load_data(path: str):
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")
    y = (df[TARGET_COL] > df[TARGET_COL].quantile(0.75)).astype(int)
    X = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    return X, y


def build_preprocessor(cat_cols: List[str], numeric_cols: List[str]):
    transformers = []
    if cat_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
        transformers.append(("cat", ohe, cat_cols))
    if numeric_cols:
        transformers.append(("num", StandardScaler(with_mean=False), numeric_cols))
    return ColumnTransformer(transformers, remainder="passthrough")


# ---------------------------
# Main
# ---------------------------


def main():
    parser = argparse.ArgumentParser(description="Train final LightGBM model and save as .pkl pipeline")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="CSV dataset path")
    parser.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH, help="Output .pkl filename")
    args = parser.parse_args()

    X, y = load_data(args.data_path)

    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = build_preprocessor(cat_cols, num_cols)

    model = LGBMClassifier(**LGBM_PARAMS)
    pipe = Pipeline([("prep", preprocessor), ("model", model)])

    print("Fitting LightGBM pipeline on full dataset…")
    pipe.fit(X, y)

    dump(pipe, args.out_path)
    print(f"Pipeline saved to {args.out_path}")


if __name__ == "__main__":
    main() 