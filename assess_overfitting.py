import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


# ---------------------------
# Configuration
# ---------------------------
DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
TARGET_COL = "apply_rate"
# Columns that cause target leakage and must be removed from the feature set
LEAKAGE_COLS = ["pop_applies_log", "pop_views_log", "apply_rate"]
# Categorical columns that require one-hot encoding
CATEGORICAL_COLS = ["jobWorkplaceTypes", "skill_categories", "exp_level_final"]

# Models (keep relatively small to avoid long runtimes)
MODELS: Dict[str, object] = {
    "LogisticRegression": LogisticRegression(max_iter=3000, solver="lbfgs", n_jobs=-1, C=1.0),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42,
        n_jobs=-1,
    ),
}


def load_data(path: str):
    """Read CSV and split into features / binary target."""
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in the provided dataset."
        )

    # Convert apply_rate (continuous) to binary: label = 1 if above 75th percentile
    y = (df[TARGET_COL] > df[TARGET_COL].quantile(0.75)).astype(int)
    X = df.drop(columns=[TARGET_COL])

    # Remove leakage-prone columns if present
    X = X.drop(columns=[c for c in LEAKAGE_COLS if c in X.columns])
    return X, y


def build_preprocessor(cat_cols: List[str], numeric_cols: List[str]):
    """Create a ColumnTransformer for categorical and numerical features."""
    transformers = []
    if cat_cols:
        # Handle API change: 'sparse' was renamed to 'sparse_output' in scikit-learn >=1.2.
        # Attempt to use the new parameter first and fall back for older versions.
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            # For scikit-learn <1.2
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

        transformers.append(
            (
                "cat",
                ohe,
                cat_cols,
            )
        )
    if numeric_cols:
        transformers.append(("num", StandardScaler(with_mean=False), numeric_cols))

    return ColumnTransformer(transformers, remainder="passthrough")


def cross_validate_model(X: pd.DataFrame, y: pd.Series, model_name: str, model):
    """Return training and validation scores across CV folds for a given model."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_roc, val_roc = [], []
    train_pr, val_pr = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
        num_cols = [c for c in X.columns if c not in cat_cols]
        preprocessor = build_preprocessor(cat_cols, num_cols)

        pipe = Pipeline(
            steps=[("prep", preprocessor), ("model", model)]
        )
        pipe.fit(X_train, y_train)

        # Probabilities for positive class
        prob_train = pipe.predict_proba(X_train)[:, 1]
        prob_val = pipe.predict_proba(X_val)[:, 1]

        # Metrics
        train_roc.append(roc_auc_score(y_train, prob_train))
        val_roc.append(roc_auc_score(y_val, prob_val))

        train_pr.append(average_precision_score(y_train, prob_train))
        val_pr.append(average_precision_score(y_val, prob_val))

        print(
            f"{model_name} | Fold {fold} | Train ROC {train_roc[-1]:.4f} | Val ROC {val_roc[-1]:.4f} | Train PR {train_pr[-1]:.4f} | Val PR {val_pr[-1]:.4f}"
        )

    results = {
        "train_roc_mean": np.mean(train_roc),
        "train_roc_std": np.std(train_roc),
        "val_roc_mean": np.mean(val_roc),
        "val_roc_std": np.std(val_roc),
        "train_pr_mean": np.mean(train_pr),
        "train_pr_std": np.std(train_pr),
        "val_pr_mean": np.mean(val_pr),
        "val_pr_std": np.std(val_pr),
    }

    # Additionally store the average gap (overfitting indicator)
    results["roc_gap"] = results["train_roc_mean"] - results["val_roc_mean"]
    results["pr_gap"] = results["train_pr_mean"] - results["val_pr_mean"]
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Assess overfitting for several models using 5-fold cross-validation."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="Path to the CSV dataset (default: %(default)s)",
    )
    args = parser.parse_args()

    X, y = load_data(args.data_path)

    summary_rows = []
    for name, model in MODELS.items():
        print(f"\n=== Evaluating {name} ===")
        res = cross_validate_model(X, y, name, model)
        summary_rows.append({"model": name, **res})

    summary_df = pd.DataFrame(summary_rows)
    print("\n===== Cross-validation summary =====")
    print(summary_df[[
        "model",
        "train_roc_mean",
        "val_roc_mean",
        "roc_gap",
        "train_pr_mean",
        "val_pr_mean",
        "pr_gap",
    ]].to_string(index=False, float_format="{:.4f}".format))

    # Save detailed results
    out_path = Path(args.data_path).with_suffix("").name + "_overfitting_report.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main() 