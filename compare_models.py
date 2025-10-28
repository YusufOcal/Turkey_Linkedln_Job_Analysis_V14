import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
LEAKAGE_COLS = ["apply_rate", "pop_views_log", "pop_applies_log"]
TARGET_COL = "apply_rate"


def load_Xy(path: str):
    df = pd.read_csv(path)
    y = (df[TARGET_COL] > df[TARGET_COL].quantile(0.75)).astype(int)
    X = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    return X, y


def evaluate(pipe, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_scores, pr_scores = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipe.fit(X_train, y_train)
        prob_val = pipe.predict_proba(X_val)[:, 1]
        roc_scores.append(roc_auc_score(y_val, prob_val))
        pr_scores.append(average_precision_score(y_val, prob_val))
    return np.mean(roc_scores), np.std(roc_scores), np.mean(pr_scores), np.std(pr_scores)


def main():
    parser = argparse.ArgumentParser(description="Compare two saved model pipelines via 5-fold CV")
    parser.add_argument("model_a", type=str, help="Path to first model (.pkl)")
    parser.add_argument("model_b", type=str, help="Path to second model (.pkl)")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="CSV dataset path")
    args = parser.parse_args()

    X, y = load_Xy(args.data)

    for label, path in [("Model A", args.model_a), ("Model B", args.model_b)]:
        print(f"\n=== Evaluating {label}: {Path(path).name} ===")
        pipe = load(path)
        roc_mean, roc_std, pr_mean, pr_std = evaluate(pipe, X, y)
        print(f"ROC-AUC: {roc_mean:.4f} ± {roc_std:.4f}\tPR-AUC: {pr_mean:.4f} ± {pr_std:.4f}")


# -----------------------------------------------------------------------------
# Patch for loading pickles generated with older scikit-learn (<1.2)
# They sometimes reference _RemainderColsList, which was removed.
# We dynamically add a dummy class to avoid AttributeError during unpickling.
# -----------------------------------------------------------------------------

try:
    from sklearn.compose import _column_transformer as _ct

    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Backport stub for compatibility while unpickling."""

        _ct._RemainderColsList = _RemainderColsList  # type: ignore
except Exception:
    # If import fails, continue – patch not needed
    pass


if __name__ == "__main__":
    main() 