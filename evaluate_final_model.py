import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
MODEL_PATH = "job_apply_lgbm_pipeline.pkl"
TARGET_COL = "apply_rate"
LEAKAGE_COLS = ["apply_rate", "pop_views_log", "pop_applies_log"]


METRIC_FUNCS = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "log_loss": log_loss,
    "brier": brier_score_loss,
}


def prepare_xy(path: str):
    df = pd.read_csv(path)
    y = (df[TARGET_COL] > df[TARGET_COL].quantile(0.75)).astype(int)
    X = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    return X, y


def evaluate(pipe, X, y, n_splits: int = 5, random_state: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metric_scores_val = {k: [] for k in METRIC_FUNCS}
    metric_scores_train = {k: [] for k in METRIC_FUNCS}

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(X_train, y_train)
        prob_val = pipe.predict_proba(X_val)[:, 1]
        pred_val = (prob_val >= 0.5).astype(int)

        # Evaluate on validation
        for name, func in METRIC_FUNCS.items():
            if name in ["log_loss", "brier"]:
                metric_scores_val[name].append(func(y_val, prob_val))
            else:
                metric_scores_val[name].append(func(y_val, pred_val if name in ["accuracy", "precision", "recall", "f1"] else prob_val))

        # Evaluate on training for gap analysis
        prob_train = pipe.predict_proba(X_train)[:, 1]
        pred_train = (prob_train >= 0.5).astype(int)
        for name, func in METRIC_FUNCS.items():
            if name in ["log_loss", "brier"]:
                metric_scores_train[name].append(func(y_train, prob_train))
            else:
                metric_scores_train[name].append(func(y_train, pred_train if name in ["accuracy", "precision", "recall", "f1"] else prob_train))

    results = {}
    for name in METRIC_FUNCS:
        results[f"{name}_train_mean"] = float(np.mean(metric_scores_train[name]))
        results[f"{name}_train_std"] = float(np.std(metric_scores_train[name]))
        results[f"{name}_val_mean"] = float(np.mean(metric_scores_val[name]))
        results[f"{name}_val_std"] = float(np.std(metric_scores_val[name]))
        results[f"{name}_gap"] = results[f"{name}_train_mean"] - results[f"{name}_val_mean"]

    # RMSE from Brier
    rmse_train = [np.sqrt(v) for v in metric_scores_train["brier"]]
    rmse_val = [np.sqrt(v) for v in metric_scores_val["brier"]]
    results["rmse_train_mean"] = float(np.mean(rmse_train))
    results["rmse_train_std"] = float(np.std(rmse_train))
    results["rmse_val_mean"] = float(np.mean(rmse_val))
    results["rmse_val_std"] = float(np.std(rmse_val))
    results["rmse_gap"] = results["rmse_train_mean"] - results["rmse_val_mean"]

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved LightGBM pipeline on multiple metrics")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="CSV dataset path")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to .pkl pipeline")
    parser.add_argument("--out", type=str, default="final_model_metrics.json", help="JSON output filename")
    args = parser.parse_args()

    X, y = prepare_xy(args.data)
    pipe = load(args.model)

    results = evaluate(pipe, X, y)
    print(json.dumps(results, indent=2))

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main() 