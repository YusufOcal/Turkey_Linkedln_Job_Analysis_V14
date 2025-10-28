import random
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import roc_auc_score, average_precision_score

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended_with_title.csv"
MODEL_PATH = "job_apply_lgbm_pipeline.pkl"

CRITERIA_SETS = [
    {
        "skills": ["Python", "Machine Learning"],
        "workplace": "remote",
        "exp_year": 3,
        "emp": "full-time",
    },
    {
        "skills": ["Sales", "Negotiation"],
        "workplace": "on-site",
        "exp_year": 1,
        "emp": "part-time",
    },
    {
        "skills": ["Project Management"],
        "workplace": "hybrid",
        "exp_year": 5,
        "emp": "full-time",
    },
]


def load_artifacts():
    df = pd.read_csv(DATA_PATH)
    pipe = load(MODEL_PATH)
    # remove title for model features
    X = df.drop(columns=[c for c in ["title"] if c in df.columns])
    prob = pipe.predict_proba(X)[:, 1]
    df = df.assign(model_prob=prob)
    return df


def global_metrics(df):
    # use model probability vs true label for sanity
    y = (df["apply_rate"] > df["apply_rate"].quantile(0.75)).astype(int)
    roc = roc_auc_score(y, df["model_prob"])
    pr = average_precision_score(y, df["model_prob"])
    return roc, pr


def random_inputs(df):
    # create random criteria values existing in dataset
    sample = {
        "skills": random.choice(df["skill_categories"].dropna()).split("|")[:2],
        "workplace": random.choice(df["jobWorkplaceTypes"].dropna().unique()),
        "exp_year": int(df["exp_years_final"].median()),
        "emp": random.choice([c.replace("emp_", "") for c in df.columns if c.startswith("emp_")]),
    }
    return sample


def main():
    df = load_artifacts()
    roc, pr = global_metrics(df)
    print(f"Global sanity check: ROC-AUC {roc:.3f}, PR-AUC {pr:.3f}\n")

    # Display test scenarios
    scenarios = CRITERIA_SETS + [random_inputs(df)]
    for i, sc in enumerate(scenarios, 1):
        print(f"=== Scenario {i} ===")
        print(sc)
        # simple filter by workplace and skills to compute match ratio similarly
        mask = df["jobWorkplaceTypes"] == sc["workplace"]
        skill_set = set(sc["skills"])
        def mratio(row):
            skills = set(str(row).split("|"))
            return len(skills & skill_set)/len(skill_set) if skill_set else 1
        df["tmp_match"] = df["skill_categories"].apply(mratio)
        # composite
        df["tmp_score"] = 0.6*df["model_prob"] + 0.4*df["tmp_match"]
        top = df[mask].sort_values("tmp_score", ascending=False).head(10)
        if top.empty:
            print("No jobs match this scenario\n")
            continue
        print(top[["title", "tmp_score", "tmp_match", "model_prob"]].head(5).to_string(index=False))
        print()

    # Suggestions
    print("=== Suggestions ===")
    print("1. Tune composite weight coefficients using grid-search on a validation set with simulated user clicks.")
    print("2. Add penalty for soon-to-expire jobs via 'job_urgency' to prioritise active postings.")
    print("3. Calibrate LightGBM probabilities (Platt scaling) for better match-score blending.")


if __name__ == "__main__":
    main() 