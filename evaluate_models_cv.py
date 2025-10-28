import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended.csv"
CATEGORICAL_COLS = ["jobWorkplaceTypes", "skill_categories", "exp_level_final"]
TARGET_COL = "apply_rate"
LEAKAGE_COLS = ["pop_applies_log", "pop_views_log", "apply_rate"]

MODELS = {
    "LogReg": LogisticRegression(max_iter=2000, solver="lbfgs", C=0.5, n_jobs=-1),
    "RF": RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=5, n_jobs=-1, class_weight="balanced"),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=300, early_stopping=True, random_state=42),
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = (df[TARGET_COL] > df[TARGET_COL].quantile(0.75)).astype(int)
    X = df.drop(columns=[TARGET_COL])
    X = X.drop(columns=[c for c in LEAKAGE_COLS if c in X.columns])
    return X, y


def build_transformer(cat_features):
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("cat", cat_encoder, cat_features),
        ("num", StandardScaler(with_mean=False), [c for c in X.columns if c not in cat_features])
    ])


if __name__ == "__main__":
    X, y = load_data()
    cat_features = [c for c in CATEGORICAL_COLS if c in X.columns]
    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ], remainder="passthrough")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in MODELS.items():
        aucs, prs = [], []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipe = Pipeline([
                ("prep", transformer),
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, proba))
            prs.append(average_precision_score(y_val, proba))

        print(f"{name}: ROC-AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f} | PR-AUC {np.mean(prs):.4f} ± {np.std(prs):.4f}") 