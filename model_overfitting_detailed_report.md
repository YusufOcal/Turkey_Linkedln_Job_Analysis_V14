# Detailed Modelling & Over-Fitting Assessment Report

> Dataset: `final_dataset_ml_ready_numeric_plus_extended.csv`  
> Rows: **13 591** job postings | Columns: **164** (after merge & new features)  
> Target: **binary** – 1 if `apply_rate` > 75-th percentile, else 0

---

## 1. Data Preparation Pipeline

| Step | Description |
|------|-------------|
| **Leakage removal** | Columns known to leak post–publishing popularity were dropped:  
`apply_rate`, `pop_views_log`, `pop_applies_log`. |
| **Categorical features** | 3 text categorical columns ⇒ **One-Hot Encoding**.<br>`OneHotEncoder(handle_unknown="ignore", sparse_output=True*)`<br>*fallback to `sparse=True` for scikit-learn &lt; 1.2.* |
| **Numeric features** | All remaining continuous / binary columns are passed through after **`StandardScaler(with_mean=False)`** to keep sparse matrix efficient. |
| **ColumnTransformer** | `remainder="passthrough"` retains the ~150 one-hot binaries already present (e.g. `func_*`, `ind_*`, `city_*` …). |
| **Cross-validation** | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` ensures class balance and reproducibility. |

---

## 2. Model Portfolio & Hyper-Parameters

All models were chosen to represent complementary inductive biases ‑- linear, bagging, boosting.

| Model | Key Hyper-Parameters |
|-------|----------------------|
| **Logistic Regression** | `solver="lbfgs"`, `max_iter=3000`, `C=1.0`, `n_jobs=-1` |
| **Random Forest** | `n_estimators=300`, `max_depth=None` (fully grown), `min_samples_leaf=2`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1` |
| **LightGBM** | `n_estimators=500`, `learning_rate=0.05`, `num_leaves=64`, `subsample=0.8`, `colsample_bytree=0.8`, `objective="binary"`, `random_state=42`, `n_jobs=-1` |

> *Note*: All models are wrapped in a `Pipeline` after the `ColumnTransformer`, so the pre-processing is included in every cross-validation split and receives no information from the validation fold.

---

## 3. Cross-Validation Results

### 3.1 Per-Fold Metrics (ROC-AUC / PR-AUC)

```
LogisticRegression | Fold 1 | Train ROC 0.885 | Val ROC 0.842 | Train PR 0.719 | Val PR 0.612
LogisticRegression | Fold 2 | Train ROC 0.881 | Val ROC 0.862 | Train PR 0.709 | Val PR 0.676
LogisticRegression | Fold 3 | Train ROC 0.882 | Val ROC 0.862 | Train PR 0.706 | Val PR 0.680
LogisticRegression | Fold 4 | Train ROC 0.879 | Val ROC 0.871 | Train PR 0.706 | Val PR 0.670
LogisticRegression | Fold 5 | Train ROC 0.882 | Val ROC 0.860 | Train PR 0.714 | Val PR 0.658

RandomForest        | Fold 1 | Train ROC 0.999 | Val ROC 0.977 | Train PR 0.997 | Val PR 0.942
RandomForest        | Fold 2 | Train ROC 0.999 | Val ROC 0.976 | Train PR 0.997 | Val PR 0.939
RandomForest        | Fold 3 | Train ROC 0.999 | Val ROC 0.976 | Train PR 0.997 | Val PR 0.934
RandomForest        | Fold 4 | Train ROC 0.999 | Val ROC 0.978 | Train PR 0.997 | Val PR 0.945
RandomForest        | Fold 5 | Train ROC 0.999 | Val ROC 0.976 | Train PR 0.997 | Val PR 0.941

LightGBM            | Fold 1 | Train ROC 1.000 | Val ROC 0.978 | Train PR 1.000 | Val PR 0.949
LightGBM            | Fold 2 | Train ROC 1.000 | Val ROC 0.975 | Train PR 1.000 | Val PR 0.942
LightGBM            | Fold 3 | Train ROC 1.000 | Val ROC 0.974 | Train PR 1.000 | Val PR 0.937
LightGBM            | Fold 4 | Train ROC 1.000 | Val ROC 0.976 | Train PR 1.000 | Val PR 0.936
LightGBM            | Fold 5 | Train ROC 1.000 | Val ROC 0.975 | Train PR 1.000 | Val PR 0.939
```

### 3.2 Aggregate Summary

| Model | ROC-AUC (train) | ROC-AUC (val) | Δ Gap | PR-AUC (train) | PR-AUC (val) | Δ Gap |
|-------|----------------:|--------------:|------:|---------------:|-------------:|------:|
| Logistic Regression | **0.882 ± 0.002** | **0.859 ± 0.009** | **0.022** | **0.711 ± 0.005** | **0.659 ± 0.025** | **0.052** |
| Random Forest       | **0.999 ± 0.000** | **0.976 ± 0.001** | **0.022** | **0.997 ± 0.000** | **0.940 ± 0.004** | **0.056** |
| LightGBM            | **1.000 ± 0.000** | **0.976 ± 0.001** | **0.024** | **1.000 ± 0.000** | **0.941 ± 0.005** | **0.059** |

---

## 4. Interpretation & Discussion

### 4.1 Why these Models?
* **Logistic Regression** – baseline linear model, interpretable coefficients, acts as sanity-check for signal strength.
* **Random Forest** – ensemble of decision trees captures non-linearities & interactions without heavy tuning; robust to outliers;   provides feature importance by permutation or impurity.
* **LightGBM** – gradient-boosting on decision leaves, state-of-the-art for tabular data; handles high-dimensional sparse one-hot features efficiently; supports built-in handling of class imbalance and regularisation.

### 4.2 Strengths & Weaknesses Observed
| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| Logistic Regression | Fast to train; interpretable sign & magnitude; **no overfit** (small gap). | Linear assumption misses complex patterns ⇒ lower absolute performance. |
| Random Forest | Very strong ROC/PR; small gap; robust to feature scaling. | Large model size (memory); explaining predictions less straightforward; may struggle with very sparse high-dimensional data speed-wise. |
| LightGBM | Best PR-AUC; fastest inference among tree ensembles; granular feature importance. | Needs hyper-parameter tuning to avoid slight overfit; gap marginally larger; requires monotonicity checks if business constraints apply. |

### 4.3 Over-Fitting Counter-Measures & Outcomes
1. **Target Leakage Removal** – critical popularity metrics (`apply_rate`, logs of views/applies) removed → prevented artificially inflated hold-out scores.
2. **Cross-Validation** – 5-fold stratified CV gives realistic variance estimates; gaps ~0.02–0.06 indicate limited overfit.
3. **Regularisation / Sampling**
   * Logistic Reg. uses L2 penalty (`C=1.0`).
   * RandomForest uses `min_samples_leaf=2`, bagging & feature subsampling implicitly regularise.
   * LightGBM uses `subsample` & `colsample_bytree` at 0.8 plus moderate `num_leaves`.
4. **Result** – Despite very high training AUCs for tree ensembles, validation AUCs remain > 0.94 with similar gaps to Logistic Regression, evidencing that most predictive power comes from genuine features rather than noise.

---

## 5. Recommendations & Next Steps
1. **Hyper-parameter Fine-Tuning** – grid / Bayesian search around LightGBM's `num_leaves`, `min_child_samples`, and `feature_fraction` might push PR-AUC further.
2. **Probability Calibration** – apply Platt scaling or isotonic regression for better calibrated apply-probabilities if used downstream.
3. **Feature Importance & SHAP** – analyse top drivers for actionable insights (e.g., which skill categories or locations boost application likelihood).
4. **External Hold-Out** – if a truly unseen temporal split exists, validate there to confirm generalisation over time.
5. **Model Size vs. Latency Trade-Off** – Logistic Regression could still be attractive for real-time scoring when latency/memory budget is tight, albeit with lower accuracy.

---

*Report generated automatically via `assess_overfitting.py` outputs (commit ‹current›).* 