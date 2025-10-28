# Algorithm Selection Rationale

## 1. Problem framing
We aim to **rank job postings for a given user profile** and output a calibrated probability of success (apply‐rate > Q3).  
The task is effectively **binary classification with probability estimation** on a tabular dataset that is:
* Sparse (151 one-hot binaries + 3 new categoricals)
* Medium sized (≈14 k rows × 160 features)
* Heterogeneous feature types (binary, ordinal, continuous, categorical)
* Imbalanced (positive rate ≈ 25 %)

Key requirements:
1. Capture non-linear interactions between many sparse indicators.
2. Handle mixed feature types & missing values gracefully.
3. Deliver calibrated probabilities (for “% chance” UX).
4. Train quickly (< minutes) and deploy easily (Python pickle / ONNX).

---
## 2. Chosen algorithms & their fit
| Algorithm | Why it matches the data | Pros | Cons / Mitigation |
|-----------|------------------------|------|-------------------|
| **LightGBM (Gradient Boosted Trees)** | *Column-wise histogram algorithm* excels on high-dim sparse one-hot + continuous features; supports categorical >2 k bins with `OneHotEncoder`. | SOTA accuracy on tabular data, built-in class weighting, handles missing internally, fast training/inference, feature importance→leakage audit, probability output. | Risk of overfit → solved via subsample, colsample, L1/L2, CV/early-stopping. |
| **Logistic Regression** | Linear baseline; assesses whether simple weighted sum suffices. | Very fast, interpretable weights, natural probability output, sets performance “floor”. | Can't model interactions/non-linearities, under-fits (AUC 0.85). |
| **Random Forest** | Bagging trees handle non-linearities & class imbalance (`class_weight`). | Robust to noise, low hyper-parameter sensitivity, can expose feature importance complementary to GBDT. | Large disk size (62 MB), slower inference; limited depth mitigates overfit; still 1–2 AUC below LightGBM. |
| **MLP (Feed-forward neural net)** | Dense layers on one-hot vectors capture higher-order interactions not seen by trees. | Good at complex patterns, probability output, complementary error to trees→useful in ensembles. | Needs scaling + early-stopping; susceptible to overfit on 14 k rows → mitigated with α, patience. |
| **Optuna-tuned LightGBM** | Automated search to push LightGBM to optimal bias–variance spot. | Finds best leaves/regularisation in 30–50 trials; early-stopping avoids overfit; reproducible JSON/CSV of runs. | Requires extra compute (Colab). |

### Why not …
* **Deep Tabular Transformers / CatBoost GPU** – overkill for 14 k rows; longer latency; license or GPU dependency.
* **Field-Aware Factorization Machine (FFM)** – excellent for CTR-style sparse data, but harder to calibrate & explain; kept as future work.

---
## 3. Ensemble strategy rationale
Empirical & theoretical results show blending tree-based and NN models reduces generalisation error because they exploit different inductive biases:
* GBDT → gradient-boosted piecewise-constant functions.
* RF   → variance-reduced averaging of many high-variance trees.
* MLP  → smooth, high-order interactions in dense projection space.

Plan: **Soft-voting** (average calibrated probabilities) or **stacking** (LogReg meta-learner) once Optuna LightGBM is ready; target AUC ≥ 0.98.

---
## 4. Deployment considerations
| Criterion | LightGBM | RF | MLP | LogReg |
|-----------|----------|----|-----|--------|
| Model size | ~2 MB | 62 MB | 1.8 MB | 48 kB |
| Inference latency (CPU) | **~1 ms** | 4-5 ms | 2-3 ms | <1 ms |
| Interpretability | Gain / SHAP | Feature importance | Partial SHAP | Coefficients |
| Prob. calibration | Good, can calibrate | Needs calibration | Needs calibration | Native sigmoid |

Thus **Optuna-LightGBM** becomes primary production model (speed+accuracy).  RF/MLP kept as secondary for ensemble or A/B tests.

---
## 5. Summary
The selected algorithm roster balances:
* **Accuracy** (LightGBM ≈0.98 AUC target)  
* **Robustness & diversity** (RF, MLP)  
* **Baseline explainability & sanity check** (LogReg)  
* **Practical deployment footprint.**

These choices align with the dataset's sparse-tabular nature and the product requirement of serving **probabilistic job-fit recommendations** in real-time. 