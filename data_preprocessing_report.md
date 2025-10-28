# Data Preparation & Feature Engineering Report

## 1. Source datasets
| File | Rows | Cols | Notes |
|------|-----:|-----:|-------|
| `final_dataset_ml_ready_numeric_plus.csv` | 13 591 | 160 | Base numeric feature set (+ pop/urgency/optwin etc.) |
| `final_dataset_ml_ready_v9.csv`           | 13 591 | 139 | Contains `jobWorkplaceTypes`, `skill_categories`, `exp_level_final` |

Both files share identical row order (same job postings). **Numeric + plus** was chosen as foundation because it already includes engineered popularity & business features.

---
## 2. Merge step
Script snippet (see terminal log 14 Jun 2025 01:54):
```python
plus = pd.read_csv('final_dataset_ml_ready_numeric_plus.csv')
v9   = pd.read_csv('final_dataset_ml_ready_v9.csv')[
           ['jobWorkplaceTypes','skill_categories','exp_level_final']]
combined = plus.join(v9)            # row-wise join
combined['promosyon_var'] = 0       # future user input
```
**New columns added (4)**
1. `jobWorkplaceTypes` – Remote / On-site / Hybrid  
2. `skill_categories` – Coarse skill buckets  
3. `exp_level_final` – Categorical experience level  
4. `promosyon_var` – placeholder 0/1 (user can set via UI)

Result saved as **`final_dataset_ml_ready_numeric_plus_extended.csv`**  
Shape → **13 591 × 164**.

---
## 3. Missing-value handling
| Column | Missing | Action |
|--------|--------:|--------|
| `exp_level_ord` | 584 | Filled with median (3.0) |
| **All others** | 0 | Already 0/"Unknown" |

After filling, dataset has **0 missing values**.

---
## 4. Column type summary
* Binary one-hot  : 151  
* Continuous      : 8    (e.g. `recency_score`, `competition_level`)  
* New object cats : 3    (`jobWorkplaceTypes`, `skill_categories`, `exp_level_final`)  
* Helper bool     : 1    (`promosyon_var`)

---
## 5. Leakage inspection
Columns identified as potential target leakage when modelling **apply_rate**:
1. `apply_rate`    – direct target statistic  
2. `pop_views_log` – future popularity (views)  
3. `pop_applies_log` – future popularity (applies)

These were **excluded** from training & CV scripts (`train_lgbm_cv.py`, `evaluate_models_cv.py`). A visual table is exported in `leakage_columns.png`.

---
## 6. Final dataset state (for modelling)
* File    : `final_dataset_ml_ready_numeric_plus_extended.csv`
* Rows    : 13 591  
* Columns : 164 (but 161 used for modelling after leakage removal)  
* Missing : 0  
* Ready for ML: One-hot + numeric, no NaNs, leakage cols flagged.

---
## 7. Importance for downstream pipeline
* Provides the richest feature space (function, industry, size, skill, workplace, popularity, urgency…).
* Compatible with Streamlit user inputs (all selectable fields reflected as one-hot or categorical).
* Clean of missing data and leakage, ensuring fair cross-validation and generalizable models.

*Report generated 14 Jun 2025.* 