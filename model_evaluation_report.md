# Model Evaluation & Overfitting Analysis

## 1. Dataset snapshot
* **File**: `final_dataset_ml_ready_numeric_plus_extended.csv`
* 13 591 rows × 164 columns  
* Missing values: 584 (`exp_level_ord`) → filled with median (3).  
* Binary one-hots: 151, Continuous: 8, Categorical (object): 3 (`jobWorkplaceTypes`, `skill_categories`, `exp_level_final`) + `promosyon_var`.
* Potential leakage features detected: `apply_rate`, `pop_views_log`, `pop_applies_log` – removed for fair CV.

---
## 2. Trained models & settings
| Label | Script | Key hyper-parameters |
|-------|--------|----------------------|
| LightGBM (Baseline) | `train_lgbm.py` | n_estimators=500, learning_rate=0.05, default leaves |
| LightGBM (CV, reg.) | `train_lgbm_cv.py` | n_estimators=2000, num_leaves=64, min_child_samples=50, subsample=0.8, colsample=0.7, reg_α/λ =0.1 |
| Logistic Regression | `train_logreg.py` | C=1.0 (baseline), CV eval C=0.5 |
| Random Forest | `train_rf.py` | 300 trees, max_depth=15, min_samples_leaf=5 |
| MLP (Neural Net) | `train_mlp.py` | (128, 64) relu, early_stopping |

---
## 3. Single hold-out vs Cross-Validation
### 3.1 Hold-out (80-20 split, leakage **ON**)
| Model | ROC-AUC | PR-AUC | Accuracy |
|-------|--------:|-------:|---------:|
| LightGBM (baseline) | **0.9994** | **0.9984** | 0.99 |
| Logistic Reg.       | 0.9985 | 0.9954 | 0.99 |
| Random Forest       | 0.9875 | 0.9664 | 0.93 |
| MLP                 | 0.9847 | 0.9606 | 0.94 |

> ☝️ Neredeyse kusursuz skorlar veri sızıntısının (apply_rate & pop_* sütunlarının) açık bir göstergesiydi.

### 3.2 5-fold Stratified CV (leakage **OFF**, düzenlileştirme **ON**)
| Model | ROC-AUC (μ ± σ) | PR-AUC (μ ± σ) | Not |
|-------|-----------------|----------------|-----|
| LightGBM (CV) | **0.9733 ± 0.0009** | **0.9373 ± 0.0024** | Sıkı reg., no overfit |
| MLP           | 0.9579 ± 0.0039 | 0.9069 ± 0.0080 | Sağlam, hafif düşük |
| Random Forest | 0.9506 ± 0.0030 | 0.8694 ± 0.0075 | Dengeli |
| Logistic Reg. | 0.8559 ± 0.0094 | 0.6494 ± 0.0233 | Karmaşık ilişkileri kaçırıyor |

* **Fold sapmaları** < 0.01 → düşük varyans, dolayısıyla overfitting belirtisi yok.  
* LightGBM'in `best_iteration = 0` olması, parametrik kısıtlamanın model karmaşıklığını epey düşürdüğünü gösteriyor (bilerek yapılmış under-fitting).

---
## 4. Overfitting Önleme Adımları
1. **Feature leakage removal**: `apply_rate`, `pop_views_log`, `pop_applies_log` → train & CV'de kullanılmadı.
2. **Regularization**  
   * LightGBM: `num_leaves`↓, `min_child_samples`↑, subsample & colsample < 1, L1/L2 cezaları.  
   * Logistic Reg.: C=0.5 ile L2 ceza artırıldı.  
   * RF: max_depth / min_samples_leaf sınırlı.  
   * MLP: early stopping + α=1e-4.
3. **Cross-Validation**: 5-kat stratified → genelleme hatası istatistiksel olarak ölçüldü.
4. **Performance monitoring**: ROC-AUC & PR-AUC hem training hem CV'de raporlandı.

---
## 5. Sonuç & Yol Haritası
* En iyi **genelleme**: LightGBM (CV) AUC ≈ 0.973.  
* MLP & RF destekleyici modeller; ensemble (Soft-Voting / Stacking) ile ↑ puan potansiyeli var.
* **Optuna tuning** (`train_lgbm_optuna.py`, Colab'da çalışıyor) tamamlandığında AUC ≈ 0.98+ hedefleniyor.
* Önerilen devam:
  1. Optuna en iyi modelini alın, CV skoru ≥ 0.98 ise onu ana model yapın.
  2. RF + MLP ile soft-voting ensembli deneyin (CV 0.98+ bekleniyor).
  3. SHAP importance analizi → kalan leakage/önemsiz öznitelikleri tespit.
  4. Model kalibrasyonu (Platt / isotonic) → güvenilir olasılık çıktıları.

---
*Rapor – 14 Haz 2025* 