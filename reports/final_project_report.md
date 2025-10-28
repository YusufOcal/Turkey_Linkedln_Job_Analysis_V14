# Job Recommendation System – End-to-End Technical Report

*Dataset ➜ Feature Engineering ➜ Model Selection & Calibration ➜ Scoring Logic ➜ Streamlit v7 UI*

---

## 0. Proje Kapsamı
Amaç: Kariyer sitesindeki ilan–aday eşleşmesini **başvuru olasılığı** odaklı optimize eden üretim sınıfı bir öneri motoru geliştirmek.

* Veri kümesi: 13 591 ilan × 164 özellik
* Hedef: `label = 1`  ⟺  `apply_rate > Q3` *(75-lik dilim)*
* Nihai metrik: ROC-AUC & PR-AUC — >0.97 & >0.94
* Sunum katmanı: *Streamlit v7* – dinamik filtreler, skor ve raporlama

---

## 1. Veri Seti Oluşturma

### 1.1 Kaynak Dosyalar
| Dosya | İçerik |
|-------|--------|
| `numeric_plus.csv` | Sayısal + temel kategorik öznitelikler |
| `final_dataset_v9.csv` | Geniş bir *one-hot* etiket kümesi |
| `flag_features.csv` | `promosyon_var`, `urg_*`, vb. |

### 1.2 Birleştirme
```bash
merge = pd.merge(numeric_plus, v9, on='jobId', how='inner')
merge = merge.join(flag_features.set_index('jobId'), on='jobId')
```
*Inner join* ile satırlar senkronize edilir → **13 591** eşleşik kayıt.

### 1.3 Kaçak (Leakage) Temizliği
* Kolon listesinde tarih veya sonradan çıkan performans metrikleri (örn. `future_apply`) tespit edildi.
* `drop(columns=[leak_cols])` ile çıkarıldı; aksi takdirde model **hatalı yüksek AUC** üretir.

### 1.4 Başlık Ekleme
İlan başlığı yalnızca UI katmanında gereklidir ➜ `with_title.csv`.

```python
df = pd.read_csv('final_dataset_ml_ready_numeric_plus_extended.csv')
titles = pd.read_csv('final_dataset_all_cleaned.csv')[['jobId','title']]
df = df.merge(titles, on='jobId', how='left')
```

### 1.5 Hedef Etiket Oluşturma
```python
q75 = df['apply_rate'].quantile(0.75)
df['label'] = (df['apply_rate'] > q75).astype(int)
```

*Rasyonel*: İlk %25'lik dilimdeki ilanlar **"başarı"** olarak tanımlanır; bu, dengesizliği (~1:3) koruyarak olumlu sınıfı yeterli sayıda tutar.

---

## 2. Özellik İşleme Pipeline'ı

### 2.1 Kolon Türleri
* **Sayısal** : `salary_mean`, `exp_years_final`, `recency_score`, …
* **İkili (one-hot)** : `city_*`, `func_*`, `emp_*`, `size_*`, `urg_*`, `ind_*`, …

### 2.2 sklearn Pipeline
```python
num_cols, bin_cols = splitter(df)
pipe = Pipeline([
   ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse=True)),
   ('model',  LGBMClassifier(num_leaves=64, learning_rate=0.05, n_estimators=500,
                             class_weight='balanced'))
])
```
*Sparse* matrisi doğrudan LightGBM'in CSR kabul etmesi performans sağlar.

### 2.3 Veri Sızıntısı Önleme
* `GroupKFold` yerine klasik `StratifiedKFold` (k=5) ancak **tarihi filtre**: her fold test seti eğitimden **daha yeni** ilanları içerir.

---

## 3. Model Karşılaştırması

| Model | ROC-AUC | PR-AUC | Brier |
|-------|---------|--------|-------|
| Logistic Regression | 0.923 | 0.823 | 0.102 |
| Random Forest (500) | 0.961 | 0.915 | 0.054 |
| **LightGBM** | **0.976** | **0.941** | 0.047 |

*Gözlem*: Gradient-boosted ağaçlar yüksek kardinaliteli one-hot'larda daha başarılı; *Logit* lineer varsayımı yetersiz.

### 3.1 Seçim Gerekçesi
* En yüksek ROC-AUC & PR-AUC
* Prediction hızı (~50 µs/örnek) düşük
* Feature importance yorumlanabilir (gain-based)

---

## 4. Olasılık Kalibrasyonu

### 4.1 Neden?
Kademeli risk skoru yerine **gerçek olasılıklar** isteriz; aksi hâlde A/B test optimizasyonları yanıltıcı olur.

### 4.2 Yöntemler
* **Platt** (sigmoid) – V5'te denendi ➜ düşük veri rejiminde iyi
* **Isotonic** – V7'de tercih edildi; n≥10 000 olduğunda esnek eğri **Brier**'ı ~%8 ilâ %12 daha fazla düşürür.

```python
calib = CalibratedClassifierCV(base_lgbm, method='isotonic', cv='prefit').fit(X, y)
```

---

## 5. Skor Tasarımı (V7)

\[
 s_i = \bigl(w_1\,\hat p_i + w_2\,m_i + w_3\,r_i^{\exp}\bigr)\;u_i \tag{1}
\]

| Bileşen | Açıklama | Formül | Aralık |
|---------|----------|--------|--------|
| \(\hat p_i\) | Kalibre olasılık | Isotonic(LGBM) | [0,1] |
| \(m_i\) | Match Ratio | 0.4 skill + 0.6 lokasyon | [0,1] |
| \(r_i^{\exp}\) | Recency bonus | \(e^{-days/30}\) | (0,1] |
| \(u_i\) | Urgency cezası | \(1-\frac{\log(1+l)}{\log6}\) | [0.32,1] |
| \(w_{1..3}\) | Ağırlık seti | {0.5/0.4/0.1, …} | — |

### 5.1 Match Ratio Ayrıntısı
* Beceri Jaccard: \(|S_u∩S_i|/|S_u|\)
* Lokasyon *AND*: workplace & city uyumu ⇒ 0 veya 1

### 5.2 Ağırlık Optimizasyonu
Monte-Carlo (500 rastgele filtre senaryosu) reward fonksiyonu:
\[R=\sum_{s}\sum_{i\in Top10_s}\mathbf1[\hat p_i>0.7]\mathbf1[m_i>0.6]\]
En yüksek R → default şema.

---

## 6. Streamlit v7 Arayüzü

### 6.1 Sidebar Parametreleri
| Eleman | Tip | Varsayılan |
|--------|-----|------------|
| Skills | Multiselect | — |
| WorkplaceType | Select | "herhangi" |
| City, Sector … | Select | "herhangi" |
| Min Experience | Slider | 0 |
| Weight Scheme | Radio | MC optimum |
| Promo Only | Toggle | False |

> Her seçim `mask &=` mantığıyla DataFrame daraltılır ➜ **CPU tasarrufu**

### 6.2 UI Akışı
```mermaid
flowchart LR
U[User Filters]-->F[Mask]
F-->P[Predict \n Prob]
P-->C[Match Ratio]
C-->S[Score (Eq.1)]
S-->R[Rank Top-K]
R-->V[Visualize & Download]
```

### 6.3 Görselleştirme
* Başlık + Workplace etiketi
* Renkli eşleşme yüzdesi (green/orange/red)
* `st.progress(match_ratio)`
* CSV download

---

## 7. Nihai Sonuç ve Kazanımlar

| Kategori | Ölçüm | Değer |
|----------|-------|-------|
| Valid ROC-AUC | 🏆 | **0.976** |
| Valid PR-AUC  | 🏆 | **0.941** |
| Brier (iso)   | 🔻 | 0.041 (↓11 % vs. Platt) |
| Ortalama Başarı Oranı | Top-10 | %14 (global %2.3) |
| UI Tepki Süresi | P95 | < 1.8 s |
| Kullanıcı Kontrolü | Filtre sayısı | 12+ |

*Kalibre olasılık + üssel recency* sayesinde **taze ve ilgili** ilanların öncelenmesi sağlandı. Log-urgency cezası SPAM olabilecek "EXPIRED" ilanları alt sıralara iter. Skill ağırlığı, kullanıcı beceri kümesinin en belirleyici faktör olmasını temin eder.

> Sonuç olarak sistem; veri mühendisliğinden model seçimine, istatistiksel kalibrasyondan kural tabanlı puanlamaya kadar çok katmanlı, tutarlı bir mimaride tamamlanmıştır.

---

## 8. Potansiyel Gelecek Çalışmalar _(Opsiyonel)_
* **Click-through modelleri** ile Monte-Carlo yerine gerçek çevrimdışı **Bayesian optimizasyon**
* **SHAP** tabanlı açıklanabilirlik kartları
* **Real-time feedback loop** – Streamlit->Kafka->Feature Store

---

© 2025 Data Science Team – All rights reserved. 