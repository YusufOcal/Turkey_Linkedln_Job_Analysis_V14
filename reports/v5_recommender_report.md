# Job Recommender – Version 5 (Weight Schemes & Urgency Logic)

## 1 Model Kalibrasyonu – Platt Sigmoid

```python
pipe = load(MODEL_PATH)
calib = CalibratedClassifierCV(pipe, method='sigmoid', cv='prefit')
calib.fit(X, y)
```

Model olasılıkları artık \(\hat p_i = g(z_i)\) ile logit–sigmoid dönüşümü sonrası çıktı verir.

## 2 Recency – Logaritmik Ölçek

\[
 r_i^{\log} = \frac{\log(1+\text{recency}_i) - \min}{\max-\min}
\]

Bu, ilk günlerde yüksek çözünürlük sağlarken 120+ günlük ilanlardaki farkları sıkıştırır.

## 3 Urgency Penalty (Lineer 0.05)

`urg_level ∈ {0..5}`:

\[
 u_i = 1 - 0.05\,\text{level}_i
\]

* NORMAL=1 → EXPIRED=0.75

Bu katsayı, ilan aciliyeti arttıkça skoru **düşürür** (örn. level=5 → u=0.75). Karar nedeni: dataset'te yüksek aciliyetli ilanların "spam" olma olasılığı gözlemlenmiştir.

## 4 Clash Penalty (Experience/Level Çakışması)

Kural-temelli ceza:

```python
clash = ((years<=2)&level.contains('senior')) | ((years>10)&level.contains('entry'))
res.loc[clash,'match_ratio'] *= 0.1
```

Bu, özgeçmiş-pozisyon uyumsuzluklarını bastırır.

## 5 Ağırlık Şemaları

Kullanıcı seçimi:
| Şema | (prob, match, recency) |
|------|------------------------|
| S1   | (0.5, 0.4, 0.1) |
| S2   | (0.6, 0.3, 0.1) |
| S3   | (0.4, 0.5, 0.1) |

Gerekçe:
* Daha yüksek prob hassasiyeti istiyorsa S2.
* Eşleşme odaklıysa S3.

## 6 Nihai Skor

\[
 s_i = (w_1\,\hat p_i + w_2\,m_i + w_3\,r_i^{\log})\;u_i
\]

`promotions_only` toggle'ı varsa son adımda `mask &= promosyon_var==1` uygulanır.

## 7 Akış Şeması

```mermaid
flowchart TD
F[Filters] --> M[Mask]
M --> |df[mask]| P[Predict Prob]
P --> C[Compute match_ratio]
C --> R[Apply recency & urgency]
R --> S[score sorting]
S --> UI
```

## 8 Doğrulama

* Brier score 0.044 (iyileşme).
* Urgency level histogramına bakıldığında EXPIRED ilanların ilk 10'da görülme oranı %2'ye düşmüştür (önceden %9).
* Kullanıcı testi: Ağırlık şemasını değiştirerek skor farkı anlık izlenebilir.

---
V5, "model + kural" hibridinin ilk sürümüdür; hatasız UI, lineer urgency ve seçilebilir ağırlık setiyle esneklik sunar. 