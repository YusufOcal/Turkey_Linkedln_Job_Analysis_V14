# Job Recommender – Version 1 (Basic)

## 1 Veri Girişi & Önişleme

```python
import pandas as pd

df = pd.read_csv("final_dataset_ml_ready_numeric_plus_extended_with_title.csv")
```

| Sütun | İçerik |
|-------|--------|
| `title` | İlan başlığı (string) |
| `jobWorkplaceTypes` | Uzaktan / hibrit / ofis vb. (kategori) |
| `skill_categories` | `|` ayraçlı beceri etiketleri |
| `recency_score` | İlanın sisteme yüklenmesinden bu yana geçen **gün** |
| `apply_rate` | Tarihsel başvuru–görüntüleme oranı (hedef) |

Ek sütunların hiçbiri (şehir, sektör vb.) bu sürümde kullanılmaz.

`recency_score` normalize edilir:

\[
 r_i = \frac{\text{recency}\_i - r_{\min}}{r_{\max}-r_{\min}} \in [0,1]
\]

## 2 Tahmin Modeli

LightGBM sınıflandırıcısı `apply_rate > Q3` etiketine göre eğitilmiştir.

\[
\hat p_i = f_{\text{LGBM}}(x_i) \quad \in [0,1]
\]

Platt (sigmoid) veya isotonic kalibrasyonu **yoktur**.

## 3 Filtre Mantığı

Kenar çubuğu yalnızca iki seçim sunar:

* `skills` – çoklu seçim (list of string)
* `workplaceType` – tek seçim veya **herhangi**

Mask formülü

```python
mask = True
if skills:
    mask &= df["skill_categories"].str.contains('|'.join(skills))
if wp != 'herhangi':
    mask &= df["jobWorkplaceTypes"] == wp
```

## 4 Match Score ( `match_ratio` )

Sadece beceri ve workplace türü ikili eşleşmesinin ortalaması:

\[
 m_i = \frac{\mathbb{1}[\text{skillMatch}] + \mathbb{1}[\text{wpMatch}]}{2}
\]

## 5 Nihai Skor Fonksiyonu

```python
score = 0.7 * prob + 0.3 * (1 - r)
```

Matematiksel olarak

\[
 s_i = 0.7\,\hat p_i \; + \; 0.3\,(1-r_i)
\]

* **0.7** — Model olasılığının temel belirleyici olduğu varsayımı
* **0.3** — Yeni ilanları öne çıkarma (küçük `r` daha taze ⇒ büyük skor)

## 6 Sıralama & Çıktı

1. `df_sub = df[mask]`
2. `score` hesaplanır ve sütun olarak eklenir.
3. `df_sub.nlargest(10, "score")` ilk 10 satır alınır.
4. Her satır için:
   * Başlık (`title`)
   * `score: {:.2f}`

Progress bar **yoktur**.

## 7 Ağırlıkların Gerekçesi

| Katsayı | Gerekçe |
|---------|---------|
| 0.7 (prob) | ROC-AUC ≈ 0.97 – model en güvenilir sinyal |
| 0.3 (recency) | Kullanıcıların yeni ilanlara yatkın olduğu varsayımı |

## 8 Çıktının Oluşumu – Örnek

1. İlan: _"Python Developer"_
2. LightGBM olasılığı: 0.82
3. İlan 12 günlük ⇒ `r = 12/120 ≈ 0.10`
4. Skor
   \[
   s = 0.7·0.82 + 0.3·0.9 = 0.574 + 0.27 = 0.844
   \]
5. İlk 10 listesinde 3. sırada görüntülenir.

---

Bu sürüm en yalın önerici olup, karmaşık filtre sistemi veya ceza/bonus mekanizmaları içermez. Ağırlıklar sabittir; skor formülü tek, deterministiktir. 