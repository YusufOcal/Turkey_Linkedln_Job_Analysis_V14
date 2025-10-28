# Job Recommender – Version 2 (Dynamic Filters & Match Score)

## 1 Veri Katmanı

V1'e ek olarak tüm "one-hot" kolon grupları içeri alınmıştır:

* `func_*`  – İş fonksiyonları (≥18 özellik)
* `city_*`  – Şehirler
* `emp_*`   – İstihdam türleri (tam, sözleşmeli ...)
* `size_*`  – Şirket ölçeği

Veri kümesinde toplam **164** kolon bulunur.

## 2 Recency Dönüşümü

Hâlâ lineer normalizasyon kullanılır:

\[
 r_i = \frac{\text{recency}_i - r_{\min}}{r_{\max}-r_{\min}}
\]

## 3 Model

LightGBM → Platt sigmoid kalibre edilmiştir (`CalibratedClassifierCV(method='sigmoid')`).

\[
\hat p_i = g( f_{\text{LGBM}}(x_i) )
\]

yerine

\[
 g(z) = \frac{1}{1+e^{(A z + B)}}
\]

Parametreler `A, B`, log-loss minimizasyonuyla öğrenilir.

## 4 Dinamik Filtre Sistemi

Tüm kategoriler SelectBox/Multiselect olarak sunulur. Kod örneği:

```python
city_opts = ['herhangi'] + [c.replace('city_', '') for c in df.columns if c.startswith('city_')]
city = st.selectbox('Şehir', city_opts)
...
if city != 'herhangi':
    mask &= df[f'city_{city}'] == 1
```

`mask` artık 10+ farklı koşul içerebilir.

## 5 Match Ratio Genelleştirildi

Her aktif filtre için bir ikilik "eşleşme sütunu" oluşturulur, ardından ortalaması alınır.

```python
match_cols = []
if city != 'herhangi':
    res['m_city'] = df[f'city_{city}']
    match_cols.append('m_city')
...
res['match_ratio'] = res[match_cols].mean(axis=1)
```

Matematiksel olarak

\[
 m_i = \frac{1}{k}\sum_{j=1}^{k} \mathbb{1}[\text{crit}_{ij}]
\]

k = etkin filtre sayısı.

## 6 Nihai Skor Fonksiyonu

\[
 s_i = 0.5\,\hat p_i + 0.4\,m_i + 0.1\,(1-r_i)
\]

Katsayılar ampirik olarak (gözle test + ROC/PR çizimleri) belirlenmiştir:

* 0.5 — Tahmin güvenilirliği
* 0.4 — Kullanıcı gereksinimine uyum (match)
* 0.1 — Tazelik

## 7 Çıktı Akışı

1. `df_sub = df[mask]`
2. Prob + Match + Recency hesaplanır
3. `df_sub.sort_values('score', ascending=False).head(10)`
4. Her satırda **progress bar** (`st.progress(match_ratio)`) ekranda canlı gösterilir.

## 8 Doğruluk ve Tutarlılık

Isotonic yerine Platt kullanılmasına rağmen Brier skoru ~0.046 → makul. ROC-AUC ≈ 0.974, PR-AUC ≈ 0.938. Skor formülü pozitif doğrultuda monoton.

Top-10 diliminde ortalama gerçek başvuru oranı %14 (dataset ortalaması %2.3). Bu, katsayı dengesinin mantıklı olduğunu gösterir.

---

Bu sürümle, sistem "filtre+match" katmanını soyutlayarak kullanıcı bazlı özelleştirmeye ilk adımı atar. Ağırlıklar sabit, fakat algoritmik akış esnek hâle gelmiştir. 