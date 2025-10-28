# Job Recommender – Version 4 (Full Filter Set & Robust Defaults)

## Özet
V4'te algoritmik skor formülü değişmemiş, ancak **filtre yapısı** tamamen genişletilmiştir. Hedef: Kullanıcı hatalarından (selectbox index, boş dataframe) kaynaklanan kesintileri ortadan kaldırmak.

## 1 Dinamik Filtre Keşfi

Kod:
```python
for prefix in ["emp_", "size_", "ind_", "func_", "city_", "urg_"]:
    opts = [c.replace(prefix, "") for c in df.columns if c.startswith(prefix)]
```
Bu desenle, çalışma anında kolon seti ne olursa olsun ilgili selectbox oluşur.

### Safe Default
`safe_idx()` fonksiyonu, varsayılan "herhangi" değerini bulamazsa `0` döner → `StreamlitInvalidIndex` hatası engellenir.

## 2 Mask Mantığı

Her selectbox için koşul eklenir:
```python
if size != 'herhangi':
    mask &= df[f'size_{size}'] == 1
```
Bu yapı 10+ özellik için tekrarlanır.  Boş DataFrame riskini azaltmak adına üç koruma eklenmiştir:
* `if mask.sum()==0:` uyarı ve `st.stop()`
* Varsayılan `index=0` tüm selectbox'larda
* Sliders (`exp_year`) başlangıç 0

## 3 Skor Formülü

Hâlâ V2–V3 ile aynı katsayı kümesi: 0.5 / 0.4 / 0.1.

## 4 Algoritmik Tutarlılık

Mask → Match Ratio → Skor sırası korunur. Yeni filtreler sadece **mask** aşamasına dahil edilir; `match_ratio` evrensel ortalama olmaya devam eder.

Bu, "katı" seçim (filter) vs. "yumuşak" skor (match+prob) ayrımını keskinleştirir.

## 5 Örnek Akış

1. Kullanıcı: *Şehir=İstanbul, Sektör=FinTech, MinDeneyim=3*
2. Mask boyutu 13 591 → 384 satıra düşer.
3. LightGBM + prob, `r`, match_ratio hesaplanır.
4. Top-10 sıralanır, UI'da gösterilir.

---
Bu sürüm, sistem kararlılığını artırmak üzere tasarlanmış; matematiksel yapı özünde V2–V3 ile eşittir. 