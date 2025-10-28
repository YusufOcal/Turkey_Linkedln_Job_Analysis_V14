# Job Recommender – Version 7 (Exponential Recency, Log Urgency, Skill-Weighted Match)

## 1 Model Kalibrasyonu – Isotonic

`CalibratedClassifierCV(method='isotonic', cv='prefit')` seçilmiştir. Isotonic regresyon monotonluk kısıtlı parça-sabit fonksiyonla en küçük karesel hatayı minimize eder. Avantajı yüksek veri yoğunluğunda (13k satır) hem log-loss hem Brier skorunu Platt'a göre düşürmesidir.

\[
\hat p_i = \text{Iso}(f_{\text{LGBM}}(x_i))
\]

## 2 Recency Bonusu – Üssel Çürütme

\[
 r_i^{\exp} = e^{-\frac{\text{days}_i}{30}}
\]

* 0 gün → 1
* 30 gün → 0.367
* 90 gün → 0.049

Bu, 30 gün sonrası öneri değerini hızla azaltır.

## 3 Urgency Penalty – Logaritmik

\[
 u_i = 1 - \frac{\log(1+\text{level}_i)}{\log(1+5)}
\]

| `urg_level` | `u_i` |
|-------------|-------|
| 0 (NORMAL)  | 1.00 |
| 1           | 0.78 |
| 2           | 0.66 |
| 5 (EXPIRED) | 0.32 |

Log ölçeği başlangıçta hızlı, sonrasında yavaş düşer; EXPIRED sabit ≈0.32'ye sıkışır.

## 4 Match Ratio – Beceri Odaklı

### 4.1 Skill Jaccard
\[
 m_{\text{skill},i} = \frac{|S_u \cap S_i|}{|S_u|}
\]

### 4.2 Lokasyon Skoru (AND)
\[
 m_{\text{loc},i} = \mathbb{1}[wp\_match] \times \mathbb{1}[city\_match]
\]

### 4.3 Bileşik
\[
 m_i = 0.4\,m_{\text{skill},i} + 0.6\,m_{\text{loc},i}
\]

## 5 Ağırlık Şemaları & Otomatik Seçim

`SCHEMES = {S1:0.5/0.4/0.1, S2:0.6/0.3/0.1, S3:0.4/0.5/0.1}`  (prob/match/rec)

Monte-Carlo (500 senaryo) kullanılır; reward fonksiyonu:

\[
R=\sum_{s}\sum_{i\in Top10_s} \mathbb{1}[\hat p_i>0.7]\,\mathbb{1}[m_i>0.6]
\]

En yüksek R → `auto_scheme`. Kullanıcı yine değiştirebilir.

## 6 Nihai Skor

\[
 s_i = \bigl(w_1\,\hat p_i + w_2\,m_i + w_3\,r_i^{\exp}\bigr)\;u_i
\]

Varsayılan `TOP_K = 10`, ancak `head(min(TOP_K,len(sub)))` gösterilir.

## 7 Filtre -> Mask Akışı

* 12'den fazla selectbox/slider; her seçili kriter `mask &=` yaklaşımıyla AND'lenir. Boş küme durumunda `st.warning` + `st.stop()`.
* Mask sonrası tahmin yapılır → gereksiz satırlara CPU harcanmaz.

## 8 Sıralama & Görselleştirme

* İlanlar skor'a göre azalan sıralanır.
* Üç renkli metin (`green>70`, `orange>40`, else `red`).
* `st.progress(match_ratio)` – eşleşme oranını görsel bar ile sunar.
* CSV indirme butonu.

## 9 Matematiksel Tutarlılık Analizi

1. \(\hat p\) monoton kalibre edilmiştir; skor fonksiyonu her bileşen için **pozitif** ağırlık kullanır → Yüksek input değerleri skor'u artırır.
2. Urgency cezası \(u_i\in[0.32,1]\) çarpan olarak uygulanır → kötüleşme yalnızca aşağı yönlü, skor sıralamasını korur.
3. Recency üssel azalma 0'a asimptotik yaklaşır, eski ilanların listeye girmesi ancak prob+match yüksekse mümkündür.
4. Ağırlıkların toplamı 1 (prob+match+rec) olduğundan skor ölçeği \([0,1]\) içindedir.

## 10 Çıktı Örneği

Bir ilan için:
* \(\hat p = 0.83\)
* `days = 18` → \(r^{exp}=e^{-0.6}=0.55\)
* `m_skill=0.75`, `m_loc=1` → \(m=0.4·0.75+0.6·1 = 0.9\)
* Urgency NORMAL → \(u=1\)
* S2 seçili (0.6/0.3/0.1)

\[
 s = (0.6·0.83 + 0.3·0.9 + 0.1·0.55)·1 = 0.498 + 0.27 + 0.055 = 0.823
\]

İlan skor listesinde üst sıralarda yer alır.

---
V7, veri bilimi (isotonic, üssel decays) ve iş kuralı (log urgency, skill önceliği) bileşenlerini birleştirerek önceki sürümlerin en gelişmiş ve dengeli önericisidir. 