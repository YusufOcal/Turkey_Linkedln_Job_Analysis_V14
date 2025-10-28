# Job Recommender – Version 6 (Monte-Carlo Weight Optimiser)

## 1 Temel Fark
V6, V5'teki skor formülünü **değiştirmez**; yenilik otomatik ağırlık seçimi için Monte-Carlo (MC) simülasyonudur.

## 2 Monte-Carlo Süreci

### 2.1 Rastgele Senaryo Oluşturma
```python
for _ in range(1000):
    # Random filtre vektörü
    masks.append(random_filter())
```
Her senaryoda kullanıcı filtresi rastgele türetilir: şehir, seviye, min_exp vs. Seçim olasılıkları eşit.

### 2.2 Ağırlık Setlerinin Değerlendirilmesi

**Ölçüt (Reward)**

İlave kural  → "İlk 10'da başarı"

\[
R(\mathbf w) = \sum_{s=1}^{S} \sum_{i \in \text{Top10}(s,\mathbf w)} \mathbb{1}[\hat p_i>0.7] \cdot \mathbb{1}[m_i>0.6]
\]

Burada `Top10(s, w)` = senaryo `s` için seçilen ilk 10 ilan.

### 2.3 Seçim

Tüm şemalar için `R` hesaplanır; maksimum skoru alan şema `auto_scheme` olarak UI'da default index'e atanır.

## 3 Recency, Urgency, Clash

Aynı tanımlar V5'ten miras alınır. Urgency lineer, recency log, clash penalty uygulanır.

## 4 Performans

* 1000×3 ağırlık → 3000 skor seti
* Her set LightGBM tahmini çalıştırmaz; `prob` bir kez hesaplanıp DataFrame'e eklenir → **O(S × n)** sadece vektör aritmetiği.
* Toplam ek süre < 2 s (13.5k satır, 4-core CPU)

## 5 Matematiksel Tutarlılık

Simülasyon, kendi belirlenen "proxy-success" fonksiyonuna göre deterministik bir optimum bulur. Gerçek kullanıcı davranışını %100 yansıtmaz; fakat öncelikli amacın *yüksek olasılık + yüksek match* kombinasyonunu maksimize etmek olduğu için mantıksal çelişki yoktur.

## 6 Örnek

* S1 => 842 başarı | S2 => 867 | S3 => 810  ⇒ `auto_scheme = S2` (prob ağırlığı yüksek)

## 7 UI

Kullanıcı hâlâ şemaları manüel seçebilir; MC sonucu sadece başlangıç teklifidir.

---

Bu sürüm, seçim bilimi (bandit fikrine benzer) ekleyerek kullanıcı ayarı olmadan "iyi" ağırlık kombinasyonu sunar. Algoritmik pipeline V5 ile eş, skor fonksiyonu değişmez. 