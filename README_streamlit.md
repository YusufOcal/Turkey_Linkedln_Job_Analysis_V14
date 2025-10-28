# Job Posting Recommender – Streamlit Deployment

## 1. Ortamı Kurma

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. LightGBM Modelini Eğit & Kaydet

```bash
# Varsayılan dosya yolları: dataset ve çıktı .pkl
python train_final_model.py
```

`job_apply_lgbm_pipeline.pkl` oluşacaktır.

> Zaten aynı dizinde mevcutsa bu adımı atlayabilirsiniz.

## 3. Streamlit Uygulamasını Başlat

```bash
streamlit run job_recommender_app.py
```

Tarayıcıda `localhost:8501` adresinde arayüz açılır.  
Yan panelden profil bilgilerinizi doldurup **En iyi ilanları listele** butonuna basın;   
model, filtrelerinize uyan ilanları puanlayıp en yüksek olasılıklı ilk 20 taneyi gösterir.

---

### Sütun / Girdi Eşlemesi

| Arayüz Alanı | Dataset Sütunu / One-Hot Grubu | Açıklama |
|--------------|--------------------------------|----------|
| Teknik yetenekler | `skill_categories` | Çoklu seçim.<br>`|` ayrılmış metin dizeleri içinde geçiyor. |
| Çalışma şekli | `jobWorkplaceTypes` | `remote`, `on-site`, `hybrid` … |
| Deneyim yılı | `exp_years_final` | Sayısal yıl bilgisi. |
| Employment status | `emp_*` sütunları | `emp_full-time`, `emp_part-time`, `emp_internship` … |
| Experience level | `exp_level_final` | `entry-level`, `mid-level`, `senior` vb. |
| Promosyon | `promosyon_var` | 1: Var, 0: Yok |
| Şirket büyüklüğü | `size_*` sütunları | `size_startup`, `size_medium`, `size_large` vb. |
| Sektör | `ind_*` sütunları | `ind_technology`, `ind_marketing`, `ind_manufacturing` … |
| Job function | `func_*` sütunları | `func_engineering`, `func_sales` vb. |
| Şehir | `city_*` sütunları | `city_istanbul` … |
| İş aciliyeti | `urg_*` sütunları | `urg_HIGH`, `urg_LOW` vb. | 