import io, math
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.calibration import CalibratedClassifierCV

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended_with_title.csv"
MODEL_PATH = "job_apply_lgbm_pipeline.pkl"
TOP_K = 10

# ------------------ Caching ------------------
@st.cache_resource
def load_models(path=MODEL_PATH):
    pipe = load(path)
    # probability calibration (Platt scaling) – fit once and cache
    calib = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[c for c in ["title"] if c in df.columns])
    y = (df["apply_rate"] > df["apply_rate"].quantile(0.75)).astype(int)
    calib.fit(X, y)
    return pipe, calib

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    rec = df["recency_score"].apply(lambda x: math.log1p(x))
    df["rec_log"] = (rec - rec.min())/(rec.max()-rec.min()+1e-6)
    # urgency level mapping
    urg_map = {"NORMAL":0, "LOW":1, "MEDIUM":2, "HIGH":3, "CRITICAL":4, "EXPIRED":5}
    for key,val in urg_map.items():
        col = f"urg_{key}"
        if col in df.columns:
            df.loc[df[col]==1, "urg_level"] = val
    df["urg_level"].fillna(0, inplace=True)
    return df

# Page config
st.set_page_config(page_title="İş Önerici v5", layout="wide")
st.title("🏢📍 İş İlanı Önerici – v5")

pipe, calib_pipe = load_models()
df = load_data()

# ------------ Sidebar Filters ------------
with st.sidebar:
    st.header("Filtreler")
    skills = st.multiselect("Beceri", sorted(set("|".join(df["skill_categories"].fillna("")).split("|"))))
    wp = st.selectbox("Çalışma Tipi", ["herhangi"]+sorted(df["jobWorkplaceTypes"].dropna().unique()))
    exp_year = st.slider("Min. Deneyim",0,30,0)
    level = st.selectbox("Seviye", ["herhangi"]+sorted(df["exp_level_final"].dropna().unique()))
    # Ek filtreler (emp, size, ind, func, city, urg)
    def safe_idx(opts, val):
        try:
            return opts.index(val)
        except ValueError:
            return 0

    emp_opts = ["herhangi"] + [c.replace("emp_", "") for c in df.columns if c.startswith("emp_")]
    emp = st.selectbox("İstihdam", emp_opts, index=safe_idx(emp_opts, "herhangi"))

    size_opts = ["herhangi"] + [c.replace("size_", "") for c in df.columns if c.startswith("size_")]
    size = st.selectbox("Şirket Boyutu", size_opts, index=safe_idx(size_opts, "herhangi"))

    ind_opts = ["herhangi"] + [c.replace("ind_", "") for c in df.columns if c.startswith("ind_")]
    ind = st.selectbox("Sektör", ind_opts, index=safe_idx(ind_opts, "herhangi"))

    func_opts = ["herhangi"] + [c.replace("func_", "") for c in df.columns if c.startswith("func_")]
    func = st.selectbox("Fonksiyon", func_opts, index=safe_idx(func_opts, "herhangi"))

    city_opts = ["herhangi"] + [c.replace("city_", "") for c in df.columns if c.startswith("city_")]
    city = st.selectbox("Şehir", city_opts, index=safe_idx(city_opts, "herhangi"))

    urg_opts = ["herhangi"] + [c.replace("urg_", "") for c in df.columns if c.startswith("urg_")]
    urg = st.selectbox("Aciliyet", urg_opts, index=safe_idx(urg_opts, "herhangi"))

    promo_only = st.toggle("Yalnız Promosyonlu Göster")
    scheme_opts = ["0.5/0.4/0.1", "0.6/0.3/0.1", "0.4/0.5/0.1"]
    weight_scheme = st.radio("Skor Ağırlığı", scheme_opts, index=0, horizontal=True)

# probability prediction
X = df.drop(columns=[c for c in ["title"] if c in df.columns])
prob = calib_pipe.predict_proba(X)[:,1]
res = df.copy()
res["prob"] = prob

# Match ratio
match_cols=[]
if skills:
    sel=set(skills)
    res["m_sk"] = res["skill_categories"].apply(lambda x: len(set(str(x).split("|"))&sel)/len(sel))
    match_cols.append("m_sk")
else:
    res["m_sk"]=1
if wp!="herhangi":
    res["m_wp"]=(res["jobWorkplaceTypes"]==wp).astype(int)
    match_cols.append("m_wp")
else:
    res["m_wp"]=1
if exp_year>0:
    res["m_expyr"]=(res["exp_years_final"]>=exp_year).astype(int)
    match_cols.append("m_expyr")
else:
    res["m_expyr"]=1
if level!="herhangi":
    res["m_lvl"]=(res["exp_level_final"]==level).astype(int)
    match_cols.append("m_lvl")
else:
    res["m_lvl"]=1

# Additional binary criteria
if emp!="herhangi":
    res["m_emp"] = res.get(f"emp_{emp}",0)
    match_cols.append("m_emp")
else:
    res["m_emp"] = 1

if size!="herhangi":
    res["m_size"] = res.get(f"size_{size}",0)
    match_cols.append("m_size")
else:
    res["m_size"] = 1

if ind!="herhangi":
    res["m_ind"] = res.get(f"ind_{ind}",0)
    match_cols.append("m_ind")
else:
    res["m_ind"] = 1

if func!="herhangi":
    res["m_func"] = res.get(f"func_{func}",0)
    match_cols.append("m_func")
else:
    res["m_func"] = 1

if city!="herhangi":
    res["m_city"] = res.get(f"city_{city}",0)
    match_cols.append("m_city")
else:
    res["m_city"] = 1

if urg!="herhangi":
    res["m_urg"] = res.get(f"urg_{urg}",0)
    match_cols.append("m_urg")
else:
    res["m_urg"] = 1

# recompute match_ratio
res["match_ratio"]=res[match_cols].mean(axis=1)

# fill nan
res["match_ratio"].fillna(0, inplace=True)

# Experience-level clash penalty
clash = ((res["exp_years_final"]<=2)&(res["exp_level_final"].str.contains("senior",case=False)))|((res["exp_years_final"]>10)&(res["exp_level_final"].str.contains("entry",case=False)))
res.loc[clash,"match_ratio"]*=0.1
# urgency penalty
res["urg_pen"] = 1 - 0.05*res["urg_level"]

# choose weights
wt_map={"0.5/0.4/0.1":(0.5,0.4,0.1),"0.6/0.3/0.1":(0.6,0.3,0.1),"0.4/0.5/0.1":(0.4,0.5,0.1)}
wp1,wp2,wp3 = wt_map[weight_scheme]
res["score"]= (wp1*res["prob"] + wp2*res["match_ratio"] + wp3*res["rec_log"]) * res["urg_pen"]

if promo_only:
    res=res[res["promosyon_var"]==1]

# sort + unique title
res=res.sort_values("score",ascending=False).drop_duplicates("title").head(TOP_K)

# Display
for _,row in res.iterrows():
    st.subheader(row["title"])
    st.write(f"🏢 {row['jobWorkplaceTypes']}  |  📍 {row['skill_categories'][:80]}...")
    cols=st.columns(3)
    cols[0].metric("Skor",f"{row['score']:.2f}")
    cols[1].metric("Eşleşme %",f"{row['match_ratio']*100:.1f}%")
    cols[2].metric("Olasılık",f"{row['prob']:.2f}")
    st.progress(float(row['match_ratio']))

csv=res.to_csv(index=False).encode('utf-8')
st.download_button("CSV",csv,"jobs_v5.csv")
buf=io.BytesIO(); res.to_excel(buf,index=False)
st.download_button("Excel",buf.getvalue(),"jobs_v5.xlsx") 