import io, re
import pandas as pd
import numpy as np
import streamlit as st
from joblib import load

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended_with_title.csv"
MODEL_PATH = "job_apply_lgbm_pipeline.pkl"
TOP_K = 10

# ----------------- Helpers -----------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    return load(path)

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    rec_min, rec_max = df["recency_score"].min(), df["recency_score"].max()
    return df, rec_min, rec_max

# Initialize persistent state for filters
if "filt" not in st.session_state:
    st.session_state.filt = {}

def remember(key, default):
    return st.session_state.filt.get(key, default)

def safe_index(options, value):
    try:
        return options.index(value)
    except ValueError:
        return 0

# ----------------- Page & CSS -----------------
st.set_page_config(page_title="İş Piyasası Önerici v4", layout="wide")
st.title("📊 Web Scraping & ML ile İş İlanı Önerici – Gelişmiş UI")
st.markdown(
    """<style>
    div[data-testid="stDataFrame"] tbody td {font-size:15px;}
    div[data-testid="stDataFrame"] thead th {font-size:15px;}
    </style>""",
    unsafe_allow_html=True,
)

# Load artefacts
df, rec_min, rec_max = load_data()
pipe = load_model()

# ----------------- Sidebar Filters -----------------
with st.sidebar:
    st.header("Filtreler")
    skills_all = sorted(set("|".join(df["skill_categories"].fillna("")).split("|")))
    skills = st.multiselect("Teknik Yetenekler", skills_all, remember("skills", []))
    wp_opts = ["herhangi"] + sorted(df["jobWorkplaceTypes"].dropna().unique())
    wp = st.selectbox("Çalışma Tipi", wp_opts, index=safe_index(wp_opts, remember("wp", "herhangi")))
    exp_year = st.slider("Min. Deneyim (yıl)", 0, 30, remember("exp_year", 0))
    emp_opts = ["herhangi"] + [c.replace("emp_", "") for c in df.columns if c.startswith("emp_")]
    emp = st.selectbox("İstihdam", emp_opts, index=safe_index(emp_opts, remember("emp", "herhangi")))
    lvl_opts = ["herhangi"] + sorted(df["exp_level_final"].dropna().unique())
    exp_level = st.selectbox("Seviye", lvl_opts, index=safe_index(lvl_opts, remember("exp_level", "herhangi")))
    size_opts = ["herhangi"] + [c.replace("size_", "") for c in df.columns if c.startswith("size_")]
    size = st.selectbox("Şirket Boyutu", size_opts, index=safe_index(size_opts, remember("size", "herhangi")))
    ind_opts = ["herhangi"] + [c.replace("ind_", "") for c in df.columns if c.startswith("ind_")]
    ind = st.selectbox("Sektör", ind_opts, index=safe_index(ind_opts, remember("ind", "herhangi")))
    func_opts = ["herhangi"] + [c.replace("func_", "") for c in df.columns if c.startswith("func_")]
    func = st.selectbox("Fonksiyon", func_opts, index=safe_index(func_opts, remember("func", "herhangi")))
    city_opts = ["herhangi"] + [c.replace("city_", "") for c in df.columns if c.startswith("city_")]
    city = st.selectbox("Şehir", city_opts, index=safe_index(city_opts, remember("city", "herhangi")))
    urg_opts = ["herhangi"] + [c.replace("urg_", "") for c in df.columns if c.startswith("urg_")]
    urg = st.selectbox("Aciliyet", urg_opts, index=safe_index(urg_opts, remember("urg", "herhangi")))
    promo = st.selectbox("Promosyon", ["herhangi", "Evet", "Hayır"], index=["herhangi", "Evet", "Hayır"].index(remember("promo", "herhangi")))
    min_match = st.slider("Min. Eşleşme %", 0, 100, remember("min_match", 0), step=5)
    st.session_state.filt.update({"skills": skills, "wp": wp, "exp_year": exp_year, "emp": emp, "exp_level": exp_level, "size": size, "ind": ind, "func": func, "city": city, "urg": urg, "promo": promo, "min_match": min_match})

# ----------------- Model Prediction -----------------
X = df.drop(columns=[c for c in ["title"] if c in df.columns])
proba = pipe.predict_proba(X)[:, 1]
res = df.copy()
res["prob"] = proba

# --- Match ratio ---
match_series = []

# Skills ratio
sel_skills = set(st.session_state.filt["skills"])
if sel_skills:
    match_series.append(res["skill_categories"].apply(lambda x: len(set(str(x).split("|")) & sel_skills) / len(sel_skills)))

# Binary matches
binary_filters = {
    "wp": ("jobWorkplaceTypes", st.session_state.filt["wp"]),
    "emp": (f"emp_{st.session_state.filt['emp']}", 1),
    "exp_level": ("exp_level_final", st.session_state.filt["exp_level"]),
    "size": (f"size_{st.session_state.filt['size']}", 1),
    "ind": (f"ind_{st.session_state.filt['ind']}", 1),
    "func": (f"func_{st.session_state.filt['func']}", 1),
    "city": (f"city_{st.session_state.filt['city']}", 1),
    "urg": (f"urg_{st.session_state.filt['urg']}", 1),
    "promo": ("promosyon_var", 1 if st.session_state.filt["promo"] == "Evet" else 0 if st.session_state.filt["promo"] == "Hayır" else None),
}

for key, (col, val) in binary_filters.items():
    if val is None or val == "herhangi":
        continue
    if col not in res.columns:
        continue
    match_series.append((res[col] == val).astype(int))

# Experience year criterion
if st.session_state.filt["exp_year"] > 0:
    match_series.append((res["exp_years_final"] >= st.session_state.filt["exp_year"]).astype(int))

# If no criteria -> match 1
if not match_series:
    res["match_ratio"] = 1.0
else:
    res["match_ratio"] = pd.concat(match_series, axis=1).mean(axis=1)

# Score with recency bonus
res["rec_norm"] = (res["recency_score"] - rec_min) / (rec_max - rec_min + 1e-6)
res["score"] = 0.5*res["prob"] + 0.4*res["match_ratio"] + 0.1*res["rec_norm"]

# Duplicate drop & filter
res = res.sort_values("score", ascending=False).drop_duplicates("title")
res = res[res["match_ratio"]*100 >= min_match]

top = res.head(TOP_K)

# ----------------- TABS UI -----------------
results_tab, charts_tab = st.tabs(["🔎 Sonuçlar", "📈 Grafikler"])
with results_tab:
    if top.empty:
        st.warning("Eşleşen ilan bulunamadı.")
    else:
        for _, row in top.iterrows():
            col1, col2 = st.columns([3,1])
            with col1:
                st.subheader(row["title"])
                st.caption(f"{row['jobWorkplaceTypes']} | {row['skill_categories'][:100]}...")
            with col2:
                st.metric("Eşleşme %", f"{row['match_ratio']*100:.1f}%")
                st.progress(row['match_ratio'])
        # downloads
        csv = top.to_csv(index=False).encode("utf-8")
        st.download_button("CSV İndir", csv, file_name="recommended_jobs.csv")
        buf = io.BytesIO()
        top.to_excel(buf, index=False)
        st.download_button("Excel İndir", buf.getvalue(), file_name="recommended_jobs.xlsx")

with charts_tab:
    if not top.empty:
        st.bar_chart(top.set_index("title")["score"]) 