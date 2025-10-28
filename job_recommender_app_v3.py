import re
import io
import pandas as pd
import numpy as np
import streamlit as st
from joblib import load

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended_with_title.csv"
MODEL_PATH = "job_apply_lgbm_pipeline.pkl"
TOP_K = 10

@st.cache_resource
def load_model(path=MODEL_PATH):
    return load(path)

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    rec_min, rec_max = df["recency_score"].min(), df["recency_score"].max()
    return df, rec_min, rec_max

# --- Initialization of session state ---
if "filters" not in st.session_state:
    st.session_state.filters = {}

# --- PAGE CONFIG & CSS ---
st.set_page_config(page_title="İş Piyasası Trend Analizi", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] tbody td {font-size:15px;}
    div[data-testid="stDataFrame"] thead th {font-size:15px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Web Scraping & ML ile İş İlanı Önerici – Minimal UI")

# Load artefacts
df, rec_min, rec_max = load_data()
pipe = load_model()

# ---------- SIDEBAR FILTERS -----------------
with st.sidebar:
    st.header("Filtreler")
    def multi(key, options):
        return st.multiselect(key, options, st.session_state.filters.get(key, []))

    def select(key, options):
        return st.selectbox(key, options, index=options.index(st.session_state.filters.get(key, options[0])))

    skills = multi("Beceri", sorted(set("|".join(df["skill_categories"].fillna("")).split("|"))))
    wp = select("Çalışma Tipi", ["herhangi"] + sorted(df["jobWorkplaceTypes"].dropna().unique()))
    exp_year = st.slider("Min. Deneyim (yıl)", 0, 30, st.session_state.filters.get("exp_year", 0))
    emp = select("İstihdam", ["herhangi"] + [c.replace("emp_", "") for c in df.columns if c.startswith("emp_")])
    ind = select("Sektör", ["herhangi"] + [c.replace("ind_", "") for c in df.columns if c.startswith("ind_")])
    func = select("Fonksiyon", ["herhangi"] + [c.replace("func_", "") for c in df.columns if c.startswith("func_")])
    city = select("Şehir", ["herhangi"] + [c.replace("city_", "") for c in df.columns if c.startswith("city_")])
    min_match = st.slider("Min. Eşleşme %", 0, 100, st.session_state.filters.get("min_match", 0), step=5)

    # save selections
    st.session_state.filters.update({
        "Beceri": skills, "Çalışma Tipi": wp, "exp_year": exp_year,
        "İstihdam": emp, "Sektör": ind, "Fonksiyon": func, "Şehir": city,
        "min_match": min_match,
    })

# ------------- MODEL PROBABILITIES -------------
X = df.drop(columns=[c for c in ["title"] if c in df.columns])
prob = pipe.predict_proba(X)[:, 1]
res = df.copy()
res["prob"] = prob

# Match ratio calculation
def bin_match(col, val):
    if val == "herhangi":
        return pd.Series(1, index=res.index)
    if col.startswith("emp_") or col.startswith("ind_") or col.startswith("func_") or col.startswith("city_"):
        return res[col] if col in res.columns else 0
    return (res[col] == val).astype(int)

ratios = []
res["m_skill"] = 1
if skills:
    sel = set(skills)
    res["m_skill"] = res["skill_categories"].apply(lambda x: len(set(str(x).split("|")) & sel)/len(sel))
    ratios.append("m_skill")

res["m_wp"] = bin_match("jobWorkplaceTypes", wp)
ratios.append("m_wp")

if exp_year > 0:
    res["m_expyr"] = (res["exp_years_final"] >= exp_year).astype(int)
    ratios.append("m_expyr")
else:
    res["m_expyr"] = 1

res["m_emp"] = bin_match(f"emp_{emp}", emp)
res["m_ind"] = bin_match(f"ind_{ind}", ind)
res["m_func"] = bin_match(f"func_{func}", func)
res["m_city"] = bin_match(f"city_{city}", city)
ratios += ["m_emp", "m_ind", "m_func", "m_city"]

res["match_ratio"] = res[ratios].mean(axis=1)

# Composite score with recency bonus
res["rec_norm"] = (res["recency_score"] - rec_min) / (rec_max - rec_min + 1e-6)
res["score"] = 0.5*res["prob"] + 0.4*res["match_ratio"] + 0.1*res["rec_norm"]

# duplicate drop and filtering
res = res.sort_values("score", ascending=False).drop_duplicates("title")
res = res[res["match_ratio"]*100 >= min_match]

top = res.head(TOP_K)

# ---------------- TABS ----------------
results_tab, charts_tab = st.tabs(["🔎 Sonuçlar", "📈 Grafikler"])
with results_tab:
    if top.empty:
        st.warning("Seçilen filtrelerle eşleşen ilan bulunamadı.")
    else:
        for _, row in top.iterrows():
            with st.container():
                st.subheader(row["title"])
                st.metric("Eşleşme %", f"{row['match_ratio']*100:.1f}%")
                st.progress(row['match_ratio'])
                st.caption(f"Model Olasılığı: {row['prob']:.2f} | Güncellik: {row['rec_norm']:.2f}")
        # download buttons
        csv = top.to_csv(index=False).encode("utf-8")
        st.download_button("CSV İndir", csv, file_name="recommended_jobs.csv", mime="text/csv")
        xbuf = io.BytesIO()
        top.to_excel(xbuf, index=False)
        st.download_button("Excel İndir", xbuf.getvalue(), file_name="recommended_jobs.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with charts_tab:
    st.bar_chart(top.set_index("title")[["score"]]) 