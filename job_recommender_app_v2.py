import re
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended_with_title.csv"
MODEL_PATH = "job_apply_lgbm_pipeline.pkl"
TOP_K = 10

# Load artefacts with caching
@st.cache_resource
def load_pipeline(path=MODEL_PATH):
    return load(path)

@st.cache_data
def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    # keep numeric recency min/max for later
    return df, df["recency_score"].min(), df["recency_score"].max()

# Helper to get options dynamically
def col_options(prefix, df):
    return sorted([c.replace(prefix, "") for c in df.columns if c.startswith(prefix)])

def main():
    st.set_page_config(page_title="Web Scraping ve Makine Öğrenmesi ile İş Piyasası Trendlerinin Analizi", layout="wide")
    st.title("📊 Web Scraping ve Makine Öğrenmesi ile İş Piyasası Trendlerinin Analizi")

    # Larger font inside dataframes
    st.markdown(
        """
        <style>
        /* Works for Streamlit >=1.25 AgGrid-based dataframe component */
        div[data-testid="stDataFrame"] tbody td {
            font-size: 16px !important;
            line-height: 24px !important;
        }
        div[data-testid="stDataFrame"] thead th {
            font-size: 16px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Model ve veri yükleniyor…"):
        df, rec_min, rec_max = load_dataset()
        pipe = load_pipeline()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Filtreler & Tercihler")
        selected_skills = st.multiselect("Teknik Yetenekler", options=sorted(set("|".join(df["skill_categories"].fillna("")).split("|"))))
        wp_choice = st.selectbox("Çalışma Tipi", options=["herhangi"] + sorted(df["jobWorkplaceTypes"].dropna().unique()))
        exp_year = st.slider("Min. Deneyim (yıl)", 0, 30, 0)
        emp_choice = st.selectbox("İstihdam Tipi", options=["herhangi"] + col_options("emp_", df))
        exp_level = st.selectbox("Seviye", options=["herhangi"] + sorted(df["exp_level_final"].dropna().unique()))
        size_choice = st.selectbox("Şirket Boyutu", ["herhangi"] + col_options("size_", df))
        ind_choice = st.selectbox("Sektör", ["herhangi"] + col_options("ind_", df))
        func_choice = st.selectbox("Fonksiyon", ["herhangi"] + col_options("func_", df))
        city_choice = st.selectbox("Şehir", ["herhangi"] + col_options("city_", df))
        urg_choice = st.selectbox("İlan Aciliyeti", ["herhangi"] + col_options("urg_", df))
        promo = st.selectbox("Promosyon", ["herhangi", "Evet", "Hayır"])
        min_match = st.slider("Minimum Eşleşme %", 0, 100, 0, step=5)
        btn = st.button("Sonuçları Göster", type="primary")

    if not btn:
        return

    # Prepare X for model prob (drop title)
    X = df.drop(columns=[c for c in ["title"] if c in df.columns])
    model_prob = pipe.predict_proba(X)[:, 1]

    result = df.copy()
    result["model_prob"] = model_prob

    # --- MATCH RATIO ---
    criteria_scores = []

    if selected_skills:
        sel_set = set(selected_skills)
        def skill_ratio(row):
            sk = set(str(row).split("|"))
            return len(sk & sel_set) / len(sel_set)
        result["m_skills"] = result["skill_categories"].apply(skill_ratio)
        criteria_scores.append("m_skills")
    else:
        result["m_skills"] = 1

    def add_binary(col_name, df_col, expected_val):
        result[col_name] = (df_col == expected_val).astype(int)
        criteria_scores.append(col_name)

    if wp_choice != "herhangi":
        add_binary("m_wp", result["jobWorkplaceTypes"].str.lower(), wp_choice.lower())
    else:
        result["m_wp"] = 1

    if exp_year > 0 and "exp_years_final" in result.columns:
        result["m_expyr"] = (result["exp_years_final"] >= exp_year).astype(int)
        criteria_scores.append("m_expyr")
    else:
        result["m_expyr"] = 1

    if emp_choice != "herhangi":
        add_binary("m_emp", result[f"emp_{emp_choice}"] if f"emp_{emp_choice}" in result.columns else 0, 1)
    else:
        result["m_emp"] = 1

    if exp_level != "herhangi":
        add_binary("m_explev", result["exp_level_final"], exp_level)
    else:
        result["m_explev"] = 1

    if size_choice != "herhangi":
        add_binary("m_size", result[f"size_{size_choice}"] if f"size_{size_choice}" in result.columns else 0, 1)
    else:
        result["m_size"] = 1

    if ind_choice != "herhangi":
        add_binary("m_ind", result[f"ind_{ind_choice}"] if f"ind_{ind_choice}" in result.columns else 0, 1)
    else:
        result["m_ind"] = 1

    if func_choice != "herhangi":
        add_binary("m_func", result[f"func_{func_choice}"] if f"func_{func_choice}" in result.columns else 0, 1)
    else:
        result["m_func"] = 1

    if city_choice != "herhangi":
        add_binary("m_city", result[f"city_{city_choice}"] if f"city_{city_choice}" in result.columns else 0, 1)
    else:
        result["m_city"] = 1

    if urg_choice != "herhangi":
        add_binary("m_urg", result[f"urg_{urg_choice}"] if f"urg_{urg_choice}" in result.columns else 0, 1)
    else:
        result["m_urg"] = 1

    if promo != "herhangi":
        pv = 1 if promo == "Evet" else 0
        result["m_promo"] = (result["promosyon_var"] == pv).astype(int)
        criteria_scores.append("m_promo")
    else:
        result["m_promo"] = 1

    # Overall match ratio (mean of criteria columns)
    result["match_ratio"] = result[criteria_scores].mean(axis=1)

    # Recency normalised 0-1
    rec_norm = (result["recency_score"] - rec_min) / (rec_max - rec_min + 1e-6)
    result["recency_norm"] = rec_norm.fillna(0)

    # Composite score
    result["score"] = 0.5 * result["model_prob"] + 0.4 * result["match_ratio"] + 0.1 * result["recency_norm"]

    # Filter by min match percentage
    result = result[result["match_ratio"] * 100 >= min_match]

    # Remove duplicates by title, keep highest score
    if "title" in result.columns:
        result = result.sort_values("score", ascending=False).drop_duplicates(subset=["title"], keep="first")

    # Sort again and pick top K
    top = result.sort_values("score", ascending=False).head(TOP_K)

    if top.empty:
        st.warning("Seçtiğiniz kriterlerle eşleşen ilan bulunamadı.")
        return

    display = top[["title", "score", "match_ratio", "model_prob", "recency_norm"]].rename(columns={
        "title": "İlan Başlığı",
        "score": "Toplam Skor",
        "match_ratio": "Eşleşme Oranı",
        "model_prob": "Model Olasılığı",
        "recency_norm": "İlan Güncellik",
    })
    display["Eşleşme Oranı"] = (display["Eşleşme Oranı"] * 100).round(1)
    st.dataframe(display.reset_index(drop=True))


if __name__ == "__main__":
    main() 