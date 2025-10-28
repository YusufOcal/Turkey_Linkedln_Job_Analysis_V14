import re
from typing import List

import pandas as pd
import streamlit as st
from joblib import load

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended_with_title.csv"
MODEL_PATH = "job_apply_lgbm_pipeline.pkl"
TOP_K = 10  # number of jobs to show


# -----------------------------------------------------------------------------
# Helpers for loading artefacts
# -----------------------------------------------------------------------------

@st.cache_resource
def load_pipeline(path: str = MODEL_PATH):
    """Load the pre-trained LightGBM pipeline (.pkl)."""
    return load(path)


@st.cache_data
def load_dataset(path: str = DATA_PATH):
    df = pd.read_csv(path)
    # Remove obvious leakage cols if still present
    for col in ["apply_rate", "pop_views_log", "pop_applies_log"]:
        if col in df.columns:
            df = df.drop(columns=col)
    return df


def extract_unique_skills(df: pd.DataFrame) -> List[str]:
    skills_set = set()
    for entry in df["skill_categories"].fillna(""):
        skills_set.update([s.strip() for s in str(entry).split("|") if s])
    return sorted(skills_set)


def column_options(prefix: str, df: pd.DataFrame) -> List[str]:
    """Return list of options by stripping prefix from columns."""
    return sorted([c.replace(prefix, "") for c in df.columns if c.startswith(prefix)])


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Job Recommender", layout="wide")
    st.title("📌 Akıllı İş İlanı Önerici")

    st.markdown("""
    Bu uygulama, LightGBM modeli yardımıyla profilinize en uygun iş ilanlarını puanlayıp sıralar.
    Seçenekleri doldurun ve **✨ İlanları Listele** butonuna basın.
    """)

    # Load artefacts
    with st.spinner("Veri ve model yükleniyor…"):
        df = load_dataset()
        pipe = load_pipeline()
    
    # Sidebar inputs
    with st.sidebar:
        st.header("🛠️ Filtreler")

        # Technical skills multiselect
        all_skills = extract_unique_skills(df)
        selected_skills = st.multiselect("Teknik yetenekler (skill_categories)", options=all_skills)

        # Workplace type
        workplace_opts = ["hepsi"] + sorted(df["jobWorkplaceTypes"].dropna().unique())
        workplace_choice = st.selectbox("Çalışma şekli (remote / on-site / hybrid)", options=workplace_opts)

        # Experience years
        exp_year = st.number_input("Deneyim yılı", min_value=0, max_value=40, value=0, step=1)

        # Employment status (one-hot columns emp_*)
        emp_options = ["herhangi"] + column_options("emp_", df)
        emp_choice = st.selectbox("Employment status", options=emp_options)

        # Experience level (categorical)
        exp_level_opts = ["herhangi"] + sorted(df["exp_level_final"].dropna().unique())
        exp_level_choice = st.selectbox("Experience level", options=exp_level_opts)

        # Promotion flag
        promo_choice = st.selectbox("Promosyon var mı?", options=["herhangi", "Evet", "Hayır"])

        # Company size (size_*)
        size_options = ["herhangi"] + column_options("size_", df)
        size_choice = st.selectbox("Şirket büyüklüğü", options=size_options)

        # Industry (ind_*)
        ind_options = ["herhangi"] + column_options("ind_", df)
        ind_choice = st.selectbox("Şirket sektörü", options=ind_options)

        # Job function (func_*)
        func_options = ["herhangi"] + column_options("func_", df)
        func_choice = st.selectbox("Job function", options=func_options)

        # Country / City
        city_options = ["herhangi"] + column_options("city_", df)
        city_choice = st.selectbox("Şehir", options=city_options)

        # Job urgency (urg_*)
        urg_options = ["herhangi"] + column_options("urg_", df)
        urg_choice = st.selectbox("İş aciliyeti", options=urg_options)

        submit = st.button("En iyi ilanları listele")

    if submit:
        st.subheader("🔎 Sonuçlar")
        filt_df = df.copy()

        # Work on a copy for scoring
        work_df = df.copy()

        # Track criteria for match score
        criteria_checks = []  # list of lambdas row->score 0/1 or ratio

        # Apply filters
        if workplace_choice != "hepsi":
            criteria_checks.append(lambda r, v=workplace_choice: int(r["jobWorkplaceTypes"] == v))
            # no filtering; scoring handles relevance

        if selected_skills:
            pattern = "|".join([re.escape(s) for s in selected_skills])
            def skill_ratio(row, patt=pattern):
                sks=set(str(row).split("|"))
                sel=set(selected_skills)
                return len(sks.intersection(sel))/len(sel)
            criteria_checks.append(skill_ratio)
            # no filtering

        if exp_year > 0 and "exp_years_final" in work_df.columns:
            criteria_checks.append(lambda r, v=exp_year: int(r["exp_years_final"]>=v))
            # no filtering

        if emp_choice != "herhangi":
            col=f"emp_{emp_choice}"
            if col in work_df.columns:
                criteria_checks.append(lambda r, c=col: int(r[c]==1))
                # no filtering

        if exp_level_choice != "herhangi":
            criteria_checks.append(lambda r, v=exp_level_choice: int(r["exp_level_final"]==v))
            # no filtering

        if promo_choice != "herhangi":
            pv = 1 if promo_choice=="Evet" else 0
            criteria_checks.append(lambda r, pv=pv: int(r["promosyon_var"]==pv))
            # no filtering

        if size_choice != "herhangi":
            sc=f"size_{size_choice}"
            if sc in work_df.columns:
                criteria_checks.append(lambda r, c=sc: int(r[c]==1))
                # no filtering

        if ind_choice != "herhangi":
            ic=f"ind_{ind_choice}"
            if ic in work_df.columns:
                criteria_checks.append(lambda r, c=ic: int(r[c]==1))
                # no filtering

        if func_choice != "herhangi":
            fc=f"func_{func_choice}"
            if fc in work_df.columns:
                criteria_checks.append(lambda r, c=fc: int(r[c]==1))
                # no filtering

        if city_choice != "herhangi":
            cc=f"city_{city_choice}"
            if cc in work_df.columns:
                criteria_checks.append(lambda r, c=cc: int(r[c]==1))
                # no filtering

        if urg_choice != "herhangi":
            uc=f"urg_{urg_choice}"
            if uc in work_df.columns:
                criteria_checks.append(lambda r, c=uc: int(r[c]==1))
                # no filtering

        if not criteria_checks:
            st.warning("Lütfen en az bir kriter seçin.")
            return

        # Prepare prediction (model must see same feature set as training – drop non-model columns)
        X_pred = work_df.drop(columns=[c for c in ["title"] if c in work_df.columns]).copy()
        proba = pipe.predict_proba(X_pred)[:, 1]

        # Compute match ratio for every row
        def calc_match(row):
            scores=[f(row) for f in criteria_checks]
            return sum(scores)/len(scores)

        match_ratio = work_df.apply(calc_match, axis=1)

        # Composite score
        composite = 0.6 * proba + 0.4 * match_ratio

        result_df = df.assign(apply_probability=proba, match_ratio=match_ratio, score=composite)

        # Remove rows with zero match to avoid irrelevant jobs
        result_df = result_df[result_df["match_ratio"] > 0]

        if result_df.empty:
            st.warning("Seçtiğiniz kriterlerle eşleşen ilan bulunamadı.")
            return

        # Ensure uniqueness by a proxy column list
        result_df = result_df.drop_duplicates(subset=list(df.columns))

        top_df = result_df.sort_values("score", ascending=False).head(TOP_K)

        st.caption(f"Top {TOP_K} ilan gösteriliyor (toplam {len(result_df)} uygun ilan arasından)")
        display_cols=["score","apply_probability","match_ratio"]
        # attempt to add a title-like column if exists
        for cand in ["title","job_title","posting_title","standardizedTitle"]:
            if cand in top_df.columns:
                display_cols.append(cand)
                break
        display_cols+= ["jobWorkplaceTypes","skill_categories","exp_level_final"]
        top_show = top_df[display_cols].round(3)
        st.dataframe(top_show)

        st.download_button(
            "Sonuçları CSV olarak indir",
            data=top_df.to_csv(index=False).encode("utf-8"),
            file_name="recommended_jobs.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main() 