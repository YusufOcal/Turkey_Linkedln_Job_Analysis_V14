# v6: weight scheme auto-selected via Monte Carlo simulation
import io, math, random
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.calibration import CalibratedClassifierCV

DATA_PATH = "final_dataset_ml_ready_numeric_plus_extended_with_title.csv"
MODEL_PATH = "job_apply_lgbm_pipeline.pkl"
TOP_K = 10
SCHEMES = ["0.5/0.4/0.1", "0.6/0.3/0.1", "0.4/0.5/0.1"]
SCHEME_WEIGHTS = {"0.5/0.4/0.1":(0.5,0.4,0.1),"0.6/0.3/0.1":(0.6,0.3,0.1),"0.4/0.5/0.1":(0.4,0.5,0.1)}

@st.cache_resource
def load_calibrated():
    pipe = load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[c for c in ["title"] if c in df.columns])
    y = (df["apply_rate"] > df["apply_rate"].quantile(0.75)).astype(int)
    calib = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
    calib.fit(X, y)
    return calib

@st.cache_data
def load_df():
    df = pd.read_csv(DATA_PATH)
    rec = df["recency_score"].apply(lambda x: math.log1p(x))
    df["rec_log"] = (rec-rec.min())/(rec.max()-rec.min()+1e-6)
    return df

@st.cache_data
def simulate_best_scheme(n_samples:int=1000):
    df = load_df()
    model = load_calibrated()
    X = df.drop(columns=[c for c in ["title"] if c in df.columns])
    prob = model.predict_proba(X)[:,1]
    df["prob"] = prob
    # simple match proxy: random subset of criteria
    best_scheme = SCHEMES[0]; best_reward=-1
    for scheme in SCHEMES:
        w1,w2,w3 = SCHEME_WEIGHTS[scheme]
        reward=0
        for _ in range(n_samples):
            filt_prob = random.uniform(0,1)
            filt_match = random.uniform(0.5,1)
            filt_rec  = random.uniform(0,1)
            score = w1*df["prob"] + w2*filt_match + w3*filt_rec
            top = score.nlargest(10).index
            reward += (df.loc[top,"prob"]>0.7).sum()
        if reward>best_reward:
            best_reward=reward; best_scheme=scheme
    return best_scheme

best_scheme_default = simulate_best_scheme()

# ---------- UI ----------
st.set_page_config(page_title="İş Önerici v6", layout="wide")
st.title("🏢📍 İş İlanı Önerici – v6 (Auto Weight)")

df = load_df(); model = load_calibrated()

with st.sidebar:
    st.header("Filtreler")
    skills = st.multiselect("Beceri", sorted(set("|".join(df["skill_categories"].fillna("")).split("|"))))
    wp = st.selectbox("Çalışma Tipi", ["herhangi"]+sorted(df["jobWorkplaceTypes"].dropna().unique()))
    exp_year = st.slider("Min. Deneyim",0,30,0)
    level = st.selectbox("Seviye", ["herhangi"]+sorted(df["exp_level_final"].dropna().unique()))
    promo_only = st.toggle("Yalnız Promosyonlu")
    weight_scheme = st.radio("Skor Ağırlığı", SCHEMES, horizontal=True, index=SCHEMES.index(best_scheme_default))

# Prediction
X=df.drop(columns=[c for c in ["title"] if c in df.columns])
prob=model.predict_proba(X)[:,1]
res=df.copy(); res["prob"]=prob
# simple match ratio placeholder
res["match_ratio"]=1
w1,w2,w3=SCHEME_WEIGHTS[weight_scheme]
res["score"]=w1*res["prob"]+w2*res["match_ratio"]+w3*res["rec_log"]
res=res.sort_values("score",ascending=False).head(TOP_K)

for _,row in res.iterrows():
    st.subheader(row['title'])
    st.metric("Skor",f"{row['score']:.2f}") 