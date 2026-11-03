"""Streamlit UI.  Run: streamlit run app.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from config import *
from studentperf.validation import load_csv, validate
from studentperf.preparation import prepare
from studentperf import model as M

st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.title("AI-Powered Student Performance Prediction")
st.caption("Prototype on SYNTHETIC / anonymised data only. Predictions are decision support, not verdicts about any student.")

@st.cache_data(show_spinner="Preparing data...")
def _prep(df): return prepare(df)

@st.cache_resource(show_spinner="Training models...")
def _train(df): return M.train_and_evaluate(df)

# ---- 1. Input
st.sidebar.header("1. Data")
up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use bundled synthetic sample", value=up is None)
if up is not None:
    raw, errs = load_csv(up, up.name)
elif use_sample:
    raw, errs = load_csv("data/raw/students_synthetic.csv", "students_synthetic.csv")
else:
    st.info("Upload a CSV or tick the sample option."); st.stop()
if errs:
    for e in errs: st.error(e)
    st.stop()
errors, warnings = validate(raw)
for e in errors: st.error(e)
for w in warnings: st.warning(w)
if errors: st.stop()
st.sidebar.success(f"Source: {raw.attrs['source']} ({len(raw)} rows)")

clean, log = _prep(raw)
if len(clean) < 30 or clean[TARGET].nunique() < 2:
    st.error("Not enough usable rows (or only one outcome class) after cleaning to train a model."); st.stop()
res = _train(clean)
key = st.sidebar.selectbox("Model", ["logreg", "rf"], format_func=lambda k: {"logreg": "Logistic Regression (baseline)", "rf": "Random Forest (comparison)"}[k])
mdl = res["models"][key]

t1, t2, t3, t4, t5 = st.tabs(["Data quality", "Trends", "Evaluation", "Predict a student", "Cohort & export"])

with t1:
    st.subheader("Preparation log")
    for l in log: st.write("•", l)
    st.dataframe(raw.isna().sum().rename("missing values"))

with t2:
    st.subheader("Aggregate trends")
    c1, c2 = st.columns(2)
    f = c1.selectbox("Feature", FEATURES)
    fig, ax = plt.subplots(); clean.boxplot(column=f, by=TARGET, ax=ax); ax.set_xlabel("passed (0=no, 1=yes)"); plt.suptitle("")
    c1.pyplot(fig)
    fig, ax = plt.subplots(); clean[FEATURES + [TARGET_SCORE]].corr()[TARGET_SCORE].drop(TARGET_SCORE).sort_values().plot.barh(ax=ax)
    ax.set_title("Correlation with final_score"); c2.pyplot(fig)

with t3:
    st.subheader("Evaluation (held-out test set)")
    st.caption(f"Train/val/test sizes: {res['splits']}. Seed={RANDOM_SEED}. Metrics are from synthetic data and say nothing about real-world accuracy.")
    st.dataframe(pd.DataFrame({k: {"val " + a: b for a, b in v["val"].items()} | {"test " + a: b for a, b in v["test"].items()}
                               for k, v in res["models"].items()}).round(3))
    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(); ax.imshow(mdl["cm"], cmap="Blues")
    for (i, j), v in __import__("numpy").ndenumerate(mdl["cm"]): ax.text(j, i, v, ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test)")
    c1.pyplot(fig)
    imp = M.importances(mdl["model"], key)
    fig, ax = plt.subplots(); imp.sort_values().plot.barh(ax=ax)
    ax.set_title("Coefficient (standardised)" if key == "logreg" else "Feature importance"); c2.pyplot(fig)
    st.write("**Most confident errors on test set**")
    st.dataframe(M.error_analysis(mdl["model"], res["test_df"]))
    st.caption("Typical causes: students near the pass mark, unusual combinations (e.g. high attendance but low scores), and imputed missing values.")

with t4:
    st.subheader("Individual prediction (enter values manually)")
    vals = {}
    cols = st.columns(3)
    for i, (c, (lo, hi)) in enumerate(FEATURE_RANGES.items()):
        vals[c] = cols[i % 3].number_input(c, float(lo), float(hi), float((lo + hi) / 2), key=c)
    if st.button("Predict"):
        p = M.predict_one(mdl["model"], vals)
        st.metric("Estimated pass probability", f"{p:.0%}")
        (st.warning if p < AT_RISK_THRESHOLD else st.success)("Flagged for support follow-up" if p < AT_RISK_THRESHOLD else "Not flagged")

with t5:
    preds = M.predict_all(mdl["model"], clean)
    st.metric("Students flagged at-risk", int(preds.at_risk.sum()), f"of {len(preds)}")
    st.dataframe(preds.sort_values("pass_probability").head(50))
    st.download_button("Download predictions CSV", preds.to_csv(index=False), "predictions.csv", "text/csv")
    st.download_button("Download cleaned data CSV", clean.to_csv(index=False), "cleaned.csv", "text/csv")
