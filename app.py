from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

st.set_page_config(page_title="Breast Cancer Model Evaluator", page_icon="🧬", layout="wide")
st.markdown("""
<style>
.main {background: linear-gradient(135deg,#f7fbff,#fff7fb)}
[data-testid='stMetric'] {background:#ffffff; border:1px solid #dbeafe; padding:12px; border-radius:14px}
.title-card {padding:18px;border-radius:16px;background:linear-gradient(90deg,#4f46e5,#0891b2);color:white;margin-bottom:16px}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
with open(BASE / "model" / "metadata.json", encoding="utf-8") as f:
    META = json.load(f)

# Cache trained models to avoid loading the model again after every interaction.
@st.cache_resource
def load_model(model_name):
    return joblib.load(BASE / "model" / META["model_files"][model_name])

st.markdown('<div class="title-card"><h1>🧬 Wisconsin Breast Cancer Classifier</h1><p>Evaluate five pre-trained classifiers on uploaded labelled test data without retraining.</p></div>', unsafe_allow_html=True)

st.caption("Machine Learning Assignment 2 | M.Tech AIML | Kumar Anish")

st.sidebar.header("Evaluation controls")
model_name = st.sidebar.selectbox("Select classification model", list(META["model_files"].keys()))

model_descriptions = {
    "Logistic Regression":
        "A linear classifier that estimates class probability using scaled input features.",

    "Decision Tree":
        "A rule-based classifier that divides the data using feature thresholds.",

    "KNN":
        "A distance-based classifier that predicts from the seven nearest training samples.",

    "Naive Bayes":
        "A probabilistic classifier based on Gaussian feature distributions.",

    "Support Vector Machine":
        "An RBF-kernel classifier that can identify nonlinear decision boundaries."
}

st.sidebar.info(model_descriptions[model_name])
uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"], help="CSV must contain all 30 features and the target column.")
st.sidebar.caption("Tip: use the included test_data.csv for evaluation.")

st.info("Target encoding: 0 = malignant, 1 = benign. Models were trained once using an 80/20 stratified split with random_state=42.")
if uploaded is None:
    st.subheader("Required CSV schema")
    st.write(f"Rows may vary, but the file must contain **{len(META['feature_names'])} feature columns** plus **target**.")
    st.dataframe(pd.DataFrame({"Required feature": META["feature_names"]}), use_container_width=True, hide_index=True)
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

# Verify that the uploaded CSV contains every feature used during training.
required = META["feature_names"] + [META["target_column"]]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()
if df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

X = df[META["feature_names"]]
y = df[META["target_column"]]
if X.isnull().any().any() or y.isnull().any():
    st.error("Missing values detected. Upload complete test data without blank values.")
    st.stop()
if not set(y.unique()).issubset({0, 1}):
    st.error("The target column must contain only 0 and 1.")
    st.stop()

# Load the selected saved model and generate predictions without retraining.
model = load_model(model_name)
pred = model.predict(X)
score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
metrics = {
    "Accuracy": accuracy_score(y, pred), "AUC": roc_auc_score(y, score),
    "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred, zero_division=0),
    "F1": f1_score(y, pred, zero_division=0), "MCC": matthews_corrcoef(y, pred)
}

st.subheader(f"Results: {model_name}")
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns([1, 1.35])
with left:
    st.markdown("#### Confusion matrix")
    # Compare actual and predicted classes for malignant and benign samples.
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label")
    st.pyplot(fig, clear_figure=True)
with right:
    st.markdown("#### Classification report")
    report = classification_report(y, pred, target_names=["malignant", "benign"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

with st.expander("Preview predictions"):
    out = df.copy(); out["predicted_target"] = pred; out["benign_probability"] = score
    st.dataframe(out.head(100), use_container_width=True)
    st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
