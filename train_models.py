from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# A fixed random state makes the train-test split reproducible.
# Running the script again will produce the same split and results.
RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Load the Breast Cancer Wisconsin Diagnostic dataset.
# The dataset contains 569 records, 30 numeric features, and a binary target.
data = load_breast_cancer(as_frame=True)

X = data.data.copy() # X contain Input measurements used by the classifiers
y = data.target.rename("target")

# Keep 20% of the dataset as unseen test data.
# Stratification maintains the malignant and benign class proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Define the five classification algorithms required by the assignment.
# Scaling is added for algorithms affected by feature magnitude or distance.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=RANDOM_STATE))
    ]),
}

file_names = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

# Train every model using the same training data and evaluate all models
# using the same test data to ensure a fair comparison.
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # AUC requires probability or decision scores instead of final class labels.
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    else:
        score = model.decision_function(X_test)
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, score),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    # Save the trained model so the Streamlit app can use it without retraining.
    joblib.dump(model, MODEL_DIR / file_names[name])

pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(BASE / "test_data.csv", index=False)
pd.DataFrame(results).to_csv(BASE / "model_metrics.csv", index=False)
metadata = {
    "feature_names": list(X.columns),
    "target_column": "target",
    "target_names": [str(x) for x in data.target_names],
    "model_files": file_names,
    "dataset_rows": int(X.shape[0]),
    "dataset_features": int(X.shape[1]),
    "test_rows": int(X_test.shape[0]),
    "random_state": RANDOM_STATE,
    "source": "UCI Machine Learning Repository; loaded through scikit-learn load_breast_cancer"
}
with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(pd.DataFrame(results).round(4).to_string(index=False))
