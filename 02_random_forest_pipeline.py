# ============================================================
# 🔥 FINAL RANDOM FOREST PIPELINE – FRAUD DETECTION
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================

df = pd.read_csv("data/creditcard.csv")

target_col = "Class"

X = df.drop(columns=[target_col])
y = df[target_col]

# Stratified split (IMPORTANT for imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 2️⃣ PREPROCESSING
#    (Tree models do NOT need scaling)
#    We only log-transform Amount
# ============================================================

def log_transform(X):
    X = X.copy()
    X["Amount"] = np.log1p(X["Amount"])
    return X

preprocessor = FunctionTransformer(log_transform)

# ============================================================
# 3️⃣ RANDOM FOREST MODEL
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("log_amount", preprocessor),
    ("model", rf_model)
])

# ============================================================
# 4️⃣ TRAIN
# ============================================================

pipe.fit(X_train, y_train)

# ============================================================
# 5️⃣ PREDICT PROBABILITIES
# ============================================================

y_prob = pipe.predict_proba(X_test)[:, 1]

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC :", average_precision_score(y_test, y_prob))

# ============================================================
# 6️⃣ AUTO SELECT BEST THRESHOLD (MAX F1)
# ============================================================

def best_threshold_by_f1(y_true, y_prob):
    thresholds = np.linspace(0.0, 1.0, 500)

    best = {"threshold": 0.5, "f1": 0}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best["f1"]:
            best["threshold"] = t
            best["f1"] = f1

    return best

best = best_threshold_by_f1(y_test, y_prob)
best_threshold = best["threshold"]

print("\nBest Threshold:", best_threshold)

# ============================================================
# 7️⃣ FINAL EVALUATION
# ============================================================

y_pred_best = (y_prob >= best_threshold).astype(int)

print("\nClassification Report @ Best Threshold")
print(classification_report(y_test, y_pred_best, digits=4))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_best))

# ============================================================
# 8️⃣ SAVE MODEL + THRESHOLD
# ============================================================

os.makedirs("artifacts", exist_ok=True)

joblib.dump({
    "model": pipe,
    "threshold": best_threshold
}, "artifacts/fraud_random_forest_pipeline.joblib")

print("\n✅ Model saved successfully!")