# ============================================================
# ✅ FINAL LOGISTIC REGRESSION PIPELINE – FRAUD DETECTION
#    + Automatic threshold selection (max F1)
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================

df = pd.read_csv("data/creditcard.csv")

target_col = "Class"
X = df.drop(columns=[target_col])
y = df[target_col]

# Stratified split is IMPORTANT for fraud imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 2️⃣ PREPROCESSING
#    - Log transform Amount
#    - Scale all numeric features (LR needs scaling)
# ============================================================

# Amount: log1p then scale
amount_pipe = Pipeline([
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler())
])

# Other numeric columns (all except Amount)
all_cols = X.columns.tolist()
other_numeric_cols = [c for c in all_cols if c != "Amount"]

preprocessor = ColumnTransformer(
    transformers=[
        ("amount", amount_pipe, ["Amount"]),
        ("num", StandardScaler(), other_numeric_cols),
    ],
    remainder="drop"
)

# ============================================================
# 3️⃣ MODEL
# ============================================================

logreg = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    n_jobs=None
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", logreg)
])

# ============================================================
# 4️⃣ TRAIN
# ============================================================

pipe.fit(X_train, y_train)

# ============================================================
# 5️⃣ PROBABILITIES + BASE METRICS
# ============================================================

y_prob = pipe.predict_proba(X_test)[:, 1]

print("\n==============================")
print("ROC-AUC / PR-AUC")
print("==============================")
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC :", average_precision_score(y_test, y_prob))

# ============================================================
# 6️⃣ AUTO THRESHOLD SELECTION (MAX F1 using PR curve)
# ============================================================

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# precision & recall have length n+1, thresholds has length n
precision_t = precision[:-1]
recall_t = recall[:-1]

# Compute F1 for each threshold (your method)
f1_scores = (2 * precision_t * recall_t) / (precision_t + recall_t + 1e-12)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_precision = precision_t[best_idx]
best_recall = recall_t[best_idx]

print("\n" + "="*60)
print("✅ BEST THRESHOLD (MAX F1 FOR FRAUD CLASS=1)")
print("="*60)
print(f"Best Threshold : {best_threshold:.6f}")
print(f"Best Precision : {best_precision:.6f}")
print(f"Best Recall    : {best_recall:.6f}")
print(f"Best F1        : {best_f1:.6f}")
print("="*60)

# Apply best threshold
y_pred_best = (y_prob >= best_threshold).astype(int)

print("\nClassification Report @ Best Threshold")
print(classification_report(y_test, y_pred_best, digits=4))

print("\nConfusion Matrix @ Best Threshold")
print(confusion_matrix(y_test, y_pred_best))

# ============================================================
# 7️⃣ SAVE MODEL + THRESHOLD TOGETHER
# ============================================================

os.makedirs("artifacts", exist_ok=True)

joblib.dump(
    {"model": pipe, "threshold": float(best_threshold)},
    "artifacts/fraud_logreg_pipeline.joblib"
)

print("\n✅ Saved: artifacts/fraud_logreg_pipeline.joblib")