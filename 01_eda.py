# ============================================================
# 📊 EXPLORATORY DATA ANALYSIS (EDA) – INITIAL INSPECTION
# ============================================================

# Import required libraries
# pandas → data manipulation
# seaborn → statistical visualization
# matplotlib → plotting graphs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score, precision_score, recall_score
)


# ============================================================
# 1️⃣ LOAD DATASET
# ============================================================

# Load the dataset from CSV file
# Make sure the file path is correct relative to your project directory
df = pd.read_csv("data/creditcard.csv")


# ============================================================
# 2️⃣ BASIC STRUCTURE OVERVIEW
# ============================================================

# Display first 5 rows
# Helps understand column names and sample data
print("Head of the dataset:")
print(df.head())

# Check dataset shape
# shape → (number_of_rows, number_of_columns)
print("\nDataset Shape:")
print(df.shape)

# Display column names
# Helps understand feature list
print("\nDataset Columns:")
print(df.columns)


# ============================================================
# 3️⃣ DATA TYPES & NULL CHECK
# ============================================================

# df.info() shows:
# - Data types of each column
# - Non-null counts
# - Memory usage (summary level)
# Helps detect missing values and incorrect data types
print("\nDataset Info:")
print(df.info())


# ============================================================
# 4️⃣ MEMORY USAGE ANALYSIS
# ============================================================

# Check detailed memory usage of each column
# deep=True ensures object columns are fully calculated
# Important for large datasets
print("\nMemory Usage (in bytes):")
print(df.memory_usage(deep=True))


# ============================================================
# 📂 FEATURE TYPE IDENTIFICATION
# ============================================================

# ------------------------------------------------------------
# 1️⃣ Identify Numerical Columns
# ------------------------------------------------------------

# select_dtypes() filters columns based on datatype
# include=['int64','float64'] selects numeric columns
# .columns returns column names only
# These columns will be used for:
# - Scaling
# - Outlier detection
# - Correlation analysis
num_cols = df.select_dtypes(include=['int64','float64']).columns


# ------------------------------------------------------------
# 2️⃣ Identify Categorical Columns
# ------------------------------------------------------------

# include=['object'] selects string/text columns
# These columns typically require encoding
cat_cols = df.select_dtypes(include=['object']).columns


# ------------------------------------------------------------
# 3️⃣ Display Results
# ------------------------------------------------------------

print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical Columns:")
print(cat_cols)


# ============================================================
# 🔎 MISSING VALUES ANALYSIS
# ============================================================

print("\nMissing Values Analysis")
print("="*60)

# ------------------------------------------------------------
# 1️⃣ Count of Missing Values
# ------------------------------------------------------------

# df.isna() returns True for missing values
# .sum() counts how many missing values per column
print("\nMissing Values (Count):")
print(df.isna().sum())


# ------------------------------------------------------------
# 2️⃣ Missing Values Percentage (Manual Calculation)
# ------------------------------------------------------------

# Divide missing count by total rows
# Multiply by 100 to convert to percentage
print("\nMissing Values Percentage (Manual):")
print(df.isna().sum() / len(df) * 100)


# ------------------------------------------------------------
# 3️⃣ Missing Values Percentage (Using Mean)
# ------------------------------------------------------------

# df.isna() converts:
# True  → 1
# False → 0
# .mean() calculates proportion of missing values
# Multiply by 100 for percentage
print("\nMissing Values Percentage (Using Mean):")
print(df.isna().mean() * 100)

print("="*60)


# ============================================================
# 🎯 TARGET VARIABLE ANALYSIS
# ============================================================

# ------------------------------------------------------------
# 1️⃣ Target Distribution (Count)
# ------------------------------------------------------------

# value_counts() counts how many samples belong to each class
# Used to check class imbalance
print("\nTarget Variable Distribution:")
print(df['Class'].value_counts())


# ------------------------------------------------------------
# 2️⃣ Target Distribution (Percentage)
# ------------------------------------------------------------

# normalize=True converts counts to proportions
# Multiply by 100 for percentage format
print("\nTarget Variable Distribution (Percentage):")
print(df['Class'].value_counts(normalize=True) * 100)


# ============================================================
# 📊 NUMERICAL FEATURE SUMMARY
# ============================================================

# describe() gives:
# - count
# - mean
# - std
# - min
# - 25%, 50%, 75%
# - max
# Helps detect extreme values and scale differences
# df["Amount_log"] = np.log1p(df["Amount"])
# columns_to_describe = "Amount_log"
print(df[num_cols].describe())


# ============================================================
# 📈 VISUALIZATION & DISTRIBUTION ANALYSIS
# ============================================================

# Loop through each numerical column
for col in num_cols:

    # --------------------------------------------------------
    # Boxplot
    # --------------------------------------------------------
    # Used to detect outliers via IQR method
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

    # --------------------------------------------------------
    # Histogram + KDE
    # --------------------------------------------------------
    # Histogram shows distribution
    # KDE shows smooth probability density curve
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

    # --------------------------------------------------------
    # KDE Plot Using hue Parameter
    # --------------------------------------------------------
    # KDE shows smooth probability density curve
    # 'hue' automatically separates Normal (0) and Fraud (1)
    # Useful to visually compare overlap between classes
    plt.figure()
    sns.kdeplot(data=df, x=col, hue='Class')
    plt.title(f"{col} Distribution by Class")
    plt.show()

    # --------------------------------------------------------
    # Manual KDE Plot (Separate Class Filtering)
    # --------------------------------------------------------
    # Explicitly filter Normal transactions (Class = 0)
    # Explicitly filter Fraud transactions (Class = 1)
    # Gives more control if custom styling is needed
    plt.figure()
    sns.kdeplot(data=df[df['Class']==0], x=col, label='Normal')
    sns.kdeplot(data=df[df['Class']==1], x=col, label='Fraud')
    plt.legend()
    plt.title(f"{col} Distribution by Class")
    plt.show()

    # --------------------------------------------------------
    # Skewness
    # --------------------------------------------------------
    # Measures symmetry of distribution
    # Positive → Right skew
    # Negative → Left skew
    # Near 0 → Symmetric
    print(f"{col} Skew:", df[col].skew())

    # --------------------------------------------------------
    # Mean Comparison by Class
    # --------------------------------------------------------
    # Calculates average value of feature for each class
    # Helps identify direction of shift (which class has higher/lower mean)
    print(f"{col} Mean:", df.groupby("Class")[col].mean())

# ============================================================
# 🚨 OUTLIER DETECTION USING IQR METHOD
# ============================================================
# This section identifies potential outliers in all numerical columns
# using the Interquartile Range (IQR) method.
#
# IQR Method Logic:
# 1. Compute Q1 (25th percentile)
# 2. Compute Q3 (75th percentile)
# 3. Calculate IQR = Q3 - Q1
# 4. Define lower bound = Q1 - 1.5 * IQR
# 5. Define upper bound = Q3 + 1.5 * IQR
# 6. Any values outside these bounds are considered potential outliers
#
# NOTE:
# This does NOT remove outliers.
# It only measures their percentage for analysis purposes.
# ============================================================

for col in num_cols:

    # Calculate 1st Quartile (25th percentile)
    Q1 = df[col].quantile(0.25)

    # Calculate 3rd Quartile (75th percentile)
    Q3 = df[col].quantile(0.75)

    # Compute Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Define lower boundary for outlier detection
    # Any value below this will be considered an outlier
    lower = Q1 - 1.5 * IQR

    # Define upper boundary for outlier detection
    # Any value above this will be considered an outlier
    upper = Q3 + 1.5 * IQR

    # Identify rows where values fall outside the lower and upper bounds
    # These rows are potential outliers
    outliers = df[(df[col] < lower) | (df[col] > upper)]

    # Calculate and print percentage of outliers in this column
    # (number of outlier rows divided by total dataset size)
    print(col, "Outlier %:", len(outliers)/len(df)*100)

# ============================================================
# ✅ END OF OUTLIER DETECTION SECTION
# ============================================================


# ============================================================
# 📊 CORRELATION MATRIX ANALYSIS (Numerical vs Numerical)
# ============================================================
# This section computes and visualizes the correlation matrix
# for all numerical features in the dataset.
#
# Purpose:
# 1. Measure linear relationship between numerical variables
# 2. Detect strong positive or negative correlations
# 3. Identify potential multicollinearity issues
# 4. Check correlation strength with the target variable
#
# Correlation Value Range:
# +1   → Perfect positive linear relationship
#  0   → No linear relationship
# -1   → Perfect negative linear relationship
#
# What to Check:
# - Any correlation > 0.8 or < -0.8? (Possible multicollinearity)
# - Which features strongly correlate with target?
# - Are some features redundant?
#
# IMPORTANT:
# Correlation ≠ Causation
# A strong correlation does NOT mean one variable causes the other.
# ============================================================

corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Matrix", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ============================================================
# ✅ END OF CORRELATION ANALYSIS SECTION
# ============================================================

# --------------------------------------------------------
# TARGET COLUMN NAME
# --------------------------------------------------------
# Specify your target column (change if needed)
target_col = "Class"


# --------------------------------------------------------
# SELECT NUMERIC FEATURES (EXCLUDING TARGET)
# --------------------------------------------------------
# Select all numeric columns (int64, float64)
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Remove target column from numeric feature list
# We don't want to compare target with itself
num_cols = num_cols.drop(target_col)


# --------------------------------------------------------
# BOX PLOT: EACH NUMERIC FEATURE VS TARGET
# --------------------------------------------------------
# Purpose:
# - Compare distribution of each numeric feature
#   between target classes
# - Helps detect:
#   • Separation power
#   • Distribution shift
#   • Outliers per class
#   • Potentially strong predictors

# for col in num_cols:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(x=target_col, y=col, data=df)
#     plt.title(f"{col} vs {target_col}")
#     plt.tight_layout()
#     plt.show()


# --------------------------------------------------------
# CORRELATION WITH TARGET
# --------------------------------------------------------
# Compute correlation of all numeric features with target
# abs() is used because:
#   - We care about strength, not direction (+/-)
#   - Both strong positive and strong negative are useful

corr_with_target = df.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)


# --------------------------------------------------------
# SELECT TOP 5 MOST CORRELATED FEATURES
# --------------------------------------------------------
# index[0] is the target itself (correlation = 1)
# So we skip it using [1:6]
# This selects top 5 features most correlated with target

top_features = corr_with_target.index[1:6]


# --------------------------------------------------------
# BOX PLOT: TOP 5 FEATURES VS TARGET
# --------------------------------------------------------
# Purpose:
# - Focus only on strongest predictors
# - Clear visual comparison
# - Helps understand which features drive prediction most

for col in top_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target_col, y=col, data=df)
    plt.title(f"{col} vs {target_col}")
    plt.tight_layout()
    plt.show()


# for col in num_cols:
#     print("\n" + "="*60)
#     print(f"FRAUD DISTRIBUTION FOR: {col.upper()} (Quantile Binned)")
#     print("="*60)
    
#     # Create 10 quantile bins
#     df[f"{col}_bin"] = pd.qcut(df[col], q=10, duplicates="drop")
    
#     fraud_table = pd.crosstab(
#         df[f"{col}_bin"], 
#         df["Class"], 
#         normalize="index"
#     ) * 100
    
#     print(fraud_table.round(2))
#     print("="*60)


# --------------------------------------------------------
# 2) Split X / y
# --------------------------------------------------------
target_col = "Class"
X = df.drop(columns=[target_col])
y = df[target_col]

# Stratified split is IMPORTANT for imbalanced datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------------
# 3) Preprocessing
#    - Log transform Amount (because skew is huge)
#    - Scale all numeric features
# --------------------------------------------------------
# Amount column transformer: log1p then scale
amount_pipe = Pipeline([
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler())
])

# Other numeric columns (everything except Amount)
numeric_cols = X.columns.tolist()
other_numeric_cols = [c for c in numeric_cols if c != "Amount"]

preprocessor = ColumnTransformer(
    transformers=[
        ("amount", amount_pipe, ["Amount"]),
        ("num", StandardScaler(), other_numeric_cols),
    ],
    remainder="drop"
)

# --------------------------------------------------------
# 4) Model Pipeline
# --------------------------------------------------------
# class_weight='balanced' helps with imbalanced target
model = LogisticRegression(max_iter=2000, class_weight="balanced")

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# --------------------------------------------------------
# 5) Train
# --------------------------------------------------------
pipe.fit(X_train, y_train)

# --------------------------------------------------------
# 6) Predict + Evaluate
# --------------------------------------------------------
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print("\n==============================")
print("Classification Report")
print("==============================")
print(classification_report(y_test, y_pred))

print("\n==============================")
print("Confusion Matrix")
print("==============================")
print(confusion_matrix(y_test, y_pred))

print("\n==============================")
print("ROC-AUC")
print("==============================")
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# --------------------------------------------------------
# 7) Tune Decision Threshold (VERY IMPORTANT for Imbalanced Data)
# --------------------------------------------------------
# Default classification uses threshold = 0.5.
# For fraud detection (highly imbalanced), threshold tuning is critical:
# - Higher threshold  -> fewer fraud alerts -> Precision increases, Recall decreases
# - Lower threshold   -> more fraud alerts  -> Recall increases, Precision decreases
#
# We already computed:
#   y_prob = pipe.predict_proba(X_test)[:, 1]
# So here we test multiple thresholds and print the report for each.

thresholds = np.arange(0.10, 0.99, 0.05)

for t in thresholds:
    y_pred_custom = (y_prob > t).astype(int)
    print(f"\nThreshold: {t:.2f}")
    print(classification_report(y_test, y_pred_custom))
    # print("\n==============================")
    # print("Classification Report")
    # print("==============================")
    # print(classification_report(y_test, y_pred_custom))

    # print("\n==============================")
    # print("Confusion Matrix")
    # print("==============================")
    # print(confusion_matrix(y_test, y_pred_custom))

    # print("\n==============================")
    # print("ROC-AUC")
    # print("==============================")
    # print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# --------------------------------------------------------
# ✅ Auto-pick best threshold by maximizing F1 (fraud=1)
# --------------------------------------------------------
# y_prob must be predicted probabilities for class 1:
# y_prob = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# precision_recall_curve returns:
# precision: length n+1
# recall:    length n+1
# thresholds:length n
# So, align precision/recall with thresholds by removing last element
precision_t = precision[:-1]
recall_t = recall[:-1]

# Compute F1 for each threshold
f1_scores = (2 * precision_t * recall_t) / (precision_t + recall_t + 1e-12)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_precision = precision_t[best_idx]
best_recall = recall_t[best_idx]

print("\n" + "="*60)
print("✅ BEST THRESHOLD (MAX F1 FOR FRAUD CLASS=1)")
print("="*60)
print(f"Best Threshold : {best_threshold:.4f}")
print(f"Best Precision : {best_precision:.4f}")
print(f"Best Recall    : {best_recall:.4f}")
print(f"Best F1        : {best_f1:.4f}")
print("="*60)

# Apply best threshold and evaluate
y_pred_best = (y_prob >= best_threshold).astype(int)

print("\nClassification Report @ Best Threshold:")
print(classification_report(y_test, y_pred_best))

print("Confusion Matrix @ Best Threshold:")
print(confusion_matrix(y_test, y_pred_best))

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_best)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# plt.figure()
# sns.boxplot(x=df[columns_to_describe])
# plt.title(f"Boxplot of {columns_to_describe}")
# plt.show()

# plt.figure()
# sns.histplot(df[columns_to_describe], kde=True)
# plt.title(f"Distribution of {columns_to_describe}")
# plt.show()

# plt.figure()
# sns.kdeplot(data=df, x=columns_to_describe, hue='Class')
# plt.title(f"{columns_to_describe} Distribution by Class")
# plt.show()

# plt.figure()
# sns.kdeplot(data=df[df['Class']==0], x=columns_to_describe, label='Normal')
# sns.kdeplot(data=df[df['Class']==1], x=columns_to_describe, label='Fraud')
# plt.legend()
# plt.title(f"{columns_to_describe} Distribution by Class")
# plt.show()

# print(f"{columns_to_describe} Skew:", df[columns_to_describe].skew())
# print(f"{columns_to_describe} Mean:", df.groupby("Class")[columns_to_describe].mean())

# ------------------------------------------------------------
# 1) Helper: pick best threshold by brute-force (safe + correct)
# ------------------------------------------------------------
def best_threshold_by_f1(y_true, y_prob, n_steps=200):
    thresholds = np.linspace(0.0, 1.0, n_steps)

    best = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best["f1"]:
            best.update({"threshold": t, "precision": p, "recall": r, "f1": f1})

    return best


# ------------------------------------------------------------
# 2) Evaluate one model
# ------------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    # IMPORTANT: use probabilities for PR curve
    y_prob = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    best = best_threshold_by_f1(y_test, y_prob, n_steps=500)
    t = best["threshold"]
    y_pred_best = (y_prob >= t).astype(int)

    print("\n" + "="*70)
    print(f"MODEL: {name}")
    print("="*70)
    print(f"PR-AUC (Average Precision): {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print("-"*70)
    print("✅ BEST THRESHOLD (MAX F1 for fraud=1)")
    print(f"Threshold : {best['threshold']:.4f}")
    print(f"Precision : {best['precision']:.4f}")
    print(f"Recall    : {best['recall']:.4f}")
    print(f"F1        : {best['f1']:.4f}")
    print("-"*70)

    print("\nClassification Report @ Best Threshold:")
    print(classification_report(y_test, y_pred_best, digits=4))

    print("Confusion Matrix @ Best Threshold:")
    print(confusion_matrix(y_test, y_pred_best))

    return {
        "model": name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "best_threshold": best["threshold"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_f1": best["f1"],
    }

# ------------------------------------------------------------
# 3) Load your data (edit this part to match your file)
# ------------------------------------------------------------
# Example assumes you already have df with target column "Class"
# df = pd.read_csv("data/creditcard.csv")

# X = df.drop(columns=["Class"])
# y = df["Class"]

# Split
# Use stratify=y for fraud imbalance
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# ------------------------------------------------------------
# 4) Define models
# ------------------------------------------------------------
models = {
    # "LogReg (scaled, balanced)": Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=None))
    # ]),
     "LogReg (scaled, balanced)": Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ]),
    "RandomForest (balanced)": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

# ------------------------------------------------------------
# 5) Run comparison
# ------------------------------------------------------------
def run_all(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        results.append(evaluate_model(name, model, X_train, X_test, y_train, y_test))

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n" + "="*70)
    print("🏁 FINAL COMPARISON (sorted by PR-AUC)")
    print("="*70)
    print(results_df.to_string(index=False))
    return results_df

# After you load/split your data, run:
results_df = run_all(models, X_train, X_test, y_train, y_test)
print("\n" + "="*70)
print("🏁 FINAL COMPARISON (sorted by PR-AUC)")
print("="*70)
print(results_df.to_string(index=False))
print("\n" + "="*70)
print("✅ BEST MODEL:")
print(results_df.iloc[0])
print("="*70)   
print("\n" + "="*70)
print("✅ BEST THRESHOLD FOR BEST MODEL:")