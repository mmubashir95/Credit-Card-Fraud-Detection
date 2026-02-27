# ============================================================
# 📊 EXPLORATORY DATA ANALYSIS (EDA) – INITIAL INSPECTION
# ============================================================

import pandas as pd


# ============================================================
# 1️⃣ LOAD DATASET
# ============================================================
# Make sure the file path is correct relative to your project
df = pd.read_csv("data/creditcard.csv")


# ============================================================
# 2️⃣ BASIC STRUCTURE OVERVIEW
# ============================================================

# Display first 5 rows of the dataset
# Helps to visually inspect column names and sample data
print("Head of the dataset:")
print(df.head())

# Check number of rows and columns
# shape → (rows, columns)
print("\nDataset Shape:")
print(df.shape)

# Display column names
# Helps understand feature list
print("\nDataset Columns:")
print(df.columns)


# ============================================================
# 3️⃣ DATA TYPES & NULL CHECK
# ============================================================

# df.info() gives:
# - Data types of each column
# - Non-null count
# - Memory usage (basic)
print("\nDataset Info:")
print(df.info())


# ============================================================
# 4️⃣ MEMORY USAGE ANALYSIS
# ============================================================

# Check detailed memory usage of each column
# deep=True ensures object columns are fully calculated
print("\nMemory Usage (in bytes):")
print(df.memory_usage(deep=True))


# ============================================================
# 📂 FEATURE TYPE IDENTIFICATION
# ============================================================

# ------------------------------------------------------------
# 1️⃣ Identify Numerical Columns
# ------------------------------------------------------------
# select_dtypes() filters columns based on data type
# include=['int64','float64'] → selects all numeric columns
# .columns → returns only the column names
# This helps us know which features need scaling, outlier detection, etc.
num_cols = df.select_dtypes(include=['int64','float64']).columns


# ------------------------------------------------------------
# 2️⃣ Identify Categorical Columns
# ------------------------------------------------------------
# include=['object'] → selects text/string columns
# These usually require encoding (OneHot, Label Encoding, etc.)
cat_cols = df.select_dtypes(include=['object']).columns


# ------------------------------------------------------------
# 3️⃣ Display Results
# ------------------------------------------------------------
# Printing numerical features
print("\nNumerical Columns:")
print(num_cols)

# Printing categorical features
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
# df.isna() → returns True where value is missing (NaN)
# .sum() → counts how many True values exist per column
# This gives total number of missing values in each column
print("\nMissing Values (Count):")
print(df.isna().sum())


# ------------------------------------------------------------
# 2️⃣ Missing Values Percentage (Manual Calculation)
# ------------------------------------------------------------
# df.isna().sum() → total missing per column
# len(df) → total number of rows
# Dividing gives proportion of missing values
# Multiply by 100 → convert to percentage
print("\nMissing Values Percentage (Manual):")
print(df.isna().sum() / len(df) * 100)


# ------------------------------------------------------------
# 3️⃣ Missing Values Percentage (Using Mean)
# ------------------------------------------------------------
# df.isna() converts:
#   True  → 1
#   False → 0
# .mean() calculates average of 1s per column
# That average equals missing percentage (in decimal form)
# Multiply by 100 → convert to percentage
# This method is cleaner and commonly used in EDA
print("\nMissing Values Percentage (Using Mean):")
print(df.isna().mean() * 100)

print("="*60)


print("\nTarget Variable Distribution:")
print(df['Class'].value_counts())
print("\nTarget Variable Distribution (Percentage):")
print(df['Class'].value_counts(normalize=True) * 100)