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
