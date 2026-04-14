# Univariate Analysis Report

## Key Findings

- The target distribution remains highly imbalanced after cleaning.
- `Amount` remains strongly right-skewed, and the log-transformed version provides a more stable view for later modeling.
- `Time` spans the full observation window and may require normalization or feature engineering depending on the model choice.
- PCA-based variables vary in spread and outlier behavior, but their direct business interpretation remains limited.
- Scaling and transformation decisions should be considered before training linear models such as logistic regression.

## Class Imbalance Impact

- The dataset is highly imbalanced, with approximately 99.83% non-fraud transactions and 0.17% fraud transactions.
- Accuracy is misleading in this setting because a model can appear strong by predicting the majority class only.
- Evaluation should focus on precision, recall, F1-score, PR-AUC, and ROC-AUC, with particular attention to recall and PR-AUC for the fraud class.
- Imbalance handling should rely on stratified splitting, class weighting, threshold tuning, and controlled resampling if required.

## Impact on Modeling

- `Amount` should be log-transformed and scaled because its strong right skew and extreme tail can distort model learning.
- `Time` should be scaled, and time-based feature engineering can be evaluated later if it improves fraud separation.
- `V1` to `V28` should remain unchanged because they are already PCA-transformed, centered, scaled, and decorrelated.
- Outliers should be retained and handled through robust preprocessing rather than removed blindly, because extreme observations may contain important fraud signal.
- Linear models such as Logistic Regression benefit from these preprocessing steps, while tree-based models are naturally more tolerant of outliers and non-linear feature behavior.

## Key Insights

- The fraud class remains extremely rare, so the dataset is not suitable for accuracy-based evaluation and requires imbalance-aware modeling and validation.
- `Amount` is strongly right-skewed and needs transformation, while `Time` shows broader spread across the observation window and may benefit from scaling or later feature engineering.
- Several PCA features show elevated kurtosis, which indicates rare but potentially important extreme transaction patterns that may strengthen anomaly detection.
- A high outlier count is expected in fraud detection and should be treated as potential signal rather than automatic noise.
- `V1` to `V28` should remain unchanged because they are already PCA-transformed, centered, scaled, and decorrelated.
- Apply `log1p` and scaling to `Amount`, scale `Time`, preserve PCA features as-is, and choose evaluation and modeling strategies that are robust to imbalance and rare-event behavior.

## Plot Interpretation Guide

- Explain what the feature distribution reveals about typical transaction behavior rather than only describing the plot shape.
- Identify whether the feature is symmetric, right-skewed, left-skewed, or contains unusual tail behavior, and explain why that matters.
- Describe whether rare values, heavy tails, or unusual spread may help separate fraudulent transactions from normal ones.
- Conclude with a practical preprocessing action such as keep as-is, scale, apply `log1p`, monitor outliers, or engineer a better representation.

## Saved Tables

- `reports/tables/05_univariate_analysis/target_distribution_after_cleaning.csv`
- `reports/tables/05_univariate_analysis/summary_statistics.csv`
- `reports/tables/05_univariate_analysis/skewness_summary.csv`
- `reports/tables/05_univariate_analysis/amount_summary.csv`
- `reports/tables/05_univariate_analysis/time_summary.csv`
- `reports/tables/05_univariate_analysis/pca_feature_summary.csv`
- `reports/tables/05_univariate_analysis/outlier_summary.csv`
- `reports/tables/05_univariate_analysis/feature_interpretations.md`
- `reports/tables/05_univariate_analysis/univariate_report.md`

## Saved Figures

- `reports/figures/05_univariate_analysis/target_distribution_after_cleaning.png`
- `reports/figures/05_univariate_analysis/amount_distribution.png`
- `reports/figures/05_univariate_analysis/amount_boxplot.png`
- `reports/figures/05_univariate_analysis/amount_log_distribution.png`
- `reports/figures/05_univariate_analysis/time_distribution.png`
- `reports/figures/05_univariate_analysis/time_boxplot.png`
- `reports/figures/05_univariate_analysis/v1_distribution.png`
- `reports/figures/05_univariate_analysis/v1_boxplot.png`
- `reports/figures/05_univariate_analysis/v2_distribution.png`
- `reports/figures/05_univariate_analysis/v2_boxplot.png`
- `reports/figures/05_univariate_analysis/v3_distribution.png`
- `reports/figures/05_univariate_analysis/v3_boxplot.png`
- `reports/figures/05_univariate_analysis/v4_distribution.png`
- `reports/figures/05_univariate_analysis/v4_boxplot.png`
- `reports/figures/05_univariate_analysis/v5_distribution.png`
- `reports/figures/05_univariate_analysis/v5_boxplot.png`
- `reports/figures/05_univariate_analysis/v6_distribution.png`
- `reports/figures/05_univariate_analysis/v6_boxplot.png`
