# Univariate Analysis Report

## Key Findings

- The target distribution remains highly imbalanced after cleaning.
- `Amount` remains strongly right-skewed, and the log-transformed version provides a more stable view for later modeling.
- `Time` spans the full observation window and may require normalization or feature engineering depending on the model choice.
- PCA-based variables vary in spread and outlier behavior, but their direct business interpretation remains limited.
- Scaling and transformation decisions should be considered before training linear models such as logistic regression.

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
