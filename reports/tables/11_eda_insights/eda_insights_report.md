# EDA Insights Report

## Purpose

This notebook consolidates the main insights from the exploratory analysis, feature engineering, outlier review, and feature selection stages.

## Main Insights

- The dataset is clean enough for modeling after duplicate removal, with no meaningful missing-value problem.
- Fraud is extremely rare, so threshold-aware and imbalance-aware evaluation is required.
- PCA-derived variables carry most of the strongest fraud signal.
- Outlier regions are often fraud-enriched and should generally be preserved.
- Feature engineering added useful interaction terms, but redundancy pruning was necessary.
- `log_amount` is the retained amount-family feature; raw `Amount` and amount ratios are dropped.
- The final selected feature set is suitable for both Logistic Regression and tree-based baseline models.

## Saved Tables

- `reports/tables/11_eda_insights/data_quality_insights_summary.csv`
- `reports/tables/11_eda_insights/signal_insights_summary.csv`
- `reports/tables/11_eda_insights/final_eda_takeaways.csv`
- `reports/tables/11_eda_insights/eda_insights_report.md`
