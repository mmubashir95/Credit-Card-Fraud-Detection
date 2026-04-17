# EDA Insights Report

## Purpose

This notebook consolidates the main insights from the exploratory analysis, feature engineering, outlier review, and feature selection stages.

## Main Insights

- The cleaned dataset is modeling-ready after duplicate removal and validation.
- Fraud is extremely rare, so imbalance-aware training and threshold-aware evaluation are required.
- Most predictive signal comes from a small set of PCA variables plus two retained interaction features.
- `log_amount` is the only retained amount-family feature.
- The selected feature set is ready for Logistic Regression, Random Forest, and downstream `BLOCK / REVIEW / APPROVE` decision design.

## Final Recommendation

- Top predictive features: `V14_V12_interaction`, `V14`, `V17_V16_interaction`, `V12`, `V17`, and `V10` should drive the baseline fraud model, with `V4`, `V16`, `V3`, `V11`, `V7`, `V18`, and `log_amount` as supporting inputs.
- Data challenges: severe class imbalance, skew in amount before transformation, and redundancy across engineered features.
- Modeling strategy: use the final selected features, scale where needed, start with class weights, compare Logistic Regression and Random Forest, and tune `BLOCK`, `REVIEW`, and `APPROVE` thresholds.
- Expected outcome: maximize fraud recall while maintaining enough precision for a practical review workflow.

## Saved Tables

- `reports/tables/11_eda_insights/data_quality_insights_summary.csv`
- `reports/tables/11_eda_insights/imbalance_strategy_summary.csv`
- `reports/tables/11_eda_insights/signal_insights_summary.csv`
- `reports/tables/11_eda_insights/feature_engineering_impact.csv`
- `reports/tables/11_eda_insights/final_dataset_summary.csv`
- `reports/tables/11_eda_insights/modeling_input_decision_summary.csv`
- `reports/tables/11_eda_insights/decision_system_feature_roles.csv`
- `reports/tables/11_eda_insights/decision_impact_framework.csv`
- `reports/tables/11_eda_insights/decision_threshold_framework.csv`
- `reports/tables/11_eda_insights/final_eda_takeaways.csv`
- `reports/tables/11_eda_insights/final_recommendation_summary.csv`
- `reports/tables/11_eda_insights/eda_insights_report.md`
