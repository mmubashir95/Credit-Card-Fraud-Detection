# Final Summary Report

## Project Overview

- Problem: credit card fraud detection.
- Goal: classify transactions and support `BLOCK`, `REVIEW`, and `APPROVE` decisions.
- Current status: data preparation, EDA synthesis, feature engineering review, feature selection, and decision design are complete.

## Final Dataset

- Cleaned rows: 283726
- Fraud rate after cleaning: 0.1667%
- Selected feature count: 13
- Selected features: V14_V12_interaction, V14, V17_V16_interaction, V12, V17, V10, V4, V16, V3, V11, V7, V18, log_amount

## Decision System

- Primary high-risk features support `BLOCK`.
- Supporting fraud features support `REVIEW`.
- Weak signal supports `APPROVE`.
- Initial threshold design: `> 0.85 -> BLOCK`, `0.60 to 0.85 -> REVIEW`, `< 0.60 -> APPROVE`.

## Modeling Plan

- Run Logistic Regression and Random Forest baselines.
- Handle imbalance with class weighting first.
- Evaluate with recall, precision, F1, and PR-AUC.
- Tune thresholds after validation.

## Final Recommendation

- Use the finalized non-redundant feature set as the shared starting point for both baseline models.
- Prioritize fraud recall and decision-threshold validation.
- Treat the next completed milestone as model comparison and threshold calibration, not more EDA.

## Saved Tables

- `reports/tables/12_final_summary/project_summary.csv`
- `reports/tables/12_final_summary/final_summary_report.md`
