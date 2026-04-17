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

## Model Evaluation Results

⚠️ Model training and evaluation will be completed in the next phase.

Planned models:
- Logistic Regression
- Random Forest

Metrics to evaluate:
- Precision
- Recall
- F1-score
- ROC-AUC

Final model selection will be based on recall (fraud detection priority).

## Final Recommendation

- Use the finalized non-redundant feature set as the shared starting point for both baseline models.
- Train Logistic Regression and Random Forest in the next phase and compare them under the same fraud-focused metrics.
- Prioritize fraud recall and decision-threshold calibration for `BLOCK`, `REVIEW`, and `APPROVE`.

## Saved Tables

- `reports/tables/12_final_summary/project_summary.csv`
- `reports/tables/12_final_summary/final_summary_report.md`
