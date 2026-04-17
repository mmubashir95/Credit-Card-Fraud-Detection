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

## Handling Class Imbalance

- The dataset is highly imbalanced, with fraud making up about `0.17%` of cleaned transactions.
- Strategy used for the modeling phase:
- `class_weight='balanced'` to make fraud errors count more during training.
- Threshold tuning to support fraud-focused `BLOCK`, `REVIEW`, and `APPROVE` decisions instead of relying on a default cutoff.
- Impact on model behavior:
- The model becomes more sensitive to rare fraud cases instead of favoring the majority non-fraud class.
- This usually increases fraud recall, but it can also increase false positives and manual review volume.
- The project therefore prioritizes recall-first behavior and then controls operational cost through threshold design.

## Decision System

- Primary high-risk features support `BLOCK`.
- Supporting fraud features support `REVIEW`.
- Weak signal supports `APPROVE`.
- Exact thresholds: `BLOCK > 0.85`, `REVIEW 0.60 to 0.85`, `APPROVE < 0.60`.
- High recall matters because missed fraud is more costly than additional manual review.
- The threshold is not `0.50` because fraud detection is highly imbalanced and requires operational decision bands rather than a default binary cutoff.

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
