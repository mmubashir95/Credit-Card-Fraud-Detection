# Cleaning Report

## Project Context

This cleaning stage prepares the credit card fraud dataset for downstream feature engineering, modeling, threshold tuning, and decision logic design. The approach is intentionally conservative because fraud datasets contain rare but valuable minority-class patterns that can be damaged by over-cleaning.

## Initial Audit Summary

- Input shape: (284807, 31)
- Duplicate rows before cleaning: 1081
- Total missing values before cleaning: 0
- Object-type columns before cleaning: 0
- Fraud rate before cleaning: 0.1727%

## Cleaning Decisions

- Exact duplicate rows were removed.
- No blanket missing-value imputation was applied because both the standard missing-value audit and the hidden-missing review found no evidence of incomplete fields.
- No blanket outlier removal was applied because rare extreme patterns may be meaningful fraud signals.
- Low-information features and leakage risk were reviewed and documented rather than changed automatically.

## Final Validation Summary

- Output shape: (283726, 31)
- Remaining duplicates: 0
- Total missing values after cleaning: 0
- Final validation status: PASS
- Final validation message: Cleaned dataset passed the implemented validation checks.
- Fraud rate after cleaning: 0.1667%

## Modeling Readiness

The cleaned dataset is more reliable for fraud modeling because duplicate leakage risk has been reduced while minority-class structure has been preserved. The dataset is ready for feature engineering and later modeling work, where imbalance-aware evaluation, threshold tuning, and business decision mapping can be handled more directly.

## Saved Tables

- `reports/tables/04_data_cleaning/initial_data_quality_audit.csv`
- `reports/tables/04_data_cleaning/missing_value_overview.csv`
- `reports/tables/04_data_cleaning/missing_value_assessment.csv`
- `reports/tables/04_data_cleaning/hidden_missing_overview.csv`
- `reports/tables/04_data_cleaning/hidden_missing_value_assessment.csv`
- `reports/tables/04_data_cleaning/dtype_audit.csv`
- `reports/tables/04_data_cleaning/unique_value_summary.csv`
- `reports/tables/04_data_cleaning/descriptive_statistics.csv`
- `reports/tables/04_data_cleaning/class_distribution_before_cleaning.csv`
- `reports/tables/04_data_cleaning/duplicate_impact_summary.csv`
- `reports/tables/04_data_cleaning/duplicate_removal_summary.csv`
- `reports/tables/04_data_cleaning/class_distribution_after_cleaning.csv`
- `reports/tables/04_data_cleaning/class_distribution_change_summary.csv`
- `reports/tables/04_data_cleaning/cleaned_dtype_audit.csv`
- `reports/tables/04_data_cleaning/feature_range_summary.csv`
- `reports/tables/04_data_cleaning/impossible_value_checks.csv`
- `reports/tables/04_data_cleaning/constant_feature_summary.csv`
- `reports/tables/04_data_cleaning/near_constant_feature_summary.csv`
- `reports/tables/04_data_cleaning/outlier_awareness_summary.csv`
- `reports/tables/04_data_cleaning/leakage_review_summary.csv`
- `reports/tables/04_data_cleaning/cleaning_actions_summary.csv`
- `reports/tables/04_data_cleaning/final_validation_summary.csv`
- `reports/tables/04_data_cleaning/cleaning_report.md`

## Saved Figures

- `reports/figures/04_data_cleaning/class_distribution_before_cleaning.png`
- `reports/figures/04_data_cleaning/class_distribution_before_after_cleaning.png`

## Final Judgment

The cleaned dataset is structurally stronger and more reliable for fraud modeling than the raw input because exact duplicates have been removed and data quality checks have been documented comprehensively. The fraud class distribution has been preserved, aggressive outlier removal has been avoided, and the dataset is ready for feature engineering, modeling, threshold tuning, and future deployment-oriented scoring workflows.
