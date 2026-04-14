# Validation Report

## Validation Summary

- Overall validation status: PASS WITH WARNINGS
- Validation message: Raw data passes schema and value checks, but duplicate rows require cleaning before modeling.
- Missing required columns: None
- Total missing values: 0
- Duplicate rows: 1081
- Unique target values: [0, 1]

## Why These Checks Matter for Fraud Detection

Schema validation confirms that downstream feature engineering and training use the expected structure. Target validation ensures the binary fraud label is reliable for supervised learning. Missing-value checks reduce the risk of silent preprocessing failures. Duplicate detection matters because repeated records can distort class frequencies, inflate metrics, and leak across train-test splits. The extreme class imbalance is not a data error, but it is a modeling risk that must be handled carefully during evaluation and training. Amount distribution checks also matter because high-value transactions can have disproportionate fraud impact, and PCA range sanity checks help surface unusual feature magnitudes that may indicate preprocessing or data-quality issues.

## Key Findings

- The schema is valid for the expected fraud detection workflow.
- No missing values were detected in the raw dataset.
- The target column uses valid binary labels.
- Duplicate rows are present and must be handled before modeling.
- The class imbalance is substantial and should be managed during modeling rather than removed as a validation issue.
- Amount distributions and PCA ranges have been summarized to support downstream cleaning and modeling decisions.

## Validation Issue Summary

| issue | severity | impact on model | action in next notebook |
| --- | --- | --- | --- |
| Schema validity | Low | Expected fraud-detection fields are present, so feature extraction and training can proceed on the intended schema. | Preserve column names and data types during cleaning. |
| Missing values | Low | No immediate risk of null-driven failures or inconsistent imputations in downstream steps. | Reconfirm after duplicate handling and any new transformations. |
| Target validity | Low | Binary labels are consistent with supervised fraud classification. | Protect the target column from unintended remapping or leakage. |
| Duplicate rows | High | Duplicates can bias class frequencies, skew validation metrics, and leak repeated records across splits. | Identify and remove or justify duplicates before feature engineering and model training. |
| Class imbalance | High | The minority fraud class represents a very small share of observations, so accuracy alone may be misleading and the model may under-detect fraud. | Keep the imbalance intact in cleaned data and plan imbalance-aware evaluation and modeling choices later. |

## Final Judgment

The raw dataset is structurally valid for fraud modeling, but duplicate records are a material data-quality issue that must be resolved before cleaning, feature engineering, and model training. In addition, the extreme class imbalance should be preserved carefully and handled during modeling rather than treated as a data error.

## Saved Tables

- `reports/tables/03_data_validation/schema_validation.csv`
- `reports/tables/03_data_validation/dtype_validation.csv`
- `reports/tables/03_data_validation/missing_value_validation.csv`
- `reports/tables/03_data_validation/duplicate_validation_summary.csv`
- `reports/tables/03_data_validation/duplicate_class_distribution.csv`
- `reports/tables/03_data_validation/target_validation.csv`
- `reports/tables/03_data_validation/range_rule_checks.csv`
- `reports/tables/03_data_validation/extreme_values_summary.csv`
- `reports/tables/03_data_validation/amount_percentile_check.csv`
- `reports/tables/03_data_validation/fraud_vs_legit_amount_comparison.csv`
- `reports/tables/03_data_validation/pca_column_range_sanity_check.csv`
- `reports/tables/03_data_validation/validation_status_summary.csv`
- `reports/tables/03_data_validation/validation_issue_summary.csv`
- `reports/tables/03_data_validation/validation_report.md`

## Saved Figures

- `reports/figures/03_data_validation/duplicate_class_distribution.png`
- `reports/figures/03_data_validation/class_imbalance_visualization.png`
