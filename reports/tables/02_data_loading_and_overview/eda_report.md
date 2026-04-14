# EDA Report

## Data Overview Observations

- The dataset is large enough for meaningful fraud analysis.
- The target variable is highly imbalanced, which will affect model evaluation strategy.
- `Time` and `Amount` are the most directly interpretable numerical variables.
- The PCA-based variables contain predictive information, but their business interpretation is limited.
- `Amount` is heavily right-skewed and will likely require log transformation or standard scaling before model training.
- `Time` spans roughly 48 hours and may require normalization or cyclical encoding depending on the selected model.
- Data quality findings from missing values and duplicates should guide the next cleaning and EDA steps.

## Saved Tables

- `reports/tables/02_data_loading_and_overview/dataset_schema_summary.csv`
- `reports/tables/02_data_loading_and_overview/missing_values_summary.csv`
- `reports/tables/02_data_loading_and_overview/duplicate_rows_summary.csv`
- `reports/tables/02_data_loading_and_overview/target_class_distribution.csv`
- `reports/tables/02_data_loading_and_overview/numerical_summary_statistics.csv`
- `reports/tables/02_data_loading_and_overview/time_amount_summary_statistics.csv`
- `reports/tables/02_data_loading_and_overview/eda_report.md`

## Saved Figures

- `reports/figures/02_data_loading_and_overview/target_class_distribution.png`
