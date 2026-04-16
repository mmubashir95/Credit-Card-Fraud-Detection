# Feature Engineering Report

## Key Findings

- The engineered dataset keeps the strongest PCA features unchanged and adds retained raw-derived features, ratio-based amount features, and a small set of interaction features.
- `Amount` and `Time` are retained in raw form, while only leakage-safe derived variants are stored in the final feature sheet.
- Multiple interaction and ratio features are created because fraud signals are more useful when transaction magnitude is evaluated alongside high-signal PCA behavior.

## Feature Engineering Interpretation

- The strongest PCA features are retained unchanged because they already represent the most stable and informative transformed signals in the dataset.
- `Amount` and `Time` are not discarded; instead, they are reviewed through log, ratio, and time-derived transformations while the raw variables are preserved for downstream modeling.
- Multiple interaction features are created because fraud behavior is unlikely to be explained by one high-signal PCA component alone.
- Ratio and magnitude features are added because suspicious transactions are often unusual relative to a normal transaction baseline, not only in absolute value.
- The interaction set is intentionally small and targeted so that later notebooks can evaluate usefulness without creating unnecessary feature explosion.
- `time_normalized` is kept only as an exploratory screening feature in this notebook and is excluded from the saved engineered dataset to avoid mixing train-fitted normalization with stored feature assets.

## Connection to Modeling and Decision System

- The engineered feature set should improve the fraud-risk model by combining strong PCA features with transformed raw variables, ratio-based amount signals, and a small number of high-value interactions.
- Interaction features such as `V17_V14_interaction`, `V17_V12_interaction`, and `V17_V10_interaction` can help the model capture joint fraud signatures that may strengthen `BLOCK` and `REVIEW` decisions.
- Ratio features such as `amount_to_mean_ratio`, `amount_to_median_ratio`, and `amount_V17_interaction` help the model recognize transactions that are unusually large for the dataset context or unusually large in combination with the strongest fraud PCA signal.
- `log_amount` and `time_day_fraction` are retained in the saved dataset because they are stable derived features rather than train-fitted scaling steps.
- Any normalization or standardization of `Time` should be handled later inside the modeling pipeline after the train-test split, not stored in the final feature sheet.
- The final engineered dataset is designed to support both linear baselines and more flexible tree-based models before the later feature-selection stage narrows the final set.

## Key Insights

- Feature engineering should focus on a controlled set of high-signal additions rather than creating many untested derived variables.
- The top PCA features remain the core predictive signals, while transformed raw features, ratio features, and interactions act as supporting enhancements.
- Multiple interaction features are justified because the multivariate stage suggested that fraud behavior follows structured, non-linear feature relationships.
- Ratio-based amount features complement the PCA space by expressing how extreme a transaction is relative to common transaction size baselines.
- The engineered dataset created here is the correct input for the next notebooks on outlier analysis and feature selection.

## Saved Tables

- `reports/tables/08_feature_engineering/base_feature_set.csv`
- `reports/tables/08_feature_engineering/raw_feature_transformations.csv`
- `reports/tables/08_feature_engineering/interaction_feature_summary.csv`
- `reports/tables/08_feature_engineering/ratio_feature_summary.csv`
- `reports/tables/08_feature_engineering/engineered_feature_screening.csv`
- `reports/tables/08_feature_engineering/feature_comparison_summary.csv`
- `reports/tables/08_feature_engineering/engineered_dataset_quality_summary.csv`
- `reports/tables/08_feature_engineering/feature_engineering_report.md`

## Saved Figures

- `reports/figures/08_feature_engineering/feature_engineering_comparison.png`

## Saved Data Assets

- `/Users/mmubashir/VCode/Credit-Card-Fraud-Detection/data/processed/creditcard_engineered.csv`
- `/Users/mmubashir/VCode/Credit-Card-Fraud-Detection/artifacts/feature_columns.json`
