# Feature Engineering Report

## Key Findings

- The engineered dataset keeps the strongest PCA features unchanged and adds transformed raw features plus a small set of interaction features.
- `Amount` and `Time` are retained in raw form but are also converted into more model-friendly variants.
- Multiple interaction features are created because the strongest fraud signals are likely to interact rather than act independently.

## Feature Engineering Interpretation

- The strongest PCA features are retained unchanged because they already represent the most stable and informative transformed signals in the dataset.
- `Amount` and `Time` are not discarded; instead, they are converted into more model-friendly forms through log and scale-based transformations.
- Multiple interaction features are created because fraud behavior is unlikely to be explained by one high-signal PCA component alone.
- The interaction set is intentionally small and targeted so that later notebooks can evaluate usefulness without creating unnecessary feature explosion.

## Connection to Modeling and Decision System

- The engineered feature set should improve the fraud-risk model by combining strong PCA features with transformed raw variables and a small number of high-value interactions.
- Interaction features such as `V17_V14_interaction`, `V17_V12_interaction`, and `V17_V10_interaction` can help the model capture joint fraud signatures that may strengthen `BLOCK` and `REVIEW` decisions.
- Transformed raw features such as `log_amount` and `time_normalized` make it easier for baseline linear models to use contextual information without being dominated by scale problems.
- The final engineered dataset is designed to support both linear baselines and more flexible tree-based models before the later feature-selection stage narrows the final set.

## Key Insights

- Feature engineering should focus on a controlled set of high-signal additions rather than creating many untested derived variables.
- The top PCA features remain the core predictive signals, while transformed raw features and interactions act as supporting enhancements.
- Multiple interaction features are justified because the multivariate stage suggested that fraud behavior follows structured, non-linear feature relationships.
- The engineered dataset created here is the correct input for the next notebooks on outlier analysis and feature selection.

## Saved Tables

- `reports/tables/08_feature_engineering/base_feature_set.csv`
- `reports/tables/08_feature_engineering/raw_feature_transformations.csv`
- `reports/tables/08_feature_engineering/interaction_feature_summary.csv`
- `reports/tables/08_feature_engineering/engineered_feature_screening.csv`
- `reports/tables/08_feature_engineering/feature_comparison_summary.csv`
- `reports/tables/08_feature_engineering/engineered_dataset_quality_summary.csv`
- `reports/tables/08_feature_engineering/feature_engineering_report.md`

## Saved Figures

- `reports/figures/08_feature_engineering/feature_engineering_comparison.png`

## Saved Data Assets

- `/Users/mmubashir/VCode/Credit-Card-Fraud-Detection/data/processed/creditcard_engineered.csv`
- `/Users/mmubashir/VCode/Credit-Card-Fraud-Detection/artifacts/feature_columns.json`
