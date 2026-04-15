# Multivariate Analysis Report

## Key Findings

- The multivariate notebook focuses on the strongest bivariate signals plus `Amount` and `Time` as contextual variables.
- The main objective is to confirm whether the shortlisted features remain useful together or become redundant when analyzed jointly.
- Multicollinearity is evaluated using both pairwise correlation and Variance Inflation Factor (VIF).
- Fraud and non-fraud transactions may show different correlation structure even when single-feature summaries look similar.

## Multivariate Interpretation

- The shortlisted PCA variables should be treated as the core multivariate feature set because they carry the strongest direct signal from the bivariate stage.
- Low pairwise correlation and low VIF among the selected PCA features would support keeping multiple components in the same model without severe redundancy.
- Differences between fraud and non-fraud correlation structure may indicate that fraud is characterized not only by unusual values, but also by unusual feature interactions.
- `Amount` and `Time` remain useful as contextual variables, but they are unlikely to dominate model performance without transformation or interaction-based feature engineering.

## Impact on Modeling

- Linear models such as logistic regression can benefit from the low-collinearity PCA features, especially if the strongest components remain stable in the multivariate view and VIF stays in an acceptable range.
- Tree-based models remain suitable because they can capture non-linear boundaries and interaction effects that may not appear in pairwise summaries alone.
- `Amount` should continue to be used in transformed form when appropriate, while `Time` may become more useful through derived features rather than raw inclusion alone.
- The multivariate stage should be used to finalize which features move forward unchanged, which need engineering, and which can be deprioritized.

## Key Insights

- `V17`, `V14`, `V12`, `V10`, `V16`, `V3`, `V7`, `V11`, `V4`, and `V18` form the main multivariate candidate set from the bivariate stage.
- The multivariate correlation matrix should confirm whether these features remain complementary rather than redundant.
- Class-specific correlation differences can reveal fraud-specific interaction structure that is not visible in single-feature analysis.
- `Amount` and `Time` remain part of the analysis as contextual inputs, but they still depend on transformation or combination to add stronger predictive value.
- These findings should directly support feature engineering and model-building decisions in the next project stage.

## Saved Tables

- `reports/tables/07_multivariate_analysis/selected_feature_set.csv`
- `reports/tables/07_multivariate_analysis/multivariate_correlation_matrix.csv`
- `reports/tables/07_multivariate_analysis/high_correlation_pairs.csv`
- `reports/tables/07_multivariate_analysis/vif_summary.csv`
- `reports/tables/07_multivariate_analysis/class_0_correlation_matrix.csv`
- `reports/tables/07_multivariate_analysis/class_1_correlation_matrix.csv`
- `reports/tables/07_multivariate_analysis/correlation_difference_matrix.csv`
- `reports/tables/07_multivariate_analysis/top_feature_interaction_summary.csv`
- `reports/tables/07_multivariate_analysis/multivariate_report.md`

## Saved Figures

- `reports/figures/07_multivariate_analysis/priority_feature_correlation_heatmap.png`
- `reports/figures/07_multivariate_analysis/class_specific_correlation_heatmaps.png`
- `reports/figures/07_multivariate_analysis/correlation_difference_heatmap.png`
- `reports/figures/07_multivariate_analysis/v17_v14_interaction.png`
- `reports/figures/07_multivariate_analysis/v17_v12_interaction.png`
