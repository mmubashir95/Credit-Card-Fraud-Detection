# Bivariate Analysis Report

## Key Findings

- `Amount` and `Time` should be evaluated against the fraud target because they are the most directly interpretable variables in the dataset.
- Selected PCA features may show clearer separation between fraud and non-fraud classes than raw variables alone.
- Class-wise distribution differences provide useful evidence for feature selection and modeling priorities.

## Bivariate Interpretation

- Differences in class-wise amount behavior may strengthen fraud detection once the feature is transformed appropriately.
- Time patterns are more likely to act as supporting context than as standalone fraud triggers.
- PCA variables that shift more clearly by class may become strong inputs for supervised models.
- Correlation with `Class` should be treated as a screening signal, not a final feature-selection rule.

## Statistical Significance Test

- The top 5 features by absolute relationship with `Class` should be tested using the Mann-Whitney U test.
- Very small p-values indicate that the fraud and non-fraud distributions differ significantly for those features.
- Rank-biserial correlation should be used alongside p-values to measure the strength of class separation.
- Features with moderate or strong absolute effect size should receive higher priority in downstream modeling.

## Strongest Separating Features

- Features with strong or moderate absolute rank-biserial correlation should be treated as the most practically important class separators.
- These features deserve higher priority in multivariate analysis, feature engineering, and downstream modeling.

## Final Correlation-Based Feature Importance Table

- The final correlation table converts raw feature correlations into business-friendly importance labels: `Strong`, `Moderate`, and `Weak`.
- Strong features should receive the highest modeling priority, moderate features should be retained as supporting predictors, and weak features should not be removed blindly without multivariate review.
- This table is intended as a screening summary rather than a final feature-selection rule because some weak linear features may still be useful in non-linear models.

## Final Bivariate Insights

- `V4` and `V11` show strong practical separation between fraud and non-fraud behavior and should be treated as highly predictive candidates.
- `V3` shows moderate separation, which makes it useful as a supporting predictive feature.
- `Amount` and `Time` show weaker standalone relationship with the target and should be treated mainly as contextual variables.
- PCA-based features capture more complex fraud patterns than raw variables and therefore deserve higher feature-selection priority.
- These findings should guide feature selection, multivariate analysis, and downstream model design.

## Key Insights

- `Amount` may contribute to fraud separation through differences in spread and abnormal value behavior.
- `Time` may add supporting risk context when used with other features.
- PCA features that show stronger class-wise separation deserve priority in later modeling stages.
- Bivariate analysis improves model readiness by identifying which variables are more informative with respect to the fraud target.

## Saved Tables

- `reports/tables/06_bivariate_analysis/amount_by_class_summary.csv`
- `reports/tables/06_bivariate_analysis/time_by_class_summary.csv`
- `reports/tables/06_bivariate_analysis/selected_pca_by_class_summary.csv`
- `reports/tables/06_bivariate_analysis/correlation_with_target.csv`
- `reports/tables/06_bivariate_analysis/mann_whitney_top5_features.csv`
- `reports/tables/06_bivariate_analysis/final_feature_importance_table.csv`
- `reports/tables/06_bivariate_analysis/bivariate_report.md`

## Saved Figures

- `reports/figures/06_bivariate_analysis/amount_by_class_boxplot.png`
- `reports/figures/06_bivariate_analysis/amount_by_class_distribution.png`
- `reports/figures/06_bivariate_analysis/time_by_class_boxplot.png`
- `reports/figures/06_bivariate_analysis/time_by_class_distribution.png`
- `reports/figures/06_bivariate_analysis/v1_by_class_boxplot.png`
- `reports/figures/06_bivariate_analysis/v1_by_class_distribution.png`
- `reports/figures/06_bivariate_analysis/v2_by_class_boxplot.png`
- `reports/figures/06_bivariate_analysis/v2_by_class_distribution.png`
- `reports/figures/06_bivariate_analysis/v3_by_class_boxplot.png`
- `reports/figures/06_bivariate_analysis/v3_by_class_distribution.png`
- `reports/figures/06_bivariate_analysis/v4_by_class_boxplot.png`
- `reports/figures/06_bivariate_analysis/v4_by_class_distribution.png`
- `reports/figures/06_bivariate_analysis/v5_by_class_boxplot.png`
- `reports/figures/06_bivariate_analysis/v5_by_class_distribution.png`
- `reports/figures/06_bivariate_analysis/v6_by_class_boxplot.png`
- `reports/figures/06_bivariate_analysis/v6_by_class_distribution.png`
- `reports/figures/06_bivariate_analysis/top_feature_correlations_with_class.png`
