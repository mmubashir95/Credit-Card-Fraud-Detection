# Bivariate Analysis Report

## Key Findings

- `Amount` and `Time` should be evaluated against the fraud target because they are the most directly interpretable variables in the dataset.
- The PCA visualization section focuses on the top 6 PCA features by absolute correlation with `Class`, so the plotted features are aligned with later feature-priority findings.
- Class-wise distribution differences provide useful evidence for feature selection and modeling priorities.

## Bivariate Interpretation

- Differences in class-wise amount behavior may strengthen fraud detection once the feature is transformed appropriately.
- Time patterns are more likely to act as supporting context than as standalone fraud triggers.
- The PCA features visualized in this notebook are selected by absolute correlation with `Class`, which makes the class-wise plots directly relevant to downstream modeling decisions.
- Correlation with `Class` should be treated as a screening signal, not a final feature-selection rule.
- Because `Class` is a highly imbalanced binary target, Pearson correlation may understate rare-class relationships and should be interpreted together with later non-parametric testing.

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

## Connection to Modeling

Based on the bivariate analysis, `V17` stands out as the strongest correlation-based predictor and should receive high modeling priority. A second group of PCA features including `V14`, `V12`, `V10`, `V16`, `V3`, `V7`, `V11`, `V4`, and `V18` shows moderate relationship with the fraud target and should be retained as important supporting inputs. `Amount` and `Time` show weak standalone correlation with fraud, so they should not be relied on as primary predictors and instead should be improved through transformation, scaling, or interaction-based feature engineering. Overall, these results indicate that PCA-derived variables are likely to carry the strongest predictive signal, while raw variables remain useful mainly as contextual features in the final modeling pipeline.

## Key Insights

- `V17` is the strongest correlation-based feature in the current bivariate analysis and should be prioritized in downstream modeling and multivariate review.
- `V14`, `V12`, `V10`, `V16`, `V3`, `V7`, `V11`, `V4`, and `V18` form the main group of moderate-strength predictors and should be retained as important supporting features.
- `Amount` and `Time` remain weak standalone predictors, so their value will likely come from transformation, interaction effects, or combination with stronger PCA-based variables.
- PCA-derived features carry most of the direct predictive signal in the current dataset, which reinforces their importance for feature selection and model design.
- The bivariate stage provides a clear screening outcome: prioritize high-signal PCA features, retain moderate features for model support, and treat raw variables as contextual inputs.

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
