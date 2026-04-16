# Feature Selection Report

## Key Findings

- This notebook combines univariate relevance, redundancy checks, and lightweight model-based signals to decide which engineered features move into modeling.
- Features are not selected by correlation alone; overlap between features is reviewed so the final set stays informative without unnecessary duplication.
- Logistic Regression and Random Forest are used as early model-aware checks to see whether the same features remain important once we move closer to modeling.
- The final output is a modeling-ready selected dataset plus an explicit keep or drop decision for every engineered feature.
- The amount family is resolved explicitly: neither `Amount`, `log_amount`, nor the amount ratios are retained in the final selected dataset, although `log_amount` remains the preferred fallback representation for future sensitivity checks.

## Feature Selection Logic

- Relevance is measured through correlation with `Class` and standardized fraud vs non-fraud separation.
- Redundancy is flagged through pairwise absolute feature correlation so duplicate signals can be pruned.
- Flagged redundancy pairs are also resolved into explicit correlation groups so each group states which feature(s) are kept and which are dropped.
- Model-based evidence is added using Logistic Regression coefficients and Random Forest importances.
- The default feature set follows the stricter Logistic Regression-friendly rule: redundant predictors are removed first to control multicollinearity, while tree models may still revisit some dropped correlated features in sensitivity checks.
- Features are assigned to `KEEP`, `KEEP_MONITOR`, `DROP_REDUNDANCY`, or `DROP_WEAK`, but only `KEEP` features are exported into the default modeling dataset.

## Connection to Modeling and Decision System

- Retained features form the default input space for baseline fraud models.
- Any future `KEEP_MONITOR` features should remain outside the default export until a later validation step promotes them.
- Removing redundant or weak features makes downstream model behavior easier to explain and calibrate.
- A cleaner feature set supports more stable `BLOCK`, `REVIEW`, and `APPROVE` rules in the later decision system.

## Saved Tables

- `reports/tables/10_feature_selection/feature_metadata_overview.csv`
- `reports/tables/10_feature_selection/feature_relevance_summary.csv`
- `reports/tables/10_feature_selection/feature_redundancy_pairs.csv`
- `reports/tables/10_feature_selection/feature_redundancy_summary.csv`
- `reports/tables/10_feature_selection/redundancy_group_decisions.csv`
- `reports/tables/10_feature_selection/amount_feature_resolution.csv`
- `reports/tables/10_feature_selection/model_feature_signal_summary.csv`
- `reports/tables/10_feature_selection/model_specific_feature_guidance.csv`
- `reports/tables/10_feature_selection/feature_selection_decisions.csv`
- `reports/tables/10_feature_selection/selected_feature_list.csv`
- `reports/tables/10_feature_selection/dropped_feature_list.csv`
- `reports/tables/10_feature_selection/final_feature_selection_summary.csv`
- `reports/tables/10_feature_selection/feature_selection_report.md`

## Saved Artifacts

- `data/processed/creditcard_selected_features.csv`
- `artifacts/selected_feature_names.json`
