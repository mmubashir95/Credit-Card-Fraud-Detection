# Outlier Analysis Report

## Key Findings

- The notebook reviews IQR-based outlier behavior across the engineered fraud-detection feature set rather than removing extremes automatically.
- Class-wise outlier analysis is used to determine whether tail regions are fraud-enriched and therefore potentially valuable for modeling.
- Ratio and interaction features receive special attention because engineered magnitude effects may amplify the anomaly signal.
- A direct `Amount` outlier check is included so the notebook explicitly tests whether high-amount outlier regions contain more fraud than non-outlier transactions.

## Outlier Interpretation

- Extreme values should be interpreted as candidate fraud signal first and only as potential noise second.
- In fraud detection, extreme values are not necessarily noise; they often represent abnormal behavior, which is exactly what fraud looks like.
- Outliers will not be removed blindly and should be preserved for modeling unless later evidence shows a clear benefit from targeted handling.
- Features such as `log_amount`, amount-based ratios, and PCA interactions are especially important because their tails may encode unusual transaction behavior.
- Any handling recommendations should be evaluated with downstream model sensitivity checks rather than applied blindly.

## Outlier Handling Strategy

| Feature Type | Action |
|-------------|--------|
| PCA Features | Keep (already transformed) |
| Amount | Apply log transform |
| Ratio Features | Scale / log if needed |
| Interaction Features | Keep, handle via scaling |
| Rows | Do NOT remove |

### Model Sensitivity to Outliers

- Logistic Regression -> sensitive -> needs scaling
- Tree models (RF, XGBoost) -> robust -> no issue

Therefore:
Outliers will be handled during modeling via scaling, not removal.

## Final Decision for Feature Engineering

- No rows removed
- Amount -> log transform
- Interaction features -> kept
- Scaling deferred to modeling pipeline

## Connection to Modeling and Decision System

- Fraud-enriched outlier regions can strengthen downstream `BLOCK` and `REVIEW` decisions by making anomaly-heavy transactions easier to separate.
- Robust modeling choices are preferable when outlier structure appears informative.
- The outlier review results should be used to inform the next feature-selection notebook rather than to justify broad row removal.

### Decision System Bridge

| Outlier Pattern | Implication for Decision System |
|---|---|
| High lift in fraud-enriched features | Strengthens `BLOCK` threshold |
| Moderate lift features | Informs `REVIEW` zone |
| No lift | Safe to treat as noise |

## Saved Tables

- `reports/tables/09_outlier_analysis/outlier_review_feature_list.csv`
- `reports/tables/09_outlier_analysis/iqr_outlier_summary.csv`
- `reports/tables/09_outlier_analysis/classwise_outlier_summary.csv`
- `reports/tables/09_outlier_analysis/amount_outlier_fraud_summary.csv`
- `reports/tables/09_outlier_analysis/outlier_handling_recommendations.csv`
- `reports/tables/09_outlier_analysis/outlier_analysis_report.md`

## Saved Figures

- `reports/figures/09_outlier_analysis/top_outlier_rate_features.png`
- `reports/figures/09_outlier_analysis/selected_feature_outlier_boxes.png`
