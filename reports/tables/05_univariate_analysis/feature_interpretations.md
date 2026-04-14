# Feature Interpretations

## Why `V1` to `V28` Should Not Be Further Transformed

The features `V1` to `V28` are principal components obtained through PCA (Principal Component Analysis). They are not original business variables, but transformed combinations of the original confidential features.

These PCA-based features are already centered, scaled, and decorrelated through the PCA process. As a result, they are already in a model-ready numerical form compared with raw variables such as `Amount` and `Time`.

Applying additional transformations such as log transformation is not appropriate because PCA components may contain negative values and do not represent raw measurable quantities. Re-scaling them again is usually unnecessary unless the modeling pipeline applies one common scaler to all numerical inputs for implementation consistency.

In practice, `V1` to `V28` should be used directly as input features, while preprocessing efforts should focus mainly on raw variables such as `Amount` and `Time`.

## Outlier Interpretation in Fraud Detection

In this dataset, outliers represent transactions that fall far outside the typical range of observed behavior based on the selected statistical threshold. These unusual values may appear in features such as `Amount`, `Time`, or certain PCA-transformed components, and they should not automatically be treated as data errors. In a fraud detection setting, such extreme observations can reflect the exact abnormal patterns the model is expected to identify.

A high outlier count is not surprising in fraud detection because fraudulent transactions are, by nature, rare and behaviorally different from normal transactions. Even some legitimate transactions may appear extreme due to unusual customer behavior, but these rare patterns still matter from a risk-monitoring perspective. For this reason, a large number of outliers does not necessarily indicate poor data quality; it often indicates the presence of meaningful anomaly-related information.

The recommended decision is to **retain outliers**, not remove them blindly. Removing them would risk deleting high-value fraud signals and making the model less sensitive to suspicious behavior. At the same time, extreme values in raw features such as `Amount` may still create instability for some models, so targeted preprocessing such as log transformation or scaling is more appropriate than deletion. For PCA-based variables, outliers should generally be preserved because they may carry important anomaly structure already captured by the transformed feature space.

From an industry ML perspective, the correct approach is to treat outliers as potential risk signals first and preprocessing challenges second. The objective is not to make the dataset look statistically clean, but to preserve the patterns that help distinguish fraud from legitimate activity. Therefore, outliers should be retained, monitored carefully, and handled through robust feature transformation where necessary rather than removed through a blanket rule.

## Feature: `Time`
- **Skewness interpretation:** `Time` is close to symmetric, which means transaction activity is not perfectly uniform across the observation window. Some periods contain denser transaction activity than others, which may matter when comparing behavior across fraud and non-fraud cases.
- **Kurtosis interpretation:** `Time` has relatively low kurtosis, so its distribution is flatter and less dominated by sharp extremes. This usually means the feature is more stable and less dependent on rare tail events.
- **Fraud detection implication:** For fraud detection, transaction timing can add contextual signal, but it is rarely decisive on its own. Its strongest value usually comes when combined with amount, PCA features, or engineered time-based features.
- **Transformation decision:** Normalization or standard scaling is recommended, with optional time-based feature engineering.
- **Recommended action:** Scale `Time` for baseline models and evaluate derived time-window or cyclical features later.

## Feature: `V1`
- **Skewness interpretation:** `V1` is left-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V1` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V1` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V2`
- **Skewness interpretation:** `V2` is left-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V2` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V2` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V3`
- **Skewness interpretation:** `V3` is left-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V3` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V3` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V4`
- **Skewness interpretation:** `V4` is right-skewed, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V4` has moderate kurtosis, indicating a balanced spread with some unusual values but without extreme tail dominance.
- **Fraud detection implication:** For fraud detection, `V4` is more likely to act as a supporting signal than a standalone trigger. Its value should be judged by how much it improves model separation when combined with other variables.
- **Transformation decision:** Standard scaling is sufficient.
- **Recommended action:** Retain the feature and scale it with the rest of the numerical inputs.

## Feature: `V5`
- **Skewness interpretation:** `V5` is left-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V5` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V5` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V6`
- **Skewness interpretation:** `V6` is right-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V6` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V6` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V7`
- **Skewness interpretation:** `V7` is right-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V7` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V7` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V8`
- **Skewness interpretation:** `V8` is left-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V8` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V8` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V9`
- **Skewness interpretation:** `V9` is right-skewed, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V9` has elevated kurtosis, which means extreme observations occur more often than in a stable bell-shaped distribution. This makes the feature more sensitive to abnormal cases that may matter in risk detection.
- **Fraud detection implication:** For fraud detection, `V9` is more likely to act as a supporting signal than a standalone trigger. Its value should be judged by how much it improves model separation when combined with other variables.
- **Transformation decision:** Standard scaling is sufficient.
- **Recommended action:** Retain the feature and scale it with the rest of the numerical inputs.

## Feature: `V10`
- **Skewness interpretation:** `V10` is right-skewed, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V10` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V10` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V11`
- **Skewness interpretation:** `V11` is close to symmetric, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V11` has moderate kurtosis, indicating a balanced spread with some unusual values but without extreme tail dominance.
- **Fraud detection implication:** For fraud detection, `V11` is more likely to act as a supporting signal than a standalone trigger. Its value should be judged by how much it improves model separation when combined with other variables.
- **Transformation decision:** Standard scaling is sufficient.
- **Recommended action:** Retain the feature and scale it with the rest of the numerical inputs.

## Feature: `V12`
- **Skewness interpretation:** `V12` is left-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V12` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V12` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V13`
- **Skewness interpretation:** `V13` is close to symmetric, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V13` has moderate kurtosis, indicating a balanced spread with some unusual values but without extreme tail dominance.
- **Fraud detection implication:** For fraud detection, `V13` is more likely to act as a supporting signal than a standalone trigger. Its value should be judged by how much it improves model separation when combined with other variables.
- **Transformation decision:** Standard scaling is sufficient.
- **Recommended action:** Retain the feature and scale it with the rest of the numerical inputs.

## Feature: `V14`
- **Skewness interpretation:** `V14` is left-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V14` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V14` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V15`
- **Skewness interpretation:** `V15` is close to symmetric, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V15` has moderate kurtosis, indicating a balanced spread with some unusual values but without extreme tail dominance.
- **Fraud detection implication:** For fraud detection, `V15` is more likely to act as a supporting signal than a standalone trigger. Its value should be judged by how much it improves model separation when combined with other variables.
- **Transformation decision:** Standard scaling is sufficient.
- **Recommended action:** Retain the feature and scale it with the rest of the numerical inputs.

## Feature: `V16`
- **Skewness interpretation:** `V16` is left-skewed, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V16` has elevated kurtosis, which means extreme observations occur more often than in a stable bell-shaped distribution. This makes the feature more sensitive to abnormal cases that may matter in risk detection.
- **Fraud detection implication:** For fraud detection, unusual values in `V16` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V17`
- **Skewness interpretation:** `V17` is left-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V17` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V17` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V18`
- **Skewness interpretation:** `V18` is close to symmetric, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V18` has moderate kurtosis, indicating a balanced spread with some unusual values but without extreme tail dominance.
- **Fraud detection implication:** For fraud detection, `V18` is more likely to act as a supporting signal than a standalone trigger. Its value should be judged by how much it improves model separation when combined with other variables.
- **Transformation decision:** Standard scaling is sufficient.
- **Recommended action:** Retain the feature and scale it with the rest of the numerical inputs.

## Feature: `V19`
- **Skewness interpretation:** `V19` is close to symmetric, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V19` has moderate kurtosis, indicating a balanced spread with some unusual values but without extreme tail dominance.
- **Fraud detection implication:** For fraud detection, `V19` is more likely to act as a supporting signal than a standalone trigger. Its value should be judged by how much it improves model separation when combined with other variables.
- **Transformation decision:** Standard scaling is sufficient.
- **Recommended action:** Retain the feature and scale it with the rest of the numerical inputs.

## Feature: `V20`
- **Skewness interpretation:** `V20` is left-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V20` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V20` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V21`
- **Skewness interpretation:** `V21` is right-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V21` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V21` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V22`
- **Skewness interpretation:** `V22` is close to symmetric, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V22` has moderate kurtosis, indicating a balanced spread with some unusual values but without extreme tail dominance.
- **Fraud detection implication:** For fraud detection, `V22` is more likely to act as a supporting signal than a standalone trigger. Its value should be judged by how much it improves model separation when combined with other variables.
- **Transformation decision:** Standard scaling is sufficient.
- **Recommended action:** Retain the feature and scale it with the rest of the numerical inputs.

## Feature: `V23`
- **Skewness interpretation:** `V23` is left-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V23` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V23` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V24`
- **Skewness interpretation:** `V24` is left-skewed, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V24` has moderate kurtosis, indicating a balanced spread with some unusual values but without extreme tail dominance.
- **Fraud detection implication:** For fraud detection, `V24` is more likely to act as a supporting signal than a standalone trigger. Its value should be judged by how much it improves model separation when combined with other variables.
- **Transformation decision:** Standard scaling is sufficient.
- **Recommended action:** Retain the feature and scale it with the rest of the numerical inputs.

## Feature: `V25`
- **Skewness interpretation:** `V25` is close to symmetric, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V25` has elevated kurtosis, which means extreme observations occur more often than in a stable bell-shaped distribution. This makes the feature more sensitive to abnormal cases that may matter in risk detection.
- **Fraud detection implication:** For fraud detection, `V25` is more likely to act as a supporting signal than a standalone trigger. Its value should be judged by how much it improves model separation when combined with other variables.
- **Transformation decision:** Standard scaling is sufficient.
- **Recommended action:** Retain the feature and scale it with the rest of the numerical inputs.

## Feature: `V26`
- **Skewness interpretation:** `V26` is right-skewed, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V26` has moderate kurtosis, indicating a balanced spread with some unusual values but without extreme tail dominance.
- **Fraud detection implication:** For fraud detection, `V26` is more likely to act as a supporting signal than a standalone trigger. Its value should be judged by how much it improves model separation when combined with other variables.
- **Transformation decision:** Standard scaling is sufficient.
- **Recommended action:** Retain the feature and scale it with the rest of the numerical inputs.

## Feature: `V27`
- **Skewness interpretation:** `V27` is left-skewed, which suggests a relatively stable distribution with only moderate directional imbalance. Most transactions remain close to the central pattern represented by this transformed feature.
- **Kurtosis interpretation:** `V27` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V27` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `V28`
- **Skewness interpretation:** `V28` is right-skewed, which means most transactions stay near the center of this PCA component while a smaller set extends sharply in one direction. That pattern suggests a minority of transactions behave very differently from the dominant transaction profile captured by this feature.
- **Kurtosis interpretation:** `V28` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual values in `V28` may help isolate abnormal transaction signatures that differ from the majority class. This feature is more useful as part of a multi-feature fraud pattern than as a standalone rule.
- **Transformation decision:** Standard scaling is recommended, with additional tail monitoring.
- **Recommended action:** Keep the feature, scale it consistently, and monitor whether extreme values destabilize the model.

## Feature: `Amount`
- **Skewness interpretation:** `Amount` is strongly right-skewed, which means most transactions are concentrated at lower amounts while a smaller number extend far into higher values, reaching about 25,691.16. In practice, this means a limited set of large transactions can dominate the raw distribution and distort model sensitivity if no preprocessing is applied.
- **Kurtosis interpretation:** `Amount` has very high kurtosis, indicating heavy tails and a meaningful concentration of rare extreme values. This suggests the feature captures unusual transaction behavior that may be operationally important in fraud scoring.
- **Fraud detection implication:** For fraud detection, unusual transaction amounts can be important because both very small test charges and unusually large transactions may indicate abnormal behavior. However, the raw scale can distort model training and make decision boundaries less stable if the feature is left untreated.
- **Transformation decision:** Log transformation and standard scaling are recommended.
- **Recommended action:** Apply `log1p` to reduce tail dominance and then scale the feature before training.
