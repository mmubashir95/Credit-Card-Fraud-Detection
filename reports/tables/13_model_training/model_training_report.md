# Model Training Report

## Purpose
- Train the baseline fraud-detection models on the finalized selected-feature dataset.
- Keep the holdout test set untouched during cross-validation.
- Save the trained artifacts and holdout probabilities for notebook `14_model_evaluation`.

## Training Inputs
- Modeling dataset: `/Users/mohammadmubashir/VCode/Credit-Card-Fraud-Detection/data/processed/creditcard_selected_features.csv`
- Selected feature count: `13`
- Total rows: `283726`
- Fraud rate: `0.001667`
- Test size: `0.2`
- Random state: `42`
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training split only

## Models Trained
- Logistic Regression with median imputation, standard scaling, `max_iter=1000`, and `class_weight='balanced'`
- Random Forest with median imputation, `n_estimators=100`, and `class_weight='balanced'`

## Handoff to Next Notebook
- Read `cv_summary.csv` and `cv_fold_metrics.csv` to compare fold performance.
- Read `baseline_test_predictions.csv` to evaluate both final baseline models once on the untouched holdout test set.
- Read `logistic_regression_holdout_metrics.csv` for the baseline Logistic Regression confusion matrix and core holdout metrics.
- Read `random_forest_holdout_metrics.csv` for the baseline Random Forest confusion matrix and core holdout metrics.
