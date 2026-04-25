# Model Training Report

## Purpose
- Train the baseline fraud-detection models on the finalized selected-feature dataset.
- Save the trained artifacts and test-set probabilities for notebook `14_model_evaluation`.

## Training Inputs
- Modeling dataset: `/Users/mohammadmubashir/VCode/Credit-Card-Fraud-Detection/data/processed/creditcard_selected_features.csv`
- Selected feature count: `13`
- Total rows: `283726`
- Fraud rate: `0.001667`
- Test size: `0.2`
- Random state: `42`

## Models Trained
- Logistic Regression with median imputation, standard scaling, and `class_weight='balanced'`
- Random Forest with median imputation and `class_weight='balanced'`

## Handoff to Next Notebook
- Evaluate `baseline_test_predictions.csv` in notebook `14_model_evaluation`.
- Compare the two models using precision, recall, F1, PR-AUC, ROC-AUC, and confusion matrices.
