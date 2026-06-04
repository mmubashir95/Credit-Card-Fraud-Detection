Here is a strong **Codex instructions file** you can save as:

```text
AGENTS.md
```

This is for your **Credit Card Fraud Detection + NLP Project**. It follows your project purpose: ML fraud detection, BLOCK / REVIEW / APPROVE decision logic, complaint NLP, FastAPI, and deployment.  It also respects your notebook-based ML workflow and EDA structure. 

````md
# AGENTS.md — Codex Instructions

## Project Context

This is an end-to-end AI project for Credit Card Fraud Detection and Complaint Analysis.

The final system should:

- Detect fraudulent transactions using machine learning.
- Return fraud probability.
- Apply business decision logic:
  - BLOCK
  - REVIEW
  - APPROVE
- Analyze customer complaints using NLP.
- Expose the final system through a FastAPI API.
- Be understandable as both an academic project and a portfolio-level industry project.

Expected final output example:

```json
{
  "fraud_probability": 0.91,
  "decision": "BLOCK",
  "reason": "High risk pattern",
  "complaint_analysis": {
    "sentiment": "negative",
    "summary": "Unauthorized transaction"
  }
}
````

## Main Rule

Do not over-engineer.

The code must be:

* Human-understandable
* Manageable
* Debuggable
* Easy for a learning student to explain
* Suitable for an academic ML/NLP project

Prefer clear simple code over complex abstractions.

---

## Current Project Goal

The project is being built step by step using notebooks.

The important project layers are:

1. Fraud detection ML model
2. Threshold tuning
3. Decision logic
4. Final model training
5. Inference pipeline
6. FastAPI prediction endpoint
7. Basic NLP sentiment + summary
8. Deployment-ready structure

Do not jump ahead unless asked.

---

## Coding Style Rules

Follow these rules in every change:

1. Keep the code simple and readable.
2. Add comments only where they help understanding.
3. Do not add unnecessary classes, factories, decorators, or abstractions.
4. Do not silently change existing behavior.
5. Preserve existing notebook flow and section headings.
6. Keep variable names clear and descriptive.
7. Avoid clever one-liners if a simple version is easier to understand.
8. Do not introduce new libraries unless clearly needed.
9. Do not rewrite the whole project unless explicitly asked.
10. Make the smallest safe change that solves the problem.

---

## Notebook Rules

When editing notebooks:

1. Keep the notebook phase-based.

2. Each phase should have:

   * A clear markdown explanation
   * Simple code cells
   * Clear output
   * A short explanation of why the step matters

3. Do not generate a full notebook in one giant step unless explicitly requested.

4. Prefer building notebooks phase by phase.

5. Every important ML decision should be explained in markdown.

6. Do not hide important logic inside helper functions too early.

7. Helper functions are allowed only when they make the notebook easier to understand.

8. Keep outputs clean and interpretable.

---

## ML Project Rules

When working on the ML model:

1. Do not create data leakage.

2. Do not fit transformations on the test set.

3. Do not tune thresholds on the final test set.

4. Do not report production-refit performance as honest test performance.

5. Keep validation/test results separate from production artifacts.

6. Always explain whether a result comes from:

   * Cross-validation
   * OOF validation
   * Holdout/test set
   * Full-data production refit

7. If using a train/test split:

   * Use stratification for imbalanced fraud data.
   * Keep the test set untouched until final evaluation.

8. For fraud detection, recall is important, but precision and false positives must also be shown.

---

## Threshold and Decision Logic Rules

The project uses business decision logic.

Expected structure:

```python
if fraud_probability >= block_threshold:
    decision = "BLOCK"
elif fraud_probability >= review_threshold:
    decision = "REVIEW"
else:
    decision = "APPROVE"
```

Rules:

1. Do not hardcode thresholds in multiple places.
2. Load thresholds from the saved decision policy artifact where possible.
3. Keep `review_threshold` and `block_threshold` clearly separated.
4. Explain the difference between:

   * Evaluation threshold
   * Review threshold
   * Block threshold
5. Do not confuse validation threshold results with holdout/test results.
6. If threshold values change, update decision logic and saved artifacts consistently.

---

## Artifact Rules

When saving artifacts, use clear names.

Recommended artifacts:

```text
artifacts/
├── final_validated_fraud_model.joblib
├── final_feature_columns.json
├── final_model_metadata.json
├── final_model_metrics.json
├── final_decision_policy.json
```

Rules:

1. Save model artifacts only after validation.
2. Save feature columns in the exact order used for training.
3. Save metadata explaining:

   * Model type
   * Dataset used
   * Train/test split
   * Thresholds
   * Metrics
   * Creation date
4. Reload saved artifacts and test them after saving.
5. The inference notebook must load artifacts, not retrain the model.

---

## Inference Pipeline Rules

For `18_inference_pipeline.ipynb` or inference code:

1. Do not retrain the model.
2. Load the saved final model.
3. Load saved feature columns.
4. Load saved decision policy.
5. Validate incoming transaction input.
6. Ensure feature order matches training.
7. Return an API-ready response.

Expected response format:

```python
{
    "fraud_probability": 0.91,
    "decision": "BLOCK",
    "reason": "High risk pattern"
}
```

Input validation must check:

* All required features are present.
* No required feature is missing.
* Values are numeric.
* Feature order is correct.
* Extra fields are handled safely.

---

## FastAPI Rules

When building the API:

1. Keep the API simple.
2. Use one clear `/predict` endpoint first.
3. Load model artifacts once at startup.
4. Do not retrain inside the API.
5. Validate request input before prediction.
6. Return clear JSON responses.
7. Handle errors with readable messages.
8. Keep API code separate from notebook code.

Recommended structure:

```text
src/
├── models/
│   └── predict.py
├── pipelines/
│   └── inference_pipeline.py
└── api/
    └── main.py
```

---

## NLP Rules

The NLP part is basic and should not be overcomplicated.

Goal:

* Analyze complaint sentiment.
* Generate a short complaint summary.

Rules:

1. Keep NLP simple first.
2. Do not add complex LangChain, RAG, CrewAI, or agents unless asked.
3. First version can use a basic model or simple NLP pipeline.
4. NLP output should integrate with the final API response.

Expected structure:

```json
"complaint_analysis": {
  "sentiment": "negative",
  "summary": "Unauthorized transaction"
}
```

---

## File and Folder Rules

Respect the existing project structure.

Recommended structure:

```text
data/
├── raw/
├── interim/
├── processed/

notebooks/
├── 01_problem_understanding.ipynb
├── 02_data_loading_and_overview.ipynb
├── 03_data_validation.ipynb
├── 04_data_cleaning.ipynb
├── 05_univariate_analysis.ipynb
├── 06_bivariate_analysis.ipynb
├── 07_multivariate_analysis.ipynb
├── 08_feature_engineering.ipynb
├── 09_outlier_analysis.ipynb
├── 10_feature_selection.ipynb
├── 11_model_training.ipynb
├── 12_model_evaluation.ipynb
├── 13_threshold_tuning.ipynb
├── 14_decision_logic.ipynb
├── 15_final_model_training.ipynb
├── 16_inference_pipeline.ipynb

src/
├── data/
├── features/
├── models/
├── pipelines/
├── visualization/
├── utils/

artifacts/
reports/
tests/
scripts/
```

Do not move files unless asked.

---

## Testing and Validation Rules

Before finishing any task, check:

1. Does the code run?
2. Are imports correct?
3. Are paths correct?
4. Are saved artifacts created in the expected place?
5. Are metrics calculated correctly?
6. Is there any data leakage?
7. Is feature order preserved?
8. Does the final prediction output make sense?

If tests exist, run them.

If there are no tests, add only simple useful tests when asked.

---

## Explanation Rules

When making changes, also explain:

1. What was changed.
2. Why it was changed.
3. What problem it solves.
4. What the user should check next.

Use beginner-friendly explanations.

Avoid vague statements like:

```text
Improved the model.
```

Prefer:

```text
Updated threshold loading so the final model uses the same review/block policy saved from the decision logic notebook.
```

---

## Safety Rules for ML Results

Do not exaggerate model quality.

Always be honest about:

* Class imbalance
* False positives
* False negatives
* Recall limitations
* Difference between validation and test performance
* Difference between validated model and production refit model

Never claim the model is production-ready unless:

1. It has been validated.
2. Artifacts are saved.
3. Inference is tested.
4. API is tested.
5. Monitoring and deployment concerns are discussed.

---

## Production Refit Rule

A production refit model may be trained on the full labeled dataset only after final validation.

Important:

* The validated model is used for honest performance reporting.
* The production refit model is trained on all labeled data after evaluation.
* Do not use the production refit model to report final test metrics because it has seen all data.

Use a control flag:

```python
RUN_PRODUCTION_REFIT = False
```

Only run production refit when explicitly needed.

---

## Preferred Workflow

For each task:

1. Read the existing notebook or file first.
2. Understand the current flow.
3. Make a minimal safe change.
4. Preserve existing behavior unless the user asks to change it.
5. Add markdown explanation if working in a notebook.
6. Run or explain validation checks.
7. Summarize the result clearly.

---

## Things Not To Do

Do not:

* Rewrite the whole project unnecessarily.
* Add advanced tools before the basics are complete.
* Use the test set for threshold tuning.
* Retrain the model inside inference.
* Hardcode feature columns manually if an artifact exists.
* Hardcode thresholds in multiple files.
* Hide important learning logic in complex helper files.
* Add UI work unless explicitly requested.
* Add CrewAI, RAG, n8n, or LangChain unless asked.
* Change project structure without permission.

---

## Final Response Format

When completing a task, respond in this format:

```text
Status: Completed / Needs Review / Blocked

What changed:
- ...

Why:
- ...

Validation:
- ...

Files changed:
- ...

Next step:
- ...
```

Keep the response clear and direct.

```

This version is suitable for Codex because it tells Codex **how to behave**, **what not to break**, and **how to preserve your learning flow** while building the ML/NLP project step by step.
```
