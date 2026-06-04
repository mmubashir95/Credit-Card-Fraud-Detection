# AGENTS.md — Codex Instructions

## Project Context

This is an end-to-end AI project for **Credit Card Fraud Detection and Complaint Analysis**.

The final system should:

- Detect fraudulent credit card transactions using machine learning.
- Return a fraud probability.
- Apply business decision logic:
  - `BLOCK`
  - `REVIEW`
  - `APPROVE`
- Analyze customer complaints using NLP.
- Expose the final system through a FastAPI API.
- Stay understandable as both an academic project and a portfolio-level industry project.

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
```

---

## Main Rule

**Do not over-engineer.**

The code must be:

- Human-understandable
- Manageable
- Debuggable
- Easy for a learning student to explain
- Suitable for an academic ML/NLP project
- Clean enough for a portfolio project

Prefer clear, simple code over clever abstractions.

---

## Current Project Goal

The project is being built step by step using notebooks first, then reusable Python files.

The important project layers are:

1. Fraud detection ML model
2. Threshold tuning
3. Decision logic
4. Final model training
5. Inference pipeline
6. Reusable inference helper module
7. FastAPI prediction endpoint
8. Basic NLP sentiment + summary
9. Deployment-ready structure

Do not jump ahead unless the task explicitly asks for it.

---

## Coding Style Rules

Follow these rules in every change:

1. Keep the code simple and readable.
2. Use clear variable names.
3. Avoid clever one-liners when a simple multi-line version is easier to understand.
4. Do not introduce unnecessary classes, factories, decorators, or abstractions.
5. Do not introduce new libraries unless clearly needed.
6. Do not silently change existing behavior.
7. Preserve existing notebook flow and section headings.
8. Do not rewrite the whole project unless explicitly asked.
9. Make the smallest safe change that solves the task.
10. Prefer explicit validation errors over silent failures.
11. Keep functions focused on one responsibility.
12. Do not duplicate the same logic in multiple notebooks/files when a reusable helper already exists.

---

## Code Comments and Documentation Rules

Comments are required, but they must be useful.

### Function-Level Comments / Docstrings

Every reusable function must have a short docstring that explains:

1. What the function does.
2. Important inputs.
3. What it returns.
4. Any important validation or side effect.

Use this style:

```python
def validate_transaction_input(transaction, feature_columns):
    """
    Validate one transaction before sending it to the fraud model.

    Parameters
    ----------
    transaction : dict
        Incoming transaction values keyed by feature name.
    feature_columns : list[str]
        Feature names expected by the model, in training order.

    Returns
    -------
    dict
        Clean transaction dictionary containing only model features.

    Raises
    ------
    ValueError
        If required features are missing or values are not numeric.
    """
```

### Line Comments

Use line comments only when they explain why something is done or prevent confusion.

Good line comments:

```python
# Keep the saved training order so the model receives the same column order during inference.
model_input = model_input[feature_columns]
```

```python
# Convert numpy float to plain Python float so the response is JSON serializable.
fraud_probability = float(model.predict_proba(model_input)[0, 1])
```

Avoid useless comments:

```python
# Import pandas
import pandas as pd

# Return result
return result
```

### Comment Balance

Do not comment every line.

Use comments for:

- Feature order logic
- Threshold/decision policy logic
- Data leakage prevention
- Artifact loading/saving
- Input validation
- JSON/API response formatting
- Any step that may confuse a beginner later

Avoid comments for obvious Python syntax.

---

## Reusable Function Rules

When creating or editing reusable Python functions:

1. Each function must do one clear job.
2. Each function must have a docstring.
3. Keep function names descriptive.
4. Validate inputs near the start of the function.
5. Return predictable output types.
6. Raise clear errors with beginner-friendly messages.
7. Avoid hidden global state unless loading shared artifacts intentionally.
8. Do not print inside reusable functions unless the function is specifically for notebook/demo output.
9. Keep reusable functions API-ready.
10. Add useful line comments for non-obvious steps.

Preferred function pattern:

```python
def function_name(input_value):
    """
    Explain what the function does, expected input, output, and important errors.
    """
    if input_value is None:
        raise ValueError("input_value is required.")

    # Explain the reason for any non-obvious transformation.
    cleaned_value = ...

    return cleaned_value
```

---

## Notebook Rules

When editing notebooks:

1. Keep the notebook phase-based.
2. Each phase should have:
   - A clear markdown title
   - A beginner-friendly explanation
   - Simple code cells
   - Clear output
   - A short explanation of why the step matters
3. Do not generate a full notebook in one giant step unless explicitly requested.
4. Prefer building notebooks phase by phase.
5. Every important ML decision should be explained in markdown.
6. Do not hide important learning logic inside helper functions too early.
7. Helper functions are allowed only when they make the notebook easier to understand.
8. Keep outputs clean and interpretable.
9. Do not retrain the model inside inference notebooks.
10. Do not tune thresholds again inside inference notebooks.

---

## ML Project Rules

When working on the ML model:

1. Do not create data leakage.
2. Do not fit transformations on the test set.
3. Do not tune thresholds on the final test set.
4. Do not report production-refit performance as honest test performance.
5. Keep validation/test results separate from production artifacts.
6. Always explain whether a result comes from:
   - Cross-validation
   - OOF validation
   - Holdout/test set
   - Full-data production refit
7. If using a train/test split:
   - Use stratification for imbalanced fraud data.
   - Keep the test set untouched until final evaluation.
8. For fraud detection, recall is important, but precision and false positives must also be shown.
9. Do not change selected features unless explicitly asked.
10. Do not change thresholds unless explicitly asked.
11. Do not compare validation metrics and holdout metrics as if they came from the same dataset.

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
   - Evaluation threshold
   - Review threshold
   - Block threshold
5. Do not confuse validation threshold results with holdout/test results.
6. If threshold values change, update decision logic and saved artifacts consistently.
7. Use `>=` consistently for threshold comparison.
8. Validate that `review_threshold <= block_threshold`.

---

## Artifact Rules

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
   - Model type
   - Dataset used
   - Train/test split
   - Thresholds
   - Metrics
   - Creation date
4. Reload saved artifacts and test them after saving.
5. The inference notebook must load artifacts, not retrain the model.
6. If an artifact is missing, stop with a clear error message.
7. Do not manually recreate feature columns if `final_feature_columns.json` exists.
8. Do not manually recreate thresholds if `final_decision_policy.json` exists.

---

## Inference Pipeline Rules

For `18_inference_pipeline.ipynb` and reusable inference code:

1. Do not retrain the model.
2. Do not tune thresholds again.
3. Do not change selected features.
4. Load the saved final model.
5. Load saved feature columns.
6. Load saved decision policy.
7. Validate incoming transaction input.
8. Ensure feature order matches training.
9. Return an API-ready response.
10. Keep the code easy to explain.

Expected response format:

```python
{
    "fraud_probability": 0.91,
    "decision": "BLOCK",
    "risk_level": "HIGH",
    "reason": "Fraud probability is above the block threshold."
}
```

Input validation must check:

- All required features are present.
- No required feature is missing.
- Values are numeric.
- Feature order is preserved.
- Extra fields are handled safely.

---

## Required Inference Helper Module Rules

When creating `src/inference/fraud_inference.py`, keep it simple and reusable.

Recommended functions:

1. `load_artifacts()`
2. `validate_transaction_input()`
3. `prepare_model_input()`
4. `apply_decision_policy()`
5. `predict_fraud()`
6. `predict_batch()`

### `load_artifacts()`

Purpose:

- Load the trained model, feature columns, metadata, and decision policy.
- Validate that all required artifact files exist.

Rules:

- Do not retrain the model.
- Use clear artifact paths.
- Raise `FileNotFoundError` if any artifact is missing.
- Return artifacts in a simple dictionary.
- Add comments explaining why feature columns and policy are loaded with the model.

Expected style:

```python
def load_artifacts(artifact_dir="artifacts"):
    """
    Load saved model artifacts needed for fraud inference.

    Parameters
    ----------
    artifact_dir : str
        Directory containing the saved model, feature columns, metadata, and decision policy.

    Returns
    -------
    dict
        Dictionary containing model, feature columns, decision policy, and optional metadata.

    Raises
    ------
    FileNotFoundError
        If any required artifact file is missing.
    """
```

### `validate_transaction_input()`

Purpose:

- Check one incoming transaction before prediction.

Rules:

- Check missing required features.
- Check values are numeric.
- Ignore or safely handle extra fields.
- Raise `ValueError` with a clear message if input is invalid.
- Return cleaned input containing only required model features.

### `prepare_model_input()`

Purpose:

- Convert validated transaction data into a model-ready DataFrame.

Rules:

- Preserve exact feature order from `final_feature_columns.json`.
- Create a one-row pandas DataFrame for single prediction.
- Add a line comment explaining why feature order is critical.

### `apply_decision_policy()`

Purpose:

- Convert fraud probability into business decision output.

Rules:

- Use saved `review_threshold` and `block_threshold`.
- Use the project decision order: BLOCK, REVIEW, APPROVE.
- Return decision, risk level, and reason.
- Validate that required threshold keys exist.
- Validate that `review_threshold <= block_threshold`.

### `predict_fraud()`

Purpose:

- Run the full single-transaction inference flow.

Rules:

- Validate input.
- Prepare model input.
- Predict fraud probability.
- Apply decision policy.
- Return API-ready dictionary.
- Convert numpy values to plain Python types.
- Do not print inside this function.

### `predict_batch()`

Purpose:

- Run predictions for multiple transactions.

Rules:

- Accept a list of transaction dictionaries or a pandas DataFrame.
- Validate each transaction.
- Preserve feature order.
- Return a list of API-ready prediction dictionaries.
- Keep implementation simple.
- Do not use advanced batching abstractions unless needed.

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
9. Reuse `src/inference/fraud_inference.py` instead of duplicating inference logic.

Recommended structure:

```text
src/
├── inference/
│   ├── __init__.py
│   └── fraud_inference.py
├── api/
│   └── main.py
└── nlp/
    └── complaint_analysis.py
```

---

## NLP Rules

The NLP part should stay basic first.

Goal:

- Analyze complaint sentiment.
- Generate a short complaint summary.

Rules:

1. Keep NLP simple first.
2. Do not add complex LangChain, RAG, CrewAI, or agents unless asked.
3. First version can use a basic model or simple NLP pipeline.
4. NLP output should integrate with the final API response.
5. Keep NLP functions documented with docstrings.
6. Do not mix NLP logic directly inside fraud model functions.

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

Do not move files unless asked.

Preferred project structure for the current stage:

```text
notebooks/
├── 15_threshold_tuning.ipynb
├── 16_decision_logic.ipynb
├── 17_final_model_training.ipynb
├── 18_inference_pipeline.ipynb

src/
├── inference/
│   ├── __init__.py
│   └── fraud_inference.py
├── api/
├── nlp/
├── utils/

artifacts/
├── final_validated_fraud_model.joblib
├── final_feature_columns.json
├── final_model_metadata.json
├── final_model_metrics.json
├── final_decision_policy.json

reports/
├── figures/
├── tables/

tests/
```

---

## Testing and Validation Rules

Before finishing any task, check:

1. Does the code run?
2. Are imports correct?
3. Are paths correct?
4. Are saved artifacts created in the expected place?
5. Are required artifacts loaded correctly?
6. Are metrics calculated correctly?
7. Is there any data leakage?
8. Is feature order preserved?
9. Does the prediction output make sense?
10. Does the response use plain Python/JSON-serializable values?
11. Does the notebook still run top-to-bottom?
12. Did the change avoid retraining when working on inference?

If tests exist, run them.

If there are no tests, add only simple useful tests when asked.

Suggested checks for inference helper files:

```bash
python -m py_compile src/inference/fraud_inference.py
```

If a project test suite exists:

```bash
pytest
```

---

## Explanation Rules

When making changes, also explain:

1. What was changed.
2. Why it was changed.
3. What problem it solves.
4. How to test it.
5. What the next safe step is.

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

- Class imbalance
- False positives
- False negatives
- Recall limitations
- Difference between validation and test performance
- Difference between validated model and production refit model

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

- The validated model is used for honest performance reporting.
- The production refit model is trained on all labeled data after evaluation.
- Do not use the production refit model to report final test metrics because it has seen all data.

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
3. Identify the smallest safe change.
4. Preserve existing behavior unless the task asks to change it.
5. Add markdown explanation if working in a notebook.
6. Add docstrings for reusable functions.
7. Add useful line comments where logic may confuse a beginner.
8. Run or explain validation checks.
9. Summarize the result clearly.

---

## Things Not To Do

Do not:

- Rewrite the whole project unnecessarily.
- Add advanced tools before the basics are complete.
- Use the test set for threshold tuning.
- Retrain the model inside inference.
- Hardcode feature columns manually if an artifact exists.
- Hardcode thresholds in multiple files.
- Hide important learning logic in complex helper files.
- Add UI work unless explicitly requested.
- Add CrewAI, RAG, n8n, or LangChain unless asked.
- Change project structure without permission.
- Remove existing working behavior while fixing a small issue.
- Add comments that only repeat what the code already says.

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
