# CLAUDE.md — Claude Project Instructions

## 1. Project Identity

This repository is an end-to-end **Credit Card Fraud Detection + Complaint NLP** project.

The final system should:

- Detect fraudulent credit card transactions using machine learning.
- Return a fraud probability.
- Apply clear business decision logic:
  - `BLOCK`
  - `REVIEW`
  - `APPROVE`
- Analyze customer complaint text using NLP.
- Expose the final workflow through a simple FastAPI API.
- Be understandable as both:
  - an academic MS AI / NLP project
  - a portfolio-level industry project

The project must remain **simple, explainable, and student-friendly**.

---

## 2. Main Rule

Do **not** over-engineer.

Prefer code that is:

- Human-understandable
- Easy to debug
- Easy to explain in viva / presentation
- Safe for ML experimentation
- Suitable for a real-world portfolio project

Use simple functions and clear flow before advanced abstractions.

Avoid unnecessary:

- Classes
- Factories
- Decorators
- Complex inheritance
- Hidden helper magic
- Extra dependencies

---

## 3. Claude Working Behavior

When Claude works on this project:

1. Read the existing file/notebook first.
2. Understand the current flow before changing anything.
3. Make the smallest safe change that solves the task.
4. Preserve existing behavior unless the user explicitly asks to change it.
5. Do not rewrite unrelated files.
6. Do not jump ahead to future phases.
7. Explain changes in beginner-friendly language.
8. Keep implementation practical and testable.
9. Prefer clarity over cleverness.
10. When unsure, choose the option that is easier for the user to understand and debug.

---

## 4. Current Project Flow

The project is being built step by step using notebooks and later reusable Python modules.

Follow this flow unless explicitly instructed otherwise:

1. Problem understanding
2. Data loading and overview
3. Data validation
4. Data cleaning
5. Exploratory data analysis
6. Feature engineering
7. Feature selection
8. Model training
9. Model evaluation
10. Threshold tuning
11. Decision logic
12. Final model training
13. Inference pipeline
14. Reusable inference helper module
15. FastAPI prediction endpoint
16. Basic NLP complaint analysis
17. API + NLP integration
18. Deployment preparation

Do not skip the learning flow by generating large unexplained code blocks.

---

## 5. Notebook Rules

When editing notebooks:

1. Keep the notebook phase-based.
2. Preserve existing headings and flow unless asked to reorganize.
3. Add one phase at a time when possible.
4. Each phase should include:
   - a clear markdown title
   - a short explanation of the goal
   - simple code cells
   - clean output
   - a short explanation of why the phase matters
5. Do not create a full notebook in one huge generation unless explicitly requested.
6. Do not hide important learning logic inside helper functions too early.
7. Use helper functions only when they make the notebook easier to understand.
8. Keep outputs readable and not overly noisy.
9. Explain every important ML decision in markdown.
10. Do not delete working cells unless required.
11. Do not silently change previous logic.

Good notebook style:

```python
# Load the saved feature list so inference uses the same column order as training.
with open(FEATURE_COLUMNS_PATH, "r") as file:
    feature_columns = json.load(file)
```

Bad notebook style:

```python
fc=json.load(open(p))
```

---

## 6. Code Commenting Rules

Code must be easy for the user to understand later.

### 6.1 Function Comments

Every important function must have a clear docstring.

The docstring should explain:

- what the function does
- what the inputs mean
- what the function returns
- when it raises an error, if relevant

Preferred format:

```python
def validate_transaction_input(transaction: dict, feature_columns: list[str]) -> dict:
    """
    Validate one incoming transaction before sending it to the model.

    Parameters
    ----------
    transaction:
        Dictionary containing feature names and numeric values.
    feature_columns:
        Ordered list of features used during model training.

    Returns
    -------
    dict
        Clean transaction dictionary containing only required model features.

    Raises
    ------
    ValueError
        If required features are missing or values are not numeric.
    """
```

### 6.2 Line Comments

Use line comments only where they add understanding.

Good line comments:

```python
# Keep the exact training column order before prediction.
model_input = model_input[feature_columns]
```

```python
# Convert model probability into a business decision.
decision_result = apply_decision_policy(fraud_probability, decision_policy)
```

Avoid obvious comments:

```python
# Import pandas
import pandas as pd

# Add 1 to x
x = x + 1
```

### 6.3 Comment Balance

Do not over-comment every line.

Comment when the code explains:

- ML safety
- feature order
- data leakage prevention
- threshold meaning
- artifact loading
- input validation
- business decision logic
- error handling

---

## 7. Python Coding Style Rules

Follow these rules in all Python files and notebooks:

1. Use clear variable names.
2. Keep functions small and focused.
3. Prefer simple `if/elif/else` logic over clever one-liners.
4. Avoid unnecessary global state.
5. Avoid hidden side effects.
6. Validate inputs before using them.
7. Use readable error messages.
8. Avoid broad `except Exception` unless re-raising with context.
9. Do not add new libraries unless clearly needed.
10. Do not hardcode values that already exist in artifacts.
11. Keep paths centralized near the top of the file/notebook.
12. Use `pathlib.Path` for file paths when practical.
13. Keep outputs API-ready where relevant.

---

## 8. ML Safety Rules

When working on ML code:

1. Do not create data leakage.
2. Do not fit preprocessing on the test set.
3. Do not tune thresholds on the final test set.
4. Use stratified splitting for imbalanced fraud data.
5. Keep validation metrics separate from final holdout/test metrics.
6. Keep production refit separate from honest evaluation.
7. Never report production-refit metrics as final test metrics.
8. Always preserve feature order between training and inference.
9. Always explain where metrics come from:
   - cross-validation
   - OOF validation
   - holdout/test set
   - production refit
10. Do not exaggerate model quality.

Fraud detection is imbalanced, so always consider:

- precision
- recall
- F1-score
- false positives
- false negatives
- confusion matrix
- fraud capture rate

Recall matters, but false positives still matter because blocking/reviewing too many legitimate transactions affects customers.

---

## 9. Threshold and Decision Logic Rules

The project uses business decision logic.

Expected rule:

```python
if fraud_probability >= block_threshold:
    decision = "BLOCK"
elif fraud_probability >= review_threshold:
    decision = "REVIEW"
else:
    decision = "APPROVE"
```

Rules:

1. Do not hardcode thresholds in many places.
2. Load thresholds from the saved decision policy artifact when available.
3. Keep these concepts separate:
   - evaluation threshold
   - review threshold
   - block threshold
4. If thresholds change, sync:
   - threshold tuning notebook
   - decision logic notebook
   - final model training notebook
   - saved decision policy artifact
   - inference code
5. Do not compare validation threshold metrics with holdout/test metrics as if they are the same.
6. Explain why `REVIEW` and `BLOCK` are separate decisions.

---

## 10. Artifact Rules

Use saved artifacts as the source of truth for inference.

Recommended artifact names:

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
2. Save feature columns in exact training order.
3. Save decision policy in one place.
4. Save metadata explaining:
   - model type
   - dataset used
   - train/test split
   - selected features
   - thresholds
   - metrics
   - creation date
5. Reload artifacts after saving to confirm they work.
6. Inference must load artifacts and must not retrain.
7. If an artifact is missing, raise a clear error explaining which file is missing.

---

## 11. Final Model Training Rules

For final model training:

1. Use the selected features only.
2. Use the selected model only.
3. Use the selected decision policy only.
4. Train the validated model on the training split.
5. Evaluate honestly on the holdout/test set.
6. Save validated model artifacts.
7. Reload artifacts and test one prediction.
8. Keep optional production refit controlled with:

```python
RUN_PRODUCTION_REFIT = False
```

Important explanation to preserve:

```text
The validated model is used for honest performance reporting.
The production refit model is trained on all labeled data after final validation.
The production refit model should not be used to report final test performance because it has seen all data.
```

---

## 12. Inference Pipeline Rules

For `18_inference_pipeline.ipynb` and reusable inference code:

1. Do not retrain the model.
2. Do not tune thresholds again.
3. Do not change selected features.
4. Load saved model artifact.
5. Load saved feature columns artifact.
6. Load saved decision policy artifact.
7. Validate transaction input before prediction.
8. Preserve exact feature order.
9. Return API-ready output.
10. Keep the inference flow simple and explainable.

Expected output:

```python
{
    "fraud_probability": 0.91,
    "decision": "BLOCK",
    "risk_level": "HIGH",
    "reason": "Fraud probability is greater than or equal to the block threshold."
}
```

---

## 13. Reusable Inference Helper Module Rules

Preferred module path:

```text
src/inference/fraud_inference.py
```

If `src/inference/` does not exist, create:

```text
src/inference/__init__.py
src/inference/fraud_inference.py
```

The reusable module should contain these functions:

1. `load_artifacts()`
2. `validate_transaction_input()`
3. `prepare_model_input()`
4. `apply_decision_policy()`
5. `predict_fraud()`
6. `predict_batch()`

### 13.1 `load_artifacts()`

Purpose:

- Load saved model, feature columns, and decision policy.
- Fail early with a clear message if any required artifact is missing.

Must include:

- function docstring
- clear artifact paths
- readable error messages
- no model training

### 13.2 `validate_transaction_input()`

Purpose:

- Validate one transaction dictionary before prediction.

Must check:

- all required features are present
- missing features are reported clearly
- values are numeric
- unsupported values are rejected clearly
- extra fields are ignored safely or reported depending on current project decision

Must not:

- reorder features silently without explanation
- fill missing critical model features with random/default values unless explicitly requested

### 13.3 `prepare_model_input()`

Purpose:

- Convert a validated transaction into model-ready input.

Must ensure:

- `pandas.DataFrame` shape is correct
- columns match training feature order exactly
- numeric values are converted safely

Important comment to include where relevant:

```python
# The model expects the same column order used during training.
model_input = model_input[feature_columns]
```

### 13.4 `apply_decision_policy()`

Purpose:

- Convert fraud probability into `BLOCK`, `REVIEW`, or `APPROVE`.

Must:

- use `review_threshold` and `block_threshold`
- return decision, risk level, and reason
- validate that thresholds exist
- validate that `review_threshold <= block_threshold`

Expected logic:

```python
if probability >= block_threshold:
    decision = "BLOCK"
    risk_level = "HIGH"
elif probability >= review_threshold:
    decision = "REVIEW"
    risk_level = "MEDIUM"
else:
    decision = "APPROVE"
    risk_level = "LOW"
```

### 13.5 `predict_fraud()`

Purpose:

- Run full single-transaction inference.

Flow:

1. Validate input.
2. Prepare model input.
3. Predict fraud probability.
4. Apply decision policy.
5. Return API-ready response.

Must not:

- train the model
- change thresholds
- change feature columns

### 13.6 `predict_batch()`

Purpose:

- Run inference for multiple transactions.

Must:

- validate every transaction
- preserve result order
- return a list of prediction responses
- include clear error handling if one row is invalid

Keep batch logic simple.

---

## 14. FastAPI Rules

When building FastAPI:

1. Keep the API simple.
2. Start with one `/predict` endpoint.
3. Load artifacts once at startup.
4. Do not retrain inside the API.
5. Validate request input before prediction.
6. Return clean JSON.
7. Use readable error messages.
8. Keep API code separate from notebook code.
9. Reuse `src/inference/fraud_inference.py` instead of duplicating inference logic.

Recommended first API structure:

```text
src/api/main.py
src/inference/fraud_inference.py
```

---

## 15. NLP Rules

The NLP part should stay basic first.

Goal:

- detect complaint sentiment
- create a short complaint summary

Rules:

1. Do not add LangChain, RAG, CrewAI, n8n, or agents unless explicitly requested.
2. Start with simple NLP logic that is easy to explain.
3. Keep NLP output small and API-ready.
4. Do not mix NLP code into fraud model code too early.
5. Integrate NLP only after fraud inference is stable.

Expected output:

```json
{
  "complaint_analysis": {
    "sentiment": "negative",
    "summary": "Unauthorized transaction"
  }
}
```

---

## 16. File and Folder Rules

Respect the existing project structure.

Do not move files unless explicitly asked.

Preferred structure:

```text
Credit-Card-Fraud-Detection/
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── inference/
│   ├── models/
│   ├── pipelines/
│   ├── visualization/
│   └── utils/
├── artifacts/
├── reports/
├── tests/
└── scripts/
```

Do not create duplicate folders with similar meanings.

---

## 17. Testing and Validation Rules

Before finalizing a change, check:

1. Does the notebook/code run?
2. Are imports correct?
3. Are paths correct?
4. Are artifacts loaded from the expected place?
5. Is feature order preserved?
6. Is there any data leakage?
7. Are metrics calculated correctly?
8. Does prediction output make sense?
9. Did the change avoid unrelated behavior changes?
10. Is the explanation beginner-friendly?

If tests exist, run them.

If no tests exist, only add simple useful tests when asked.

For inference helper module, useful checks include:

```python
# Example smoke test idea: load artifacts and predict one transaction.
artifacts = load_artifacts()
result = predict_fraud(sample_transaction, artifacts)
print(result)
```

---

## 18. Response Format

After completing a task, respond using this structure:

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

Keep the response direct and easy to understand.

Mention anything that could not be tested.

---

## 19. Review Mode Instructions

When asked to review code or notebook changes:

1. Check requirement coverage first.
2. Identify bugs or missing requirements.
3. Separate critical issues from small improvements.
4. Do not rewrite everything unless necessary.
5. Give exact file/section/function names where possible.
6. Suggest minimal safe fixes.
7. Confirm whether existing behavior was preserved.

Preferred review response:

```text
Status: Changes Required / Approved / Approved with Notes

Requirement Coverage:
- ...

Issues Found:
- ...

Suggested Fix:
- ...

Final Verdict:
- ...
```

---

## 20. Things Claude Must Not Do

Do not:

- Over-engineer.
- Rewrite the whole project without permission.
- Change unrelated files.
- Change selected features unless asked.
- Retrain inside inference.
- Tune thresholds inside inference.
- Tune thresholds on the final test set.
- Report production-refit metrics as final test metrics.
- Hardcode thresholds in multiple places.
- Hardcode feature columns if a saved artifact exists.
- Add advanced AI agents before the basic ML/NLP/API system is complete.
- Hide important learning logic in complex helper files.
- Move project files without asking.
- Remove useful markdown explanations from notebooks.
- Delete working code without explaining why.

---

## 21. Start Prompt for Claude

When using Claude, start with:

```text
Read CLAUDE.md first and follow it strictly before making any changes.
Make the smallest safe change only.
Preserve existing behavior unless I explicitly ask to change it.
Use clear function docstrings and useful line comments where needed.
```

---

## 22. Final Principle

This project is not only about getting code to run.

It is about building an ML/NLP system that the user can understand, explain, debug, and improve step by step.

Clarity is more important than cleverness.
