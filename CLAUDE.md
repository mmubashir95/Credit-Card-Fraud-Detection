Create this file in the **project root**:

```bash
cd /Users/mohammadmubashir/VCode/Credit-Card-Fraud-Detection
touch CLAUDE.md
code CLAUDE.md
```

Paste this inside `CLAUDE.md`:

````md
# CLAUDE.md — Project Instructions

## Project Context

This is a Credit Card Fraud Detection + NLP project.

The final goal is to build an end-to-end AI system that:

- Detects fraudulent transactions using machine learning.
- Returns fraud probability.
- Applies decision logic: BLOCK / REVIEW / APPROVE.
- Analyzes complaint text using NLP.
- Provides a FastAPI endpoint for prediction.
- Can be explained clearly as an academic and portfolio project.

The project must stay simple, understandable, and learning-friendly.

---

## Main Working Rule

Do not over-engineer.

Code should be:

- Human-understandable
- Easy to debug
- Easy to explain
- Suitable for a student learning ML/NLP
- Suitable for a real-world portfolio project

Prefer clear step-by-step code over complex abstractions.

---

## Important Project Flow

The project is built notebook by notebook.

Follow this flow unless I explicitly ask otherwise:

1. Data understanding
2. Data validation
3. Data cleaning
4. EDA
5. Feature engineering
6. Feature selection
7. Model training
8. Model evaluation
9. Threshold tuning
10. Decision logic
11. Final model training
12. Inference pipeline
13. FastAPI
14. NLP complaint analysis
15. Deployment

Do not jump ahead.

---

## Notebook Rules

When working on notebooks:

1. Keep the notebook phase-based.
2. Each phase should have:
   - Clear markdown explanation
   - Simple code
   - Clear output
   - Short explanation of why the step matters

3. Do not create a full notebook in one huge generation unless I ask.
4. Build notebooks step by step.
5. Preserve existing notebook structure and headings.
6. Do not delete existing working cells unless required.
7. Do not silently change previous logic.
8. Explain every important ML decision in markdown.
9. Keep code readable for learning.

---

## ML Rules

When working on model training, evaluation, or threshold tuning:

1. Avoid data leakage.
2. Do not fit preprocessing on the test set.
3. Do not tune thresholds on the final test set.
4. Use stratified split for imbalanced fraud data.
5. Keep validation results separate from final test results.
6. Clearly explain where metrics come from:
   - Cross-validation
   - OOF validation
   - Holdout/test set
   - Production refit

7. For fraud detection, recall is important, but always show:
   - Precision
   - Recall
   - F1-score
   - False positives
   - False negatives
   - Confusion matrix when useful

8. Do not exaggerate results.

---

## Threshold and Decision Logic Rules

The project uses this business decision flow:

```python
if fraud_probability >= block_threshold:
    decision = "BLOCK"
elif fraud_probability >= review_threshold:
    decision = "REVIEW"
else:
    decision = "APPROVE"
````

Rules:

1. Do not hardcode thresholds in many places.

2. Load thresholds from the saved decision policy artifact when available.

3. Keep these concepts separate:

   * Evaluation threshold
   * Review threshold
   * Block threshold

4. If thresholds are updated, sync:

   * Threshold tuning notebook
   * Decision logic notebook
   * Final model training notebook
   * Saved decision policy artifact

5. Do not compare validation threshold results with final holdout results as if they are the same.

---

## Final Model Training Rules

For the final model training notebook:

1. Train the validated model on the selected training split.
2. Evaluate honestly on the holdout/test set.
3. Save model artifacts.
4. Reload artifacts and test prediction.
5. Keep optional production refit controlled with:

```python
RUN_PRODUCTION_REFIT = False
```

Important explanation:

```text
The validated model is used for honest performance reporting.
The production refit model is trained on all labeled data after evaluation and is intended for deployment.
Do not use the production refit model to report final test performance because it has seen all data.
```

---

## Artifact Rules

Save final artifacts clearly.

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

1. Save feature columns in exact training order.
2. Save model metadata.
3. Save final metrics.
4. Save decision policy.
5. Reload artifacts after saving to confirm they work.
6. Inference pipeline must load artifacts, not retrain.

---

## Inference Pipeline Rules

For the inference notebook or inference code:

1. Do not retrain the model.
2. Load saved model.
3. Load saved feature columns.
4. Load saved decision policy.
5. Validate incoming transaction input.
6. Ensure feature order matches training.
7. Return API-ready output.

Expected output:

```python
{
    "fraud_probability": 0.91,
    "decision": "BLOCK",
    "reason": "High risk pattern"
}
```

Input validation should check:

* Required features are present.
* No important feature is missing.
* Values are numeric.
* Feature order is correct.
* Extra fields are handled safely.

---

## FastAPI Rules

When building FastAPI:

1. Keep the API simple.
2. Start with one `/predict` endpoint.
3. Load artifacts once at startup.
4. Do not retrain inside the API.
5. Validate input before prediction.
6. Return clean JSON.
7. Use readable error messages.
8. Keep API logic separate from notebook logic.

---

## NLP Rules

The NLP part should be basic first.

Goal:

* Detect complaint sentiment.
* Generate a short complaint summary.

Rules:

1. Do not add advanced LangChain, RAG, CrewAI, or agents unless I ask.
2. Start with simple sentiment and summary logic.
3. Keep output easy to explain.
4. Integrate NLP output with the final API response.

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

## Coding Style Rules

1. Make minimal safe changes.
2. Do not rewrite unrelated files.
3. Do not change existing behavior unless asked.
4. Use clear variable names.
5. Avoid clever one-line code.
6. Add comments only where useful.
7. Do not add unnecessary dependencies.
8. Prefer simple functions over complex classes.
9. Keep paths clear and consistent.
10. Use project artifacts instead of hardcoded values where possible.

---

## File Structure Rules

Respect the existing project structure.

Do not move files unless I ask.

Recommended structure:

```text
data/
├── raw/
├── interim/
├── processed/

notebooks/

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

---

## Review Rules

Before finalizing work, check:

1. Does the notebook/code run?
2. Are imports correct?
3. Are paths correct?
4. Are artifacts saved correctly?
5. Is feature order preserved?
6. Is there any data leakage?
7. Are metrics calculated correctly?
8. Is the final output understandable?
9. Did you avoid changing unrelated behavior?

---

## Response Format

When responding after a task, use this format:

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

Keep the response direct and beginner-friendly.

---

## Things Not To Do

Do not:

* Over-engineer.
* Rewrite the full project without permission.
* Tune thresholds on the test set.
* Retrain inside inference.
* Report production refit metrics as final test metrics.
* Hardcode thresholds in multiple places.
* Hardcode feature order if an artifact exists.
* Add advanced AI agents before the basic ML/NLP/API system is complete.
* Hide important learning logic in complex helper files.
* Change project structure without asking.

````

Then, whenever you use Claude, start with:

```text
Read CLAUDE.md first and follow it strictly before making any changes.
````

Best setup in your project root:

```text
Credit-Card-Fraud-Detection/
├── AGENTS.md      # Codex instructions
├── CLAUDE.md      # Claude instructions
├── notebooks/
├── src/
├── data/
├── artifacts/
└── reports/
```
