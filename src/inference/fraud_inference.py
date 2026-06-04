"""Reusable helpers for fraud model inference.

This module centralizes the logic for loading saved artifacts, validating
transaction input, preparing model-ready features, applying the business
decision policy, and predicting both single transactions and batches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = DEFAULT_PROJECT_ROOT / "artifacts"


def load_artifacts(
    project_root: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load the saved model and supporting inference artifacts.

    Parameters:
        project_root: Optional project root used to resolve the default artifacts folder.
        artifacts_dir: Optional explicit artifacts directory override.

    Returns:
        A dictionary containing the loaded model, metadata, thresholds, and paths.

    Raises:
        FileNotFoundError: If any required saved artifact file is missing.
    """
    # Defaults keep notebook and API usage simple, while overrides support other runtimes later.
    resolved_project_root = Path(project_root).resolve() if project_root else DEFAULT_PROJECT_ROOT
    resolved_artifacts_dir = (
        Path(artifacts_dir).resolve() if artifacts_dir else resolved_project_root / "artifacts"
    )

    artifact_paths = {
        "final_validated_model": resolved_artifacts_dir / "final_validated_fraud_model.joblib",
        "final_feature_columns": resolved_artifacts_dir / "final_feature_columns.json",
        "final_decision_policy": resolved_artifacts_dir / "final_decision_policy.json",
        "final_model_metadata": resolved_artifacts_dir / "final_model_metadata.json",
        "final_model_metrics": resolved_artifacts_dir / "final_model_metrics.json",
    }

    for artifact_name, artifact_path in artifact_paths.items():
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing required artifact: {artifact_path}")

    with open(artifact_paths["final_feature_columns"], "r", encoding="utf-8") as file_handle:
        feature_columns = json.load(file_handle)

    with open(artifact_paths["final_decision_policy"], "r", encoding="utf-8") as file_handle:
        decision_policy = json.load(file_handle)

    with open(artifact_paths["final_model_metadata"], "r", encoding="utf-8") as file_handle:
        model_metadata = json.load(file_handle)

    with open(artifact_paths["final_model_metrics"], "r", encoding="utf-8") as file_handle:
        model_metrics = json.load(file_handle)

    return {
        "project_root": resolved_project_root,
        "artifacts_dir": resolved_artifacts_dir,
        "artifact_paths": artifact_paths,
        "model": joblib.load(artifact_paths["final_validated_model"]),
        "feature_columns": feature_columns,
        "decision_policy": decision_policy,
        "model_metadata": model_metadata,
        "model_metrics": model_metrics,
    }


def validate_transaction_input(transaction: dict[str, Any] | pd.Series, feature_columns: list[str]) -> bool:
    """Validate one transaction against the saved feature schema.

    Parameters:
        transaction: One transaction as a dict or pandas Series.
        feature_columns: Required feature names in saved training order.

    Returns:
        True when the transaction is valid for inference.

    Raises:
        ValueError: If the input is empty, has missing features, or contains non-numeric values.
    """
    if isinstance(transaction, pd.Series):
        transaction_data = transaction.to_dict()
    elif isinstance(transaction, dict):
        transaction_data = transaction
    else:
        raise ValueError(
            "Transaction input validation failed: transaction must be a dict or pandas Series."
        )

    if not transaction_data:
        raise ValueError(
            "Transaction input validation failed: transaction input cannot be empty."
        )

    missing_features = [
        feature_name for feature_name in feature_columns if feature_name not in transaction_data
    ]
    if missing_features:
        raise ValueError(
            "Transaction input validation failed: missing required features: "
            f"{missing_features}"
        )

    non_numeric_features = []
    for feature_name in feature_columns:
        feature_value = transaction_data[feature_name]
        if isinstance(feature_value, bool) or not isinstance(
            feature_value, (int, float, np.integer, np.floating)
        ):
            non_numeric_features.append(
                f"{feature_name}={feature_value!r} ({type(feature_value).__name__})"
            )

    if non_numeric_features:
        raise ValueError(
            "Transaction input validation failed: required features must be numeric. "
            f"Found invalid values: {non_numeric_features}"
        )

    extra_columns = [
        column_name for column_name in transaction_data if column_name not in feature_columns
    ]
    if extra_columns:
        # Extra fields can appear in API payloads, so we ignore them instead of passing them to the model.
        print(f"Warning: ignoring extra input columns: {extra_columns}")

    return True


def prepare_model_input(
    transaction: dict[str, Any] | pd.Series, feature_columns: list[str]
) -> pd.DataFrame:
    """Convert one validated transaction into a one-row model input DataFrame.

    Parameters:
        transaction: One transaction as a dict or pandas Series.
        feature_columns: Required feature names in the exact saved training order.

    Returns:
        A one-row pandas DataFrame ready for model prediction.

    Raises:
        ValueError: If validation fails or the prepared DataFrame does not match the expected schema.
    """
    validate_transaction_input(transaction, feature_columns)

    if isinstance(transaction, pd.Series):
        transaction_data = transaction.to_dict()
    else:
        transaction_data = dict(transaction)

    filtered_transaction = {
        key: value for key, value in transaction_data.items() if key in feature_columns
    }

    model_input_df = pd.DataFrame([filtered_transaction])
    # The saved feature order must be reused exactly so inference matches training-time column order.
    model_input_df = model_input_df.reindex(columns=feature_columns)
    model_input_df = model_input_df.apply(pd.to_numeric, errors="raise")

    if model_input_df.columns.tolist() != feature_columns:
        raise ValueError(
            "Model input preparation failed: DataFrame columns do not match the saved feature order."
        )

    expected_shape = (1, len(feature_columns))
    if model_input_df.shape != expected_shape:
        raise ValueError(
            "Model input preparation failed: prepared DataFrame has the wrong shape. "
            f"Expected {expected_shape}, found {model_input_df.shape}."
        )

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in model_input_df.dtypes):
        raise ValueError(
            "Model input preparation failed: prepared DataFrame contains non-numeric values."
        )

    return model_input_df


def apply_decision_policy(probability: float, policy: dict[str, Any]) -> dict[str, str]:
    """Convert a fraud probability into a business decision.

    Parameters:
        probability: Fraud probability predicted by the saved model.
        policy: Decision policy containing review and block thresholds.

    Returns:
        A dictionary with decision, risk level, and reason.
    """
    review_threshold = policy["review_threshold"]
    block_threshold = policy["block_threshold"]

    if probability >= block_threshold:
        return {
            "decision": "BLOCK",
            "risk_level": "HIGH",
            "reason": "Transaction probability is above block threshold.",
        }

    if probability >= review_threshold:
        return {
            "decision": "REVIEW",
            "risk_level": "MEDIUM",
            "reason": "Transaction probability is between review and block thresholds.",
        }

    return {
        "decision": "APPROVE",
        "risk_level": "LOW",
        "reason": "Transaction probability is below review threshold.",
    }


def predict_fraud(
    transaction: dict[str, Any] | pd.Series,
    model: Any | None = None,
    feature_columns: list[str] | None = None,
    decision_policy: dict[str, Any] | None = None,
    model_metadata: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Predict fraud risk for one transaction and return an API-ready response.

    Parameters:
        transaction: One transaction as a dict or pandas Series.
        model: Optional already-loaded model to avoid reloading artifacts repeatedly.
        feature_columns: Optional saved feature list in model input order.
        decision_policy: Optional saved review/block threshold policy.
        model_metadata: Optional saved metadata used for model version reporting.
        project_root: Optional project root used when artifacts must be loaded.
        artifacts_dir: Optional explicit artifacts directory override.

    Returns:
        A dictionary containing fraud probability, business decision, thresholds, and model version.

    Raises:
        ValueError: If the transaction input is invalid for inference.
    """
    if model is None or feature_columns is None or decision_policy is None:
        artifacts = load_artifacts(project_root=project_root, artifacts_dir=artifacts_dir)
        model = artifacts["model"] if model is None else model
        feature_columns = artifacts["feature_columns"] if feature_columns is None else feature_columns
        decision_policy = artifacts["decision_policy"] if decision_policy is None else decision_policy
        model_metadata = artifacts["model_metadata"] if model_metadata is None else model_metadata

    model_input_df = prepare_model_input(transaction, feature_columns)
    # Column 1 of predict_proba is the positive fraud class probability used by downstream decisions.
    fraud_probability = float(model.predict_proba(model_input_df)[0][1])

    # Business thresholds are applied after scoring so model probability and policy stay separate.
    decision_result = apply_decision_policy(fraud_probability, decision_policy)
    model_metadata = model_metadata or {}
    model_version = model_metadata.get("model_version") or model_metadata.get("version")

    return {
        "fraud_probability": fraud_probability,
        "decision": decision_result["decision"],
        "risk_level": decision_result["risk_level"],
        "reason": decision_result["reason"],
        "thresholds": {
            "review_threshold": decision_policy["review_threshold"],
            "block_threshold": decision_policy["block_threshold"],
        },
        "model_version": model_version if model_version is not None else "Not available",
    }


def predict_batch(
    transactions_df: pd.DataFrame,
    model: Any | None = None,
    feature_columns: list[str] | None = None,
    decision_policy: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Predict fraud risk for a batch of transactions.

    Parameters:
        transactions_df: Batch input as a pandas DataFrame.
        model: Optional already-loaded model to avoid reloading artifacts repeatedly.
        feature_columns: Optional saved feature list in model input order.
        decision_policy: Optional saved review/block threshold policy.
        project_root: Optional project root used when artifacts must be loaded.
        artifacts_dir: Optional explicit artifacts directory override.

    Returns:
        A DataFrame with row identifiers, fraud probability, decision, risk level, and reason.

    Raises:
        ValueError: If the batch is empty, not a DataFrame, or is missing required feature columns.
    """
    if not isinstance(transactions_df, pd.DataFrame):
        raise ValueError("Batch prediction failed: input must be a pandas DataFrame.")

    if transactions_df.empty:
        raise ValueError("Batch prediction failed: input DataFrame cannot be empty.")

    if model is None or feature_columns is None or decision_policy is None:
        artifacts = load_artifacts(project_root=project_root, artifacts_dir=artifacts_dir)
        model = artifacts["model"] if model is None else model
        feature_columns = artifacts["feature_columns"] if feature_columns is None else feature_columns
        decision_policy = artifacts["decision_policy"] if decision_policy is None else decision_policy

    sample_id_series = None
    if "sample_id" in transactions_df.columns:
        sample_id_series = transactions_df["sample_id"].copy()

    # Batch-level schema checks fail fast and clearly before value-level validation looks at each row.
    missing_features = [
        column_name for column_name in feature_columns if column_name not in transactions_df.columns
    ]
    if missing_features:
        raise ValueError(
            f"Batch prediction failed: missing required feature columns: {missing_features}"
        )

    for _, row in transactions_df.iterrows():
        validate_transaction_input(row, feature_columns)

    model_input_df = transactions_df.loc[:, feature_columns].copy()
    # Reordering with saved feature_columns protects the model from accidental column shuffling.
    model_input_df = model_input_df.reindex(columns=feature_columns)
    model_input_df = model_input_df.apply(pd.to_numeric, errors="raise")

    # Column 1 gives the fraud-class probability for every row in the batch.
    fraud_probabilities = model.predict_proba(model_input_df)[:, 1]

    prediction_rows = []
    for row_number, probability in enumerate(fraud_probabilities):
        decision_result = apply_decision_policy(float(probability), decision_policy)
        prediction_row = {
            "fraud_probability": float(probability),
            "decision": decision_result["decision"],
            "risk_level": decision_result["risk_level"],
            "reason": decision_result["reason"],
        }

        if sample_id_series is not None:
            prediction_row["sample_id"] = sample_id_series.iloc[row_number]
        else:
            prediction_row["original_index"] = transactions_df.index[row_number]

        prediction_rows.append(prediction_row)

    predictions_df = pd.DataFrame(prediction_rows)
    id_column = "sample_id" if sample_id_series is not None else "original_index"
    ordered_columns = [id_column, "fraud_probability", "decision", "risk_level", "reason"]
    return predictions_df[ordered_columns]
