from __future__ import annotations

import pandas as pd

from src.config import (
    AMOUNT_COLUMN,
    NUMERICAL_COLUMNS,
    PCA_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)

REQUIRED_COLUMNS = [TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN, TARGET_COLUMN]


class DataValidationError(ValueError):
    """Raised when dataset validation fails."""


def validate_non_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataValidationError("Dataset is empty.")


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def validate_target_values(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise DataValidationError(f"Target column '{TARGET_COLUMN}' is missing.")

    allowed = {0, 1}
    actual = set(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna().astype(int).unique())
    invalid = sorted(actual - allowed)
    if invalid:
        raise DataValidationError(
            f"Invalid values in '{TARGET_COLUMN}': {invalid}"
        )


def validate_no_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise DataValidationError(f"Missing values found: {missing.to_dict()}")


def validate_numeric_columns(df: pd.DataFrame) -> None:
    missing = [column for column in NUMERICAL_COLUMNS if column not in df.columns]
    if missing:
        raise DataValidationError(f"Missing numeric columns for validation: {missing}")

    invalid_columns: dict[str, int] = {}
    for column in NUMERICAL_COLUMNS:
        coerced = pd.to_numeric(df[column], errors="coerce")
        invalid_count = int(coerced.isna().sum())
        if invalid_count > 0:
            invalid_columns[column] = invalid_count

    if invalid_columns:
        raise DataValidationError(
            f"Non-numeric values found in numeric columns: {invalid_columns}"
        )


def validate_non_negative_columns(df: pd.DataFrame) -> None:
    checks = [TIME_COLUMN, AMOUNT_COLUMN]
    negatives: dict[str, int] = {}

    for column in checks:
        series = pd.to_numeric(df[column], errors="coerce")
        negative_count = int((series < 0).sum())
        if negative_count > 0:
            negatives[column] = negative_count

    if negatives:
        raise DataValidationError(f"Negative values found: {negatives}")


def validate_raw_data(df: pd.DataFrame) -> None:
    """Run validation checks for raw fraud data."""
    validate_non_empty(df)
    validate_required_columns(df)
    validate_target_values(df)
    validate_numeric_columns(df)
    validate_non_negative_columns(df)


def validate_cleaned_data(df: pd.DataFrame) -> None:
    """Run validation checks for cleaned fraud data."""
    validate_non_empty(df)
    validate_required_columns(df)
    validate_target_values(df)
    validate_numeric_columns(df)
    validate_no_missing_values(df)
    validate_non_negative_columns(df)

