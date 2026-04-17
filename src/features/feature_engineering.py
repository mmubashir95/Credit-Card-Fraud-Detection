from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.config import AMOUNT_COLUMN


STABLE_INTERACTION_FEATURES: dict[str, tuple[str, str]] = {
    "V14_V12_interaction": ("V14", "V12"),
    "V17_V16_interaction": ("V17", "V16"),
}

EXPERIMENTAL_INTERACTION_FEATURES: dict[str, tuple[str, str]] = {
    "V17_V12_interaction": ("V17", "V12"),
    "V17_V14_interaction": ("V17", "V14"),
    "V17_V10_interaction": ("V17", "V10"),
    "amount_V17_interaction": (AMOUNT_COLUMN, "V17"),
}

EXPERIMENTAL_RATIO_FEATURES = (
    "amount_to_mean_ratio",
    "amount_to_median_ratio",
)


class FeatureEngineeringError(ValueError):
    """Raised when required feature inputs are missing."""


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise FeatureEngineeringError(
            f"Missing required columns for feature engineering: {missing}"
        )


def add_log_amount(
    df: pd.DataFrame,
    amount_column: str = AMOUNT_COLUMN,
    output_column: str = "log_amount",
) -> pd.DataFrame:
    """Add a log-transformed amount column using log1p for numerical stability."""
    _validate_required_columns(df, [amount_column])

    result = df.copy()
    result[output_column] = np.log1p(pd.to_numeric(result[amount_column], errors="coerce"))
    return result


def add_interaction_features(
    df: pd.DataFrame,
    include_experimental: bool = False,
) -> pd.DataFrame:
    """Add multiplicative interaction features used during EDA and selection."""
    feature_map = dict(STABLE_INTERACTION_FEATURES)
    if include_experimental:
        feature_map.update(EXPERIMENTAL_INTERACTION_FEATURES)

    required_columns = sorted({column for pair in feature_map.values() for column in pair})
    _validate_required_columns(df, required_columns)

    result = df.copy()
    for feature_name, (left, right) in feature_map.items():
        result[feature_name] = result[left] * result[right]
    return result


def add_amount_ratio_features(
    df: pd.DataFrame,
    amount_mean: float,
    amount_median: float,
    amount_column: str = AMOUNT_COLUMN,
) -> pd.DataFrame:
    """Add exploratory amount ratio features using supplied reference statistics."""
    _validate_required_columns(df, [amount_column])

    if amount_mean == 0 or amount_median == 0:
        raise FeatureEngineeringError(
            "Amount mean and median must be non-zero for ratio features."
        )

    result = df.copy()
    result["amount_to_mean_ratio"] = result[amount_column] / amount_mean
    result["amount_to_median_ratio"] = result[amount_column] / amount_median
    return result


def engineer_features(
    df: pd.DataFrame,
    *,
    include_experimental: bool = False,
    amount_reference_stats: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build reusable engineered features.

    Default behavior creates the stable features retained for the final project:
    `log_amount`, `V14_V12_interaction`, and `V17_V16_interaction`.

    Experimental mode preserves the broader notebook-era feature set so the same
    engineering logic can still support comparisons and future experiments.
    """
    engineered = add_log_amount(df)
    engineered = add_interaction_features(
        engineered,
        include_experimental=include_experimental,
    )

    if include_experimental:
        if amount_reference_stats is None:
            raise FeatureEngineeringError(
                "amount_reference_stats is required when include_experimental=True. "
                "Provide {'mean': ..., 'median': ...} to build ratio features."
            )

        missing_stats = sorted(
            set(("mean", "median")) - set(amount_reference_stats.keys())
        )
        if missing_stats:
            raise FeatureEngineeringError(
                f"Missing amount reference stats: {missing_stats}"
            )

        engineered = add_amount_ratio_features(
            engineered,
            amount_mean=float(amount_reference_stats["mean"]),
            amount_median=float(amount_reference_stats["median"]),
        )

    return engineered
