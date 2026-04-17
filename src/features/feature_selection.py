from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, PROJECT_ROOT, TARGET_COLUMN


DEFAULT_DECISIONS_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "10_feature_selection" / "feature_selection_decisions.csv"
)
DEFAULT_SELECTED_FEATURES_PATH = ARTIFACTS_DIR / "selected_feature_names.json"
DEFAULT_SELECTED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "final_features.csv"


class FeatureSelectionError(ValueError):
    """Raised when finalized feature-selection assets are invalid or incomplete."""


def load_feature_selection_decisions(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the finalized feature-selection decision table from disk."""
    csv_path = Path(path) if path is not None else DEFAULT_DECISIONS_PATH
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Feature-selection decisions not found at: {csv_path}")

    decisions = pd.read_csv(csv_path)
    if decisions.empty:
        raise FeatureSelectionError("Feature-selection decisions file is empty.")

    required_columns = {"feature", "selection_decision"}
    missing = sorted(required_columns - set(decisions.columns))
    if missing:
        raise FeatureSelectionError(
            f"Feature-selection decisions file is missing columns: {missing}"
        )

    return decisions


def get_selected_feature_names(
    decisions: pd.DataFrame | None = None,
    *,
    decision_label: str = "KEEP",
) -> list[str]:
    """Return the ordered list of finalized selected feature names."""
    decisions_df = decisions if decisions is not None else load_feature_selection_decisions()

    selected_features = decisions_df.loc[
        decisions_df["selection_decision"] == decision_label,
        "feature",
    ].tolist()

    if not selected_features:
        raise FeatureSelectionError(
            f"No features found with selection_decision == '{decision_label}'."
        )

    return selected_features


def build_selected_dataset(
    df: pd.DataFrame,
    *,
    selected_features: list[str] | None = None,
    include_target: bool = True,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Build the final modeling dataset from a broader engineered dataframe.

    By default, this uses the finalized `KEEP` features from the project
    feature-selection decisions and retains the target column for modeling.
    """
    features = (
        selected_features
        if selected_features is not None
        else get_selected_feature_names()
    )

    missing_features = sorted(set(features) - set(df.columns))
    if missing_features:
        raise FeatureSelectionError(
            f"Selected features are missing from the input dataframe: {missing_features}"
        )

    columns = list(features)
    if include_target:
        if target_column not in df.columns:
            raise FeatureSelectionError(
                f"Target column '{target_column}' is missing from the input dataframe."
            )
        columns.append(target_column)

    selected_df = df.loc[:, columns].copy()
    if selected_df.empty:
        raise FeatureSelectionError("Selected dataset is empty after column filtering.")

    return selected_df


def save_selected_feature_names(
    feature_names: list[str],
    path: str | Path | None = None,
) -> Path:
    """Persist the finalized feature-name list as a JSON artifact."""
    output_path = Path(path) if path is not None else DEFAULT_SELECTED_FEATURES_PATH
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    return output_path


def save_selected_dataset(
    df: pd.DataFrame,
    path: str | Path | None = None,
) -> Path:
    """Persist the final selected modeling dataset to disk."""
    output_path = Path(path) if path is not None else DEFAULT_SELECTED_DATASET_PATH
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
