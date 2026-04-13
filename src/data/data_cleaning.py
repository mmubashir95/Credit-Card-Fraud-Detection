from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CLEANED_DATA_FILE, PROJECT_ROOT


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated transaction rows while keeping the first occurrence."""
    return df.drop_duplicates().reset_index(drop=True)


def clean_creditcard_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply reusable cleaning steps for the fraud detection dataset."""
    df_clean = df.copy()
    df_clean = remove_duplicate_rows(df_clean)
    return df_clean


def save_cleaned_data(
    df: pd.DataFrame,
    path: str | Path | None = None,
) -> Path:
    """Persist the cleaned dataset to the interim data folder."""
    output_path = Path(path) if path is not None else CLEANED_DATA_FILE
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
