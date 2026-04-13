from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import (
    CLEANED_DATA_FILE,
    ENGINEERED_DATA_FILE,
    PROJECT_ROOT,
    RAW_DATA_FILE,
    SELECTED_DATA_FILE,
)

DEFAULT_RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", str(RAW_DATA_FILE)))
DEFAULT_CLEANED_DATA_PATH = Path(
    os.getenv("CLEANED_DATA_PATH", str(CLEANED_DATA_FILE))
)
DEFAULT_ENGINEERED_DATA_PATH = Path(
    os.getenv("ENGINEERED_DATA_PATH", str(ENGINEERED_DATA_FILE))
)
DEFAULT_SELECTED_DATA_PATH = Path(
    os.getenv("SELECTED_DATA_PATH", str(SELECTED_DATA_FILE))
)


def _resolve_csv_path(path: str | Path | None, default_path: Path) -> Path:
    csv_path = Path(path) if path is not None else default_path
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    return csv_path


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} dataset not found at: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} dataset is empty: {path}")

    return df


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw fraud detection dataset from disk."""
    csv_path = _resolve_csv_path(path, DEFAULT_RAW_DATA_PATH)
    return _read_csv(csv_path, "Raw")


def load_cleaned_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned fraud detection dataset from disk."""
    csv_path = _resolve_csv_path(path, DEFAULT_CLEANED_DATA_PATH)
    return _read_csv(csv_path, "Cleaned")


def load_engineered_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the engineered fraud detection dataset from disk."""
    csv_path = _resolve_csv_path(path, DEFAULT_ENGINEERED_DATA_PATH)
    return _read_csv(csv_path, "Engineered")


def load_selected_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the selected fraud detection dataset from disk."""
    csv_path = _resolve_csv_path(path, DEFAULT_SELECTED_DATA_PATH)
    return _read_csv(csv_path, "Selected")

