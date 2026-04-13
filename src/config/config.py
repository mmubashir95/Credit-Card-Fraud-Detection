from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = "data/raw/creditcard.csv"
CLEANED_DATA_PATH = "data/interim/creditcard_cleaned.csv"
ENGINEERED_DATA_PATH = "data/processed/creditcard_engineered.csv"
SELECTED_DATA_PATH = "data/processed/creditcard_selected.csv"

TARGET_COLUMN = "Class"
MODEL_TARGET_COLUMN = "target"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
NUMERICAL_COLUMNS = [TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN]
CATEGORICAL_COLUMNS: list[str] = []

RAW_DATA_FILE = PROJECT_ROOT / DATA_PATH
CLEANED_DATA_FILE = PROJECT_ROOT / CLEANED_DATA_PATH
ENGINEERED_DATA_FILE = PROJECT_ROOT / ENGINEERED_DATA_PATH
SELECTED_DATA_FILE = PROJECT_ROOT / SELECTED_DATA_PATH

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FEATURE_COLUMNS_FILE = ARTIFACTS_DIR / "feature_columns.json"
LOGREG_MODEL_FILE = ARTIFACTS_DIR / "fraud_logreg_pipeline.joblib"
RANDOM_FOREST_MODEL_FILE = ARTIFACTS_DIR / "fraud_random_forest_pipeline.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_TABLES_DIR = REPORTS_DIR / "tables"
