# Credit-Card-Fraud-Detection

Credit card fraud detection project using the Kaggle credit card transactions dataset.

## Problem Statement

Detect whether a transaction is fraudulent (`Class = 1`) or legitimate (`Class = 0`) using anonymized PCA-based features, transaction time, and transaction amount.

## Dataset

- Source: `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`
- Raw file: `data/raw/creditcard.csv`
- Target column: `Class`
- Key columns: `Time`, `V1` to `V28`, `Amount`, `Class`

## Project Structure

```text
.
├── 01_eda.py
├── 02_random_forest_pipeline.py
├── 03_logistic_regression_pipeline.py
├── artifacts/
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
│       └── creditcard.csv
├── models/
├── notebooks/
│   ├── 01_problem_understanding.ipynb
│   ├── 02_data_loading_and_overview.ipynb
│   └── 03_data_validation.ipynb
├── reports/
├── scripts/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── data/
│   │   ├── data_cleaning.py
│   │   ├── data_loader.py
│   │   └── data_validation.py
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   ├── utils/
│   └── visualization/
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Data Setup

1. Download `creditcard.csv` from Kaggle.
2. Place it in `data/raw/creditcard.csv`.

Expected layout:

```text
project-root/
├── data/
│   └── raw/
│       └── creditcard.csv
├── notebooks/
├── src/
└── README.md
```

## Data Path Configuration

Core project settings live in [src/config/config.py](/Users/mmubashir/VCode/Credit-Card-Fraud-Detection/src/config/config.py).

Default paths are:

```python
DATA_PATH = "data/raw/creditcard.csv"
CLEANED_DATA_PATH = "data/interim/creditcard_cleaned.csv"
ENGINEERED_DATA_PATH = "data/processed/creditcard_engineered.csv"
SELECTED_DATA_PATH = "data/processed/creditcard_selected.csv"
TARGET_COLUMN = "Class"
```

Raw, cleaned, engineered, and selected dataset paths can be overridden with environment variables:

```env
RAW_DATA_PATH=data/raw/creditcard.csv
CLEANED_DATA_PATH=data/interim/creditcard_cleaned.csv
ENGINEERED_DATA_PATH=data/processed/creditcard_engineered.csv
SELECTED_DATA_PATH=data/processed/creditcard_selected.csv
```

## Load Data in Code

```python
from src.data.data_loader import load_raw_data

df = load_raw_data()
```

## Run the Project

Exploratory analysis:

```bash
python 01_eda.py
```

Random Forest pipeline:

```bash
python 02_random_forest_pipeline.py
```

Logistic Regression pipeline:

```bash
python 03_logistic_regression_pipeline.py
```

## Current Script Behavior

- `01_eda.py` loads the dataset with pandas and performs initial inspection, null checks, feature-type discovery, and exploratory metrics/plots.
- `02_random_forest_pipeline.py` trains a class-balanced Random Forest model, selects a threshold using F1, evaluates predictions, and saves the trained artifact.
- `03_logistic_regression_pipeline.py` applies numeric preprocessing, trains a class-balanced Logistic Regression model, selects the best threshold from the precision-recall curve, evaluates predictions, and saves the trained artifact.

## Notes

- The reusable loader in `src/data/data_loader.py` expects the raw dataset at `data/raw/creditcard.csv`.
- The current top-level scripts still read from `data/creditcard.csv`. If you plan to run those scripts as-is, update them to use the config or move/copy the dataset accordingly.

## Next Steps

- Standardize all scripts to use the config-based paths in `src/config/config.py`
- Expand data validation and cleaning for reproducible preprocessing
- Add feature engineering and model training modules under `src/`
- Add tests for loaders, validation, and training pipelines
- Track experiment outputs and evaluation reports in `artifacts/` and `reports/`
