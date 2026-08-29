"""
Configuration file for the Customer Churn Prediction project.
Defines paths and constants used across all scripts.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# Base Directories
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# Data Paths
# ──────────────────────────────────────────────────────────────────────────

# Find the raw CSV file automatically


def _find_raw_csv() -> Path:
    """Locate the raw customer churn CSV file."""
    # Look for .csv files in a subdirectory
    csv_dir = BASE_DIR / "customer_churn_dataset-testing-master.csv"
    if csv_dir.is_dir():
        csv_files = list(csv_dir.glob("*.csv"))
        if csv_files:
            return csv_files[0]

    # Fallback: look for any CSV in BASE_DIR (except in output)
    candidates = sorted(
        path for path in BASE_DIR.rglob("*.csv")
        if path.is_file() and OUTPUT_DIR not in path.parents
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No CSV file found in {BASE_DIR}")


DATA_RAW = str(_find_raw_csv())
DATA_CLEANED = str(OUTPUT_DIR / "customer_churn_cleaned.csv")
DATA_AUDIT = str(OUTPUT_DIR / "customer_churn_cleaning_audit.csv")
REPORT_PDF = str(OUTPUT_DIR / "customer_churn_data_cleaning_report.pdf")
ML_OUTPUT_DIR = OUTPUT_DIR / "machine_learning"

# Machine learning model artifacts
DECISION_TREE_MODEL = str(ML_OUTPUT_DIR / "decision_tree_model.joblib")
RANDOM_FOREST_MODEL = str(ML_OUTPUT_DIR / "random_forest_model.joblib")
DECISION_TREE_METRICS = str(ML_OUTPUT_DIR / "decision_tree_metrics.csv")
RANDOM_FOREST_METRICS = str(ML_OUTPUT_DIR / "random_forest_metrics.csv")
MODEL_COMPARISON_CSV = str(OUTPUT_DIR / "model_comparison.csv")
MODEL_COMPARISON_METRICS_PNG = str(OUTPUT_DIR / "model_comparison_metrics.png")
MODEL_F1_SCORE_PNG = str(OUTPUT_DIR / "model_f1_score_comparison.png")

# SQL analysis artifacts
SQL_QUERY_SCRIPT = str(BASE_DIR / "06_sql_queries.py")
SQL_QUERY_REPORT_PDF = str(OUTPUT_DIR / "sql_queries_report.pdf")

DECISION_TREE_MAX_DEPTH = 8
DECISION_TREE_MIN_SAMPLES_LEAF = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────────────────

# Target variable for prediction
TARGET = "Churn"

# Column to drop (not a feature)
COLUMNS_TO_DROP = ["CustomerID"]

# Columns for outlier capping (1st and 99th percentile method)
OUTLIER_CAP_COLUMNS = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction"
]

# Categorical mapping
GENDER_MAPPING = {"Male": 0, "Female": 1}

# Columns to one-hot encode
CATEGORICAL_ENCODING = {
    "Subscription Type": True,  # drop_first=True
    "Contract Length": True,     # drop_first=True
}

# ──────────────────────────────────────────────────────────────────────────
# Data Quality Thresholds
# ──────────────────────────────────────────────────────────────────────────

EXPECTED_RANGES = {
    "CustomerID": (1, None),
    "Age": (18, 100),
    "Tenure": (0, None),
    "Usage Frequency": (0, None),
    "Support Calls": (0, None),
    "Payment Delay": (0, None),
    "Total Spend": (0, None),
    "Last Interaction": (0, None),
    "Churn": (0, 1),
}

EXPECTED_CATEGORIES = {
    "Gender": {"Female", "Male"},
    "Subscription Type": {"Basic", "Standard", "Premium"},
    "Contract Length": {"Monthly", "Quarterly", "Annual"},
}

# ──────────────────────────────────────────────────────────────────────────
# Display & Logging
# ──────────────────────────────────────────────────────────────────────────

PANDAS_DISPLAY_MAX_ROWS = 100
PANDAS_DISPLAY_MAX_COLS = 50


if __name__ == "__main__":
    """Verify configuration on script execution."""
    print("Configuration Summary:")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  DATA_RAW: {DATA_RAW}")
    print(f"  DATA_CLEANED: {DATA_CLEANED}")
    print(f"  TARGET: {TARGET}")
    print(f"\n✅ Configuration loaded successfully.")
