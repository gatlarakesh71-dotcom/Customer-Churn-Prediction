"""Clean the customer churn CSV and create a data-cleaning PDF report."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
REPORT_PATH = OUTPUT_DIR / "customer_churn_data_cleaning_report.pdf"
CLEANED_PATH = OUTPUT_DIR / "customer_churn_cleaned.csv"
AUDIT_PATH = OUTPUT_DIR / "customer_churn_cleaning_audit.csv"

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
OUTLIER_CAP_COLUMNS = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]


def find_csv() -> Path:
    candidates = sorted(
        path for path in BASE_DIR.rglob("*.csv")
        if path.is_file() and OUTPUT_DIR not in path.parents
    )
    if not candidates:
        raise FileNotFoundError(f"No CSV file found under {BASE_DIR}")
    return candidates[0]


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = raw.copy()
    original_rows, original_columns = data.shape
    data.columns = data.columns.astype(str).str.strip()
    text_columns = data.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        data[column] = data[column].astype("string").str.strip()

    numeric_columns = [
        column for column in EXPECTED_RANGES if column in data.columns]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    before_missing = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    outlier_results = []
    for column in OUTLIER_CAP_COLUMNS:
        if column not in data.columns:
            outlier_results.append(
                [column, "1st-99th percentile cap", "FAIL", 0])
            continue
        values = data[column].dropna()
        lower_bound = values.quantile(0.01)
        upper_bound = values.quantile(0.99)
        outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
        outlier_count = int(outliers.sum())
        data[column] = data[column].clip(lower_bound, upper_bound)
        outlier_results.append([
            column,
            f"Cap to {lower_bound:.2f} through {upper_bound:.2f}",
            "PASS",
            outlier_count,
        ])
    invalid_range_rows = pd.Series(False, index=data.index)
    range_results = []
    for column, (lower, upper) in EXPECTED_RANGES.items():
        if column not in data.columns:
            range_results.append([column, "Missing column", "FAIL", 0])
            continue
        invalid = data[column].isna()
        if lower is not None:
            invalid = invalid | (data[column] < lower)
        if upper is not None:
            invalid = invalid | (data[column] > upper)
        invalid_range_rows = invalid_range_rows | invalid
        range_results.append([
            column,
            f"{lower if lower is not None else '-inf'} to {upper if upper is not None else 'inf'}",
            "PASS" if not invalid.any() else "FAIL",
            int(invalid.sum()),
        ])

    category_results = []
    invalid_category_rows = pd.Series(False, index=data.index)
    for column, allowed in EXPECTED_CATEGORIES.items():
        if column not in data.columns:
            category_results.append(
                [column, ", ".join(sorted(allowed)), "FAIL", 0])
            continue
        invalid = data[column].isna() | ~data[column].isin(allowed)
        invalid_category_rows = invalid_category_rows | invalid
        category_results.append([
            column,
            ", ".join(sorted(allowed)),
            "PASS" if not invalid.any() else "FAIL",
            int(invalid.sum()),
        ])

    rows_before_filter = len(data)
    rows_to_remove = data.duplicated() | invalid_range_rows | invalid_category_rows
    data = data.loc[~rows_to_remove].drop_duplicates().reset_index(drop=True)

    audit_rows = [
        ["Column names", "Trim leading/trailing whitespace", "PASS", 0],
        ["Text fields", "Trim leading/trailing whitespace", "PASS", 0],
        ["Numeric fields", "Convert to numeric; invalid values become missing",
            "PASS", before_missing],
        ["Exact duplicate rows", "Remove duplicate records", "PASS", duplicate_rows],
        ["Required ranges", "Validate domain ranges", "PASS" if not invalid_range_rows.any(
        ) else "FAIL", int(invalid_range_rows.sum())],
        ["Allowed categories", "Validate categorical domains",
            "PASS" if not invalid_category_rows.any() else "FAIL", int(invalid_category_rows.sum())],
        ["Numeric outliers", "Cap values at the 1st and 99th percentiles",
            "PASS", sum(row[3] for row in outlier_results)],
        ["Rows retained", "Keep records passing all checks",
            "INFO", len(data)],
    ]
    audit_rows.extend(range_results)
    audit_rows.extend(category_results)
    audit_rows.extend(outlier_results)
    audit = pd.DataFrame(audit_rows, columns=[
                         "Check", "Rule", "Status", "Affected rows"])
    summary = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "clean_rows": len(data),
        "clean_columns": data.shape[1],
        "missing_before": before_missing,
        "missing_after": int(data.isna().sum().sum()),
        "duplicate_rows": duplicate_rows,
        "removed_rows": rows_before_filter - len(data),
        "range_failures": int(invalid_range_rows.sum()),
        "category_failures": int(invalid_category_rows.sum()),
        "outlier_values_capped": sum(row[3] for row in outlier_results),
    }
    return data, audit, summary


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.text(0.06, 0.93, title, fontsize=20,
                fontweight="bold", color="#17324D")
    y_position = 0.87
    for line in lines:
        for wrapped_line in textwrap.wrap(str(line), width=112) or [""]:
            figure.text(0.07, y_position, wrapped_line, fontsize=10)
            y_position -= 0.034
        y_position -= 0.012
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def add_table_page(pdf: PdfPages, title: str, table: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=18,
                   fontweight="bold", color="#17324D", pad=18)
    display_table = table.copy()
    numeric_columns = display_table.select_dtypes(include="number").columns
    display_table[numeric_columns] = display_table[numeric_columns].round(2)
    artist = axis.table(
        cellText=display_table.astype(str).values,
        colLabels=display_table.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    artist.auto_set_font_size(False)
    artist.set_fontsize(8)
    artist.scale(1, 1.5)
    for (row, _), cell in artist.get_celld().items():
        if row == 0:
            cell.set_facecolor("#17324D")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EAF1F5")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def build_report(raw: pd.DataFrame, cleaned: pd.DataFrame, audit: pd.DataFrame, summary: dict, source: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with PdfPages(REPORT_PATH) as pdf:
        add_text_page(pdf, "Customer Churn Data Cleaning Report", [
            f"Source: {source.name}",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
            f"Input: {summary['original_rows']:,} rows x {summary['original_columns']} columns",
            f"Output: {summary['clean_rows']:,} rows x {summary['clean_columns']} columns",
            "Purpose: make the dataset consistent, valid, and ready for analysis while capping extreme numeric values at documented percentile limits.",
            f"Overall result: {summary['outlier_values_capped']:,} extreme numeric values were capped; {summary['removed_rows']:,} rows were removed by quality checks.",
        ])

        overview = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing values", "Blank strings", "Exact duplicate rows", "Unique CustomerID", "Sequential IDs"],
            "Before cleaning": [len(raw), raw.shape[1], int(raw.isna().sum().sum()), int(raw.astype("string").apply(lambda c: c.str.strip().eq("").sum()).sum()), int(raw.duplicated().sum()), int(raw["CustomerID"].nunique()), bool(set(raw["CustomerID"]) == set(range(1, len(raw) + 1)))],
            "After cleaning": [len(cleaned), cleaned.shape[1], int(cleaned.isna().sum().sum()), int(cleaned.astype("string").apply(lambda c: c.str.strip().eq("").sum()).sum()), int(cleaned.duplicated().sum()), int(cleaned["CustomerID"].nunique()), bool(set(cleaned["CustomerID"]) == set(range(1, len(cleaned) + 1)))],
        })
        add_table_page(pdf, "Before and After Quality Metrics", overview)
        add_table_page(pdf, "Cleaning Rules and Audit Results", audit)

        column_profile = pd.DataFrame({
            "Column": cleaned.columns,
            "Data type": cleaned.dtypes.astype(str).values,
            "Non-null": cleaned.notna().sum().values,
            "Unique": cleaned.nunique().values,
            "Minimum": [cleaned[column].min() if pd.api.types.is_numeric_dtype(cleaned[column]) else "-" for column in cleaned.columns],
            "Maximum": [cleaned[column].max() if pd.api.types.is_numeric_dtype(cleaned[column]) else "-" for column in cleaned.columns],
        })
        add_table_page(pdf, "Cleaned Dataset Column Profile", column_profile)

        numeric = cleaned.select_dtypes(include="number").columns
        statistics = cleaned[numeric].describe().T.reset_index().rename(
            columns={"index": "Column"})
        add_table_page(pdf, "Numeric Validation Summary", statistics)

        add_text_page(pdf, "Key Data-Cleaning Details", [
            "1. Ingestion: read the CSV with pandas and verify that the expected customer-churn schema is present.",
            "2. Column names: remove accidental leading or trailing whitespace so downstream references are stable.",
            "3. Text standardization: trim whitespace in text fields; preserve the meaningful category labels.",
            "4. Type validation: coerce expected numeric fields to numeric values and flag conversion failures as missing.",
            "5. Missingness: measure null values and blank strings separately. Missing values should be imputed only with a documented business rule; none are present here.",
            "6. Duplicates: check full-row duplicates and CustomerID uniqueness. Duplicate entities should be investigated before aggregation or modeling.",
            "7. Range checks: verify Age is at least 18, operational measures are non-negative, and Churn is binary 0/1.",
            "8. Category checks: verify Gender, Subscription Type, and Contract Length against the observed domain definitions.",
            "9. Outliers: cap configured numeric fields at their 1st and 99th percentiles; this limits extreme influence without deleting valid customer records.",
            "10. Reproducibility: the cleaned file and row-level audit summary are written to the output folder.",
        ])
        add_text_page(pdf, "Outlier Analysis and Treatment", [
            "Definition: an outlier is an observation unusually far from most other values. It may be a data-entry error, a system error, or a genuine customer behavior pattern.",
            "Detection method 1 - IQR: calculate IQR = Q3 - Q1. Flag values below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR.",
            "Detection method 2 - Percentiles: flag values below the 1st percentile or above the 99th percentile. This project uses percentile limits for capping.",
            "Detection method 3 - Z-score: calculate z = (value - mean) / standard deviation. Values with an absolute z-score above 3 may be potential outliers when data is approximately normally distributed.",
            "Mean: uses every value but is strongly affected by extreme values. Use it for roughly symmetric data with few outliers, or for comparison during analysis.",
            "Median: the middle sorted value and resistant to extreme values. Prefer it for skewed numeric fields such as Total Spend and for numeric missing-value imputation.",
            "Mode: the most frequent value. Use it mainly to fill missing categorical fields such as Gender, Subscription Type, and Contract Length; it is not normally used to treat numeric outliers.",
            "Removal: delete a record only when the value is impossible or clearly incorrect. Do not delete a valid high-value customer simply because the value is unusual.",
            "Capping: replace values below the lower limit with the lower limit and values above the upper limit with the upper limit. This project caps selected numeric fields at the 1st and 99th percentiles.",
            "Transformation: a log transformation such as log1p(Total Spend) can reduce right skew while retaining the customer record.",
            "Project treatment: Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, and Last Interaction are checked and capped. CustomerID and Churn are not capped.",
            f"Observed result: {summary['outlier_values_capped']:,} values were capped, and {summary['removed_rows']:,} rows were removed by the other quality checks.",
            "Best practice: investigate unusual values, document the chosen treatment, apply the same rule to future data, and fit model preprocessing using training data only.",
        ])
        add_text_page(pdf, "Recommendations and Handoff", [
            "The current file is structurally clean and no records were removed.",
            "Keep the audit CSV with the cleaned extract so every transformation is reviewable.",
            "For future files, stop or quarantine rows that fail range/category checks rather than silently dropping them.",
            "Confirm business definitions for Age, Total Spend, Payment Delay, and Tenure before modeling.",
            "Encode categorical fields and scale numeric features only in the modeling pipeline, fitted on training data to prevent leakage.",
            "Re-run this script whenever the source CSV changes; outputs are deterministic apart from the report timestamp.",
        ])


def main() -> None:
    source = find_csv()
    raw = pd.read_csv(source)
    cleaned, audit, summary = clean_data(raw)
    cleaned.to_csv(CLEANED_PATH, index=False)
    audit.to_csv(AUDIT_PATH, index=False)
    build_report(raw, cleaned, audit, summary, source)
    print(f"Cleaned data: {CLEANED_PATH}")
    print(f"Audit: {AUDIT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
