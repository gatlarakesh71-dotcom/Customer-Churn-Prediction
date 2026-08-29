"""Explore the customer churn CSV and save a PDF data-quality report."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
REPORT_PATH = OUTPUT_DIR / "customer_churn_exploration_report.pdf"


def find_csv() -> Path:
    candidates = sorted(
        path
        for path in BASE_DIR.rglob("*.csv")
        if path.is_file() and OUTPUT_DIR not in path.parents
    )
    if not candidates:
        raise FileNotFoundError(f"No CSV file found under {BASE_DIR}")
    return candidates[0]


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
    table_artist = axis.table(
        cellText=display_table.astype(str).values,
        colLabels=display_table.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8)
    table_artist.scale(1, 1.5)
    for (row, _), cell in table_artist.get_celld().items():
        if row == 0:
            cell.set_facecolor("#17324D")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EAF1F5")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def add_bar_page(pdf: PdfPages, title: str, series: pd.Series, xlabel: str) -> None:
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    series.sort_values(ascending=False).plot.bar(ax=axis, color="#2A9D8F")
    axis.set_title(title, loc="left", fontsize=18,
                   fontweight="bold", color="#17324D")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def add_churn_rate_page(pdf: PdfPages, column: str, rates: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    rates.sort_values("Churn rate %").plot.barh(
        x=column, y="Churn rate %", ax=axis, legend=False, color="#E76F51"
    )
    axis.set_title(f"Churn Rate by {column}", loc="left", fontsize=18,
                   fontweight="bold", color="#17324D")
    axis.set_xlabel("Churn rate (%)")
    axis.set_ylabel(column)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def build_report(data: pd.DataFrame, csv_path: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    row_count, column_count = data.shape
    missing_by_column = data.isna().sum()
    blank_by_column = data.astype("string").apply(
        lambda column: column.str.strip().eq("").sum())
    duplicate_rows = int(data.duplicated().sum())
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns = data.select_dtypes(exclude="number").columns.tolist()
    unique_counts = data.nunique(dropna=False)
    churn_column = "Churn" if "Churn" in data.columns else None

    with PdfPages(REPORT_PATH) as pdf:
        add_text_page(pdf, "Customer Churn Data Exploration", [
            f"Source file: {csv_path.name}",
            f"Rows: {row_count:,} data rows",
            f"Columns: {column_count}",
            f"Duplicate rows: {duplicate_rows:,}",
            f"Numeric columns: {len(numeric_columns)} ({', '.join(numeric_columns) or 'None'})",
            f"Categorical columns: {len(categorical_columns)} ({', '.join(categorical_columns) or 'None'})",
            "Missing values means true null/NA values detected by pandas. Blank values means empty or whitespace-only strings.",
        ])

        overview = pd.DataFrame({
            "Column": data.columns,
            "Data type": data.dtypes.astype(str).values,
            "Non-null": data.notna().sum().values,
            "Nulls": missing_by_column.values,
            "Blank strings": blank_by_column.values,
            "Unique values": unique_counts.values,
            "Missing %": (missing_by_column / row_count * 100).values,
        })
        add_table_page(pdf, "Column Data-Quality Overview", overview)

        if numeric_columns:
            numeric_summary = data[numeric_columns].describe().T.reset_index()
            numeric_summary = numeric_summary.rename(
                columns={"index": "Column"})
            add_table_page(
                pdf, "Numeric Descriptive Statistics", numeric_summary)

            profile = pd.DataFrame({
                "Column": numeric_columns,
                "Minimum": data[numeric_columns].min().values,
                "Maximum": data[numeric_columns].max().values,
                "Median": data[numeric_columns].median().values,
                "Q1": data[numeric_columns].quantile(0.25).values,
                "Q3": data[numeric_columns].quantile(0.75).values,
                "Mean": data[numeric_columns].mean().values,
                "Std dev": data[numeric_columns].std().values,
                "Zeros": (data[numeric_columns] == 0).sum().values,
                "Negative values": (data[numeric_columns] < 0).sum().values,
                "Skewness": data[numeric_columns].skew().values,
            })
            add_table_page(
                pdf, "Numeric Ranges, Shape, and Special Values", profile)

            if churn_column:
                group_means = data.groupby(churn_column)[
                    numeric_columns].mean().T.reset_index()
                group_means = group_means.rename(
                    columns={"index": "Feature", 0: "Churn = 0", 1: "Churn = 1"})
                add_table_page(
                    pdf, "Numeric Averages by Churn Group", group_means)

        for column in categorical_columns:
            counts = data[column].fillna("<NULL>").value_counts().head(15)
            add_bar_page(pdf, f"Top Values: {column}", counts, column)
            if churn_column:
                grouped = data.groupby(column, dropna=False)[churn_column].agg(
                    Customers="size", Churned="sum", **{"Churn rate %": "mean"}
                ).reset_index()
                grouped["Churn rate %"] = grouped["Churn rate %"] * 100
                add_table_page(pdf, f"Churn Rate Detail: {column}", grouped)
                add_churn_rate_page(
                    pdf, column, grouped[[column, "Churn rate %"]])

        if churn_column:
            churn_summary = data[churn_column].value_counts(dropna=False).rename_axis(
                "Churn value"
            ).reset_index(name="Customers")
            churn_summary["Percentage"] = churn_summary["Customers"] / \
                row_count * 100
            add_table_page(pdf, "Target Balance", churn_summary)

        id_checks = pd.DataFrame({
            "Check": [
                "Unique CustomerID values",
                "Repeated CustomerID rows",
                "Minimum CustomerID",
                "Maximum CustomerID",
                "IDs form 1..row count",
            ],
            "Result": [
                int(data["CustomerID"].nunique()
                    ) if "CustomerID" in data.columns else "Not available",
                int(row_count - data["CustomerID"].nunique()
                    ) if "CustomerID" in data.columns else "Not available",
                int(data["CustomerID"].min()
                    ) if "CustomerID" in data.columns else "Not available",
                int(data["CustomerID"].max()
                    ) if "CustomerID" in data.columns else "Not available",
                bool(set(data["CustomerID"]) == set(range(1, row_count + 1))
                     ) if "CustomerID" in data.columns else "Not available",
            ],
        })
        add_table_page(pdf, "Customer ID Integrity", id_checks)

        if numeric_columns:
            outlier_rows = []
            for column in numeric_columns:
                values = data[column].dropna()
                first_quartile = values.quantile(0.25)
                third_quartile = values.quantile(0.75)
                iqr = third_quartile - first_quartile
                lower_bound = first_quartile - 1.5 * iqr
                upper_bound = third_quartile + 1.5 * iqr
                outlier_count = int(
                    ((values < lower_bound) | (values > upper_bound)).sum())
                outlier_rows.append(
                    [column, lower_bound, upper_bound, outlier_count])
            outliers = pd.DataFrame(outlier_rows, columns=[
                                    "Column", "Lower bound", "Upper bound", "IQR outlier count"])
            add_table_page(
                pdf, "Potential Numeric Outliers (IQR Rule)", outliers)

            correlation = data[numeric_columns].corr()
            figure, axis = plt.subplots(figsize=(9, 7))
            image = axis.imshow(correlation, cmap="RdYlBu", vmin=-1, vmax=1)
            axis.set_title("Numeric Feature Correlations",
                           fontsize=18, fontweight="bold", color="#17324D")
            axis.set_xticks(range(len(numeric_columns)),
                            numeric_columns, rotation=45, ha="right")
            axis.set_yticks(range(len(numeric_columns)), numeric_columns)
            figure.colorbar(image, ax=axis, shrink=0.8)
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)

            if churn_column:
                churn_correlations = correlation[churn_column].drop(
                    churn_column).abs().sort_values(ascending=False)
                correlation_table = pd.DataFrame({
                    "Feature": churn_correlations.index,
                    "Absolute correlation with Churn": churn_correlations.values,
                    "Signed correlation": correlation.loc[churn_correlations.index, churn_column].values,
                })
                add_table_page(
                    pdf, "Features Most Associated with Churn", correlation_table)

        add_text_page(pdf, "Exploration Findings", [
            f"The dataset has {row_count:,} rows and {column_count} columns.",
            f"Total null values: {int(missing_by_column.sum()):,}.",
            f"Total blank-string values: {int(blank_by_column.sum()):,}.",
            f"Duplicate rows found: {duplicate_rows:,}.",
            f"Target column check: {'Churn is present.' if 'Churn' in data.columns else 'No Churn column was found.'}",
            f"CustomerID integrity: {'IDs are unique and sequential.' if 'CustomerID' in data.columns and data['CustomerID'].is_unique and set(data['CustomerID']) == set(range(1, row_count + 1)) else 'Review the Customer ID integrity page.'}",
            "Review the distribution pages for imbalances and the IQR page for values that may require domain validation.",
            "Churn-rate pages compare the target outcome across categorical segments; correlation is association, not causation.",
            "This report is descriptive: it does not remove, impute, or modify any source data.",
        ])


def main() -> None:
    csv_path = find_csv()
    data = pd.read_csv(csv_path)
    build_report(data, csv_path)
    print(f"Report created: {REPORT_PATH}")


if __name__ == "__main__":
    main()
