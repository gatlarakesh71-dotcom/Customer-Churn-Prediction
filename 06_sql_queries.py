"""Local SQL queries for the churn dataset using SQLite.

This script loads the cleaned CSV into a temporary SQLite database,
then runs SQL queries and prints the results in the terminal.
It is designed for quick data exploration without requiring a server.
"""

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")


PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "output" / "customer_churn_cleaned.csv"


def load_data_to_sqlite(csv_path: Path) -> sqlite3.Connection:
    """Load the cleaned CSV into an in-memory SQLite database."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(":memory:")
    df.to_sql("customer_churn", conn, index=False, if_exists="replace")
    return conn


def run_query(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame."""
    return pd.read_sql_query(query, conn)


def print_query_result(title: str, query: str, conn: sqlite3.Connection) -> None:
    """Print SQL query title and output in the terminal."""
    print(f"\n=== {title} ===")
    result = run_query(conn, query)
    if result.empty:
        print("No results found.")
    else:
        print(result.to_string(index=False))


def get_query_list() -> list[tuple[int, str, str]]:
    """Return the list of SQL queries with their titles."""
    return [
        (1, "Total number of customers", """
        SELECT COUNT(*) AS total_customers
        FROM customer_churn;
        """),
        (2, "Churn vs Non-Churn count groups by Churn (0 or 1)", """
        SELECT Churn,
               COUNT(*) AS customer_count
        FROM customer_churn
        GROUP BY Churn
        ORDER BY Churn;
        """),
        (3, "Average age of churned vs non-churned (avg, min, max age by group)", """
        SELECT Churn,
               ROUND(AVG(Age), 2) AS avg_age,
               MIN(Age) AS min_age,
               MAX(Age) AS max_age
        FROM customer_churn
        GROUP BY Churn
        ORDER BY Churn;
        """),
        (4, "Churn rate by Subscription Type (highest churn rate first)", """
        SELECT `Subscription Type` AS subscription_type,
               COUNT(*) AS total_customers,
               SUM(Churn) AS churn_customers,
               ROUND(AVG(Churn) * 100, 2) AS churn_rate_percent
        FROM customer_churn
        GROUP BY `Subscription Type`
        ORDER BY churn_rate_percent DESC;
        """),
        (5, "Churn rate by Contract Length (highest churn rate first)", """
        SELECT `Contract Length` AS contract_length,
               COUNT(*) AS total_customers,
               SUM(Churn) AS churn_customers,
               ROUND(AVG(Churn) * 100, 2) AS churn_rate_percent
        FROM customer_churn
        GROUP BY `Contract Length`
        ORDER BY churn_rate_percent DESC;
        """),
        (6, "Average Total Spend by Churn status (avg, min, max spend by group)", """
        SELECT Churn,
               ROUND(AVG(`Total Spend`), 2) AS avg_total_spend,
               MIN(`Total Spend`) AS min_total_spend,
               MAX(`Total Spend`) AS max_total_spend
        FROM customer_churn
        GROUP BY Churn
        ORDER BY Churn;
        """),
        (7, "Top 10 highest spending customers", """
        SELECT CustomerID,
               Age,
               Gender,
               `Subscription Type`,
               `Contract Length`,
               `Total Spend`,
               Churn
        FROM customer_churn
        ORDER BY `Total Spend` DESC
        LIMIT 10;
        """),
        (8, "Average Support Calls by Churn status (avg, min, max calls by group)", """
        SELECT Churn,
               ROUND(AVG(`Support Calls`), 2) AS avg_support_calls,
               MIN(`Support Calls`) AS min_support_calls,
               MAX(`Support Calls`) AS max_support_calls
        FROM customer_churn
        GROUP BY Churn
        ORDER BY Churn;
        """),
        (9, "High vs Low support calls - churn rate", """
        SELECT CASE
                 WHEN `Support Calls` > 5 THEN 'High Support Calls (>5)'
                 ELSE 'Low Support Calls (<=5)'
               END AS support_call_group,
               COUNT(*) AS total_customers,
               SUM(Churn) AS churn_customers,
               ROUND(AVG(Churn) * 100, 2) AS churn_rate_percent
        FROM customer_churn
        GROUP BY CASE
                 WHEN `Support Calls` > 5 THEN 'High Support Calls (>5)'
                 ELSE 'Low Support Calls (<=5)'
               END
        ORDER BY churn_rate_percent DESC;
        """),
        (10, "Usage Frequency buckets and churn rate", """
        SELECT CASE
                 WHEN `Usage Frequency` <= 10 THEN 'Low Usage (<=10)'
                 WHEN `Usage Frequency` <= 20 THEN 'Medium Usage (11-20)'
                 ELSE 'High Usage (>20)'
               END AS usage_bucket,
               COUNT(*) AS total_customers,
               SUM(Churn) AS churn_customers,
               ROUND(AVG(Churn) * 100, 2) AS churn_rate_percent
        FROM customer_churn
        GROUP BY CASE
                 WHEN `Usage Frequency` <= 10 THEN 'Low Usage (<=10)'
                 WHEN `Usage Frequency` <= 20 THEN 'Medium Usage (11-20)'
                 ELSE 'High Usage (>20)'
               END
        ORDER BY churn_rate_percent DESC;
        """),
        (11, "Average Tenure by Churn status (avg, min, max tenure by group)", """
        SELECT Churn,
               ROUND(AVG(Tenure), 2) AS avg_tenure,
               MIN(Tenure) AS min_tenure,
               MAX(Tenure) AS max_tenure
        FROM customer_churn
        GROUP BY Churn
        ORDER BY Churn;
        """),
        (12, "Short vs Long tenure churn rate", """
        SELECT CASE
                 WHEN Tenure < 12 THEN 'Under 12 months'
                 ELSE '12+ months'
               END AS tenure_group,
               COUNT(*) AS total_customers,
               SUM(Churn) AS churn_customers,
               ROUND(AVG(Churn) * 100, 2) AS churn_rate_percent
        FROM customer_churn
        GROUP BY CASE
                 WHEN Tenure < 12 THEN 'Under 12 months'
                 ELSE '12+ months'
               END
        ORDER BY churn_rate_percent DESC;
        """),
        (13, "Churn rate by Gender", """
        SELECT Gender,
               COUNT(*) AS total_customers,
               SUM(Churn) AS churn_customers,
               ROUND(AVG(Churn) * 100, 2) AS churn_rate_percent
        FROM customer_churn
        GROUP BY Gender
        ORDER BY churn_rate_percent DESC;
        """),
        (14, "Average Payment Delay by Churn status (avg, min, max delay by group)", """
        SELECT Churn,
               ROUND(AVG(`Payment Delay`), 2) AS avg_payment_delay,
               MIN(`Payment Delay`) AS min_payment_delay,
               MAX(`Payment Delay`) AS max_payment_delay
        FROM customer_churn
        GROUP BY Churn
        ORDER BY Churn;
        """),
        (15, "High-risk customer segment (high support calls + low tenure + high payment delay)", """
        SELECT CustomerID,
               Gender,
               Age,
               Tenure,
               `Support Calls`,
               `Payment Delay`,
               `Subscription Type`,
               `Contract Length`,
               `Total Spend`,
               Churn
        FROM customer_churn
        WHERE `Support Calls` > 5
          AND Tenure < 12
          AND `Payment Delay` > 20
        ORDER BY `Payment Delay` DESC, `Support Calls` DESC, Tenure ASC;
        """),
    ]


def export_all_queries_to_pdf(conn: sqlite3.Connection, output_path: Path) -> Path:
    """Save all 15 SQL query results into a multipage PDF report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    queries = get_query_list()

    with PdfPages(output_path) as pdf:
        for query_num, title, query in queries:
            result = run_query(conn, query)
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f"{query_num}. {title}", fontsize=14, y=0.97)
            ax = fig.add_axes([0.05, 0.08, 0.90, 0.82])
            ax.axis("off")

            if result.empty:
                ax.text(0.5, 0.5, "No results found.",
                        ha="center", va="center", fontsize=12)
            else:
                table_data = result.copy()
                table = ax.table(
                    cellText=table_data.values,
                    colLabels=table_data.columns,
                    loc="center",
                    cellLoc="center",
                    colLoc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.8)

            pdf.savefig(fig)
            plt.close(fig)

    return output_path


def run_selected_queries(conn: sqlite3.Connection, selected: str) -> None:
    """Run either a single query number or all queries."""
    queries = get_query_list()
    selected = str(selected).strip().lower()

    if selected in {"all", "a"}:
        for query_num, title, query in queries:
            print_query_result(f"{query_num}. {title}", query, conn)
        return

    try:
        query_num = int(selected)
    except ValueError:
        print("Invalid selection. Choose a number from 1 to 15 or type 'all'.")
        return

    for item_num, title, query in queries:
        if item_num == query_num:
            print_query_result(f"{item_num}. {title}", query, conn)
            return

    print(f"Query {query_num} does not exist. Choose a number from 1 to 15.")


def main() -> None:
    conn = load_data_to_sqlite(CSV_PATH)
    pdf_path = PROJECT_DIR / "output" / "sql_queries_report.pdf"

    print("Customer Churn SQL Query Workspace")
    print(f"Dataset: {CSV_PATH}")
    print(f"PDF report: {pdf_path}")
    print("Usage: python 06_sql_queries.py [1-15 | all | pdf]")
    print("Example: python 06_sql_queries.py 5")

    requested = sys.argv[1] if len(sys.argv) > 1 else "all"
    requested_lower = requested.lower()

    if requested_lower in {"help", "-h", "--help"}:
        print("Choose one query number to view one result in the terminal, or type 'all' to print every query, or 'pdf' to export all 15 queries to a PDF report.")
        conn.close()
        return

    if requested_lower in {"pdf", "report"}:
        saved_pdf = export_all_queries_to_pdf(conn, pdf_path)
        print(f"\nAll 15 SQL query results were exported to: {saved_pdf}")
        conn.close()
        return

    run_selected_queries(conn, requested)
    conn.close()
    print("\nReady for your next SQL question.")


if __name__ == "__main__":
    main()
