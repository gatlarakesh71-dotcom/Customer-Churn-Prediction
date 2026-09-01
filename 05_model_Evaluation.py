"""Step 5: compare Decision Tree and Random Forest models and save results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from config import DATA_CLEANED, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = Path(DATA_CLEANED)
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_model(model_name: str):
    """Create the preprocessing + model pipeline."""
    data = pd.read_csv(INPUT_PATH)
    X = data.drop(columns=["Churn", "CustomerID"], errors="ignore")
    y = data["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_train.select_dtypes(
        exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )

    if model_name == "decision_tree":
        classifier = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)
    train_predictions = pipeline.predict(X_train)
    validation_predictions = pipeline.predict(X_val)
    validation_probabilities = pipeline.predict_proba(X_val)[:, 1]
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    score_table = pd.DataFrame({
        "Metric": ["Train Score", "Validation Score", "Test Score"],
        "Accuracy": [
            accuracy_score(y_train, train_predictions),
            accuracy_score(y_val, validation_predictions),
            accuracy_score(y_test, predictions),
        ],
        "Precision": [
            precision_score(y_train, train_predictions, zero_division=0),
            precision_score(y_val, validation_predictions, zero_division=0),
            precision_score(y_test, predictions, zero_division=0),
        ],
        "Recall": [
            recall_score(y_train, train_predictions, zero_division=0),
            recall_score(y_val, validation_predictions, zero_division=0),
            recall_score(y_test, predictions, zero_division=0),
        ],
        "F1_Score": [
            f1_score(y_train, train_predictions, zero_division=0),
            f1_score(y_val, validation_predictions, zero_division=0),
            f1_score(y_test, predictions, zero_division=0),
        ],
        "ROC_AUC": [
            roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1]),
            roc_auc_score(y_val, validation_probabilities),
            roc_auc_score(y_test, probabilities),
        ],
    })

    print(f"\n{model_name.upper()} - Training / Validation / Test Summary")
    print(score_table.to_string(index=False))

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1_Score": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
    }

    return metrics


def save_visualizations(comparison_df: pd.DataFrame) -> None:
    """Create comparison charts and save them to the output folder."""
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    chart_df = comparison_df[["Model"] + metrics].copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    x_positions = list(range(len(chart_df)))
    bar_width = 0.15

    for idx, metric in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2) * bar_width
        values = chart_df[metric].tolist()
        ax.bar(
            [pos + offset for pos in x_positions],
            values,
            width=bar_width,
            label=metric,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(chart_df["Model"])
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison_metrics.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    f1_scores = comparison_df["F1_Score"].tolist()
    model_names = comparison_df["Model"].tolist()
    colors = ["#4c72b0", "#55a868"]
    bars = ax.bar(model_names, f1_scores, color=colors)
    ax.set_title("F1-Score Comparison")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.1)

    for bar, value in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_f1_score_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {INPUT_PATH}")

    results = []
    results.append(build_model("decision_tree"))
    results.append(build_model("random_forest"))

    comparison_df = pd.DataFrame(results)
    save_visualizations(comparison_df)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    winner = comparison_df.sort_values("F1_Score", ascending=False).iloc[0]
    print("\nModel Comparison Results")
    print(comparison_df.to_string(index=False))
    print(f"\nBest model by F1-score: {winner['Model']}")
    print(f"Saved comparison results and graphs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
