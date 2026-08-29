"""Step 3: train and evaluate a Decision Tree churn classifier."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from config import (
    DATA_CLEANED,
    DECISION_TREE_MAX_DEPTH,
    DECISION_TREE_MIN_SAMPLES_LEAF,
    DECISION_TREE_MODEL,
    ML_OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = Path(DATA_CLEANED)
OUTPUT_DIR = ML_OUTPUT_DIR


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_PATH)

    if "Churn" not in data.columns:
        raise ValueError("The dataset must contain a 'Churn' target column.")

    X = data.drop(columns=["Churn", "CustomerID"], errors="ignore")
    y = data["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
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

    model = Pipeline([
        ("preprocessing", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=DECISION_TREE_MAX_DEPTH,
                min_samples_leaf=DECISION_TREE_MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
            ),
        ),
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions)
    metrics = {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "True negatives",
            "False positives",
            "False negatives",
            "True positives",
        ],
        "Value": [
            accuracy_score(y_test, predictions),
            precision_score(y_test, predictions, zero_division=0),
            recall_score(y_test, predictions, zero_division=0),
            f1_score(y_test, predictions, zero_division=0),
            roc_auc_score(y_test, probabilities),
            matrix[0, 0],
            matrix[0, 1],
            matrix[1, 0],
            matrix[1, 1],
        ],
    }
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(OUTPUT_DIR / "decision_tree_metrics.csv", index=False)

    report = classification_report(y_test, predictions, digits=4)
    (OUTPUT_DIR / "decision_tree_classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No churn", "Churn"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Decision Tree Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_tree_confusion_matrix.png", dpi=150)
    plt.close()

    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessing"].get_feature_names_out()
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": classifier.feature_importances_,
    }).sort_values("Importance", ascending=False)
    importance.to_csv(
        OUTPUT_DIR / "decision_tree_feature_importance.csv", index=False)

    tree_rules = export_text(
        classifier,
        feature_names=list(feature_names),
        decimals=2,
    )
    (OUTPUT_DIR / "decision_tree_rules.txt").write_text(
        tree_rules,
        encoding="utf-8",
    )

    dump(model, DECISION_TREE_MODEL)

    print("Decision Tree Classifier Results")
    print(metrics_frame.to_string(index=False))
    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
